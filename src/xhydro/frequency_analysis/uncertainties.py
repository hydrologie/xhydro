"""
Uncertainty analysis module for hydrological frequency analysis.

This module provides functions for bootstrap sampling, distribution fitting,
and uncertainty estimation in regional frequency analysis. It includes tools
for generating bootstrap samples from observed data and fitted distributions,
calculating L-moments, and estimating quantiles with uncertainty bounds.

Functions:
    bootstrap_obs: Generate bootstrap samples from observed data.
    bootstrap_dist: Generate bootstrap samples from a fitted distribution.
    fit_boot_dist: Fit distributions to bootstrap samples.
    calc_moments_iter: Calculate L-moments for each bootstrap sample.
    calc_q_iter: Calculate quantiles for each bootstrap sample and group.
    generate_combinations: Generate combinations of indices for sensitivity analysis.
"""

import warnings
from itertools import combinations

import numpy as np
import xarray as xr
import xclim
from scipy import stats

import xhydro.frequency_analysis as xhfa

from .regional import (
    calc_moments,
    calculate_return_period,
    remove_small_regions,
)

__all__ = [
    "bootstrap_dist",
    "bootstrap_obs",
    "calc_moments_iter",
    "calc_q_iter",
    "fit_boot_dist",
    "generate_combinations",
]


def bootstrap_obs(
    obs: xr.DataArray, *, n_samples: int, seed: int | None = None
) -> xr.DataArray:
    """
    Generate bootstrap samples from observed data.

    Parameters
    ----------
    obs : xarray.DataArray
        The observed data to bootstrap.
    n_samples : int
        The number of bootstrap samples to generate.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    xarray.DataArray
        Bootstrap samples with dimensions [samples, time].
    """

    def _gen_boot(f: np.array, n_samples: int, seed: int | None = None) -> np.ndarray:
        vals = f[~np.isnan(f)]
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(vals, size=(n_samples, len(vals)))
        frep = np.dstack([f] * n_samples)[0].T
        frep[~np.isnan(frep)] = idx.flatten()
        return frep

    return xr.apply_ufunc(
        _gen_boot,
        obs,
        n_samples,
        seed,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["samples", "time"]],
        vectorize=True,
    ).assign_coords(samples=range(n_samples))


def bootstrap_dist(
    ds_obs: xr.Dataset, ds_params: xr.Dataset, *, n_samples: int
) -> xr.Dataset:
    """
    Generate bootstrap samples from a fitted distribution.

    Parameters
    ----------
    ds_obs : xarray.Dataset
        The observed data.
    ds_params : xarray.Dataset
        The fitted distribution parameters.
    n_samples : int
        The number of bootstrap samples to generate.

    Returns
    -------
    xarray.Dataset
        Bootstrap samples with dimensions [samples, time].

    Notes
    -----
    This function does not support lazy evaluation.
    """

    def _calc_rvs(data, params, dist, p_names, n_samples):
        dist_obj = xclim.indices.stats.get_dist(dist)
        shape_params = [] if dist_obj.shapes is None else dist_obj.shapes.split(",")
        dist_params = shape_params + ["loc", "scale"]
        data_length = len(data)
        p_names = p_names[~np.isnan(params)]
        params = params[~np.isnan(params)]
        ordered_params = [params[d == p_names] for d in dist_params]

        samples = getattr(stats, dist).rvs(
            *ordered_params, size=(n_samples, data_length)
        )
        samples[:, np.isnan(data)] = np.nan
        return samples

    ds = xr.apply_ufunc(
        _calc_rvs,
        ds_obs.load(),
        ds_params.load(),
        ds_params.scipy_dist,
        ds_params.dparams,
        n_samples,
        input_core_dims=[["time"], ["dparams"], [], ["dparams"], []],
        output_core_dims=[["samples", "time"]],
        vectorize=True,
    ).assign_coords(samples=range(n_samples))
    return ds


def fit_boot_dist(ds: xr.Dataset) -> xr.Dataset:
    """
    Fit distributions to bootstrap samples.

    Parameters
    ----------
    ds : xarray.Dataset
        The bootstrap samples to fit.

    Returns
    -------
    xarray.Dataset
        Fitted distribution parameters for each bootstrap sample.
    """
    # we can't use xhydro fit as each dist would be fitted over each dist, resulting in nxn fits
    params_ince = []
    for dist in ds.scipy_dist.values:
        ds.sel(scipy_dist=dist)
        params_ince.append(
            xhfa.local.fit(ds.sel(scipy_dist=dist), distributions=[dist])
        )
    return xr.concat(params_ince, dim="scipy_dist")


def calc_moments_iter(ds_samples: xr.Dataset) -> xr.Dataset:
    """
    Calculate L-moments for each bootstrap sample.

    Parameters
    ----------
    ds_samples : xarray.Dataset
        The bootstrap samples.

    Returns
    -------
    xarray.Dataset
        L-moments for each bootstrap sample.
    """
    ds_mom = []
    for sample in ds_samples.samples.values:
        ds = ds_samples.sel(samples=sample)
        ds_mom.append(calc_moments(ds))
    return xr.concat(ds_mom, dim="samples")


def _calc_q_iter_da(
    bv: str,
    da_groups: xr.DataArray,
    da_moments_iter: xr.DataArray,
    *,
    return_period: np.array,
    small_regions_threshold: int | None = 5,
    l1: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Calculate quantiles for each bootstrap sample and group.

    Parameters
    ----------
    bv : str
        The basin identifier or 'all' to proceed on all basins (needed for ungauged).
        The associated dimension must have a 'cf_role: timeseries_id' attribute.
    da_groups: xr.DataArray,
        The grouped data.
    da_moments_iter: xr.DataArray
        The L-moments for each bootstrap sample.
    return_period : array-like
        The return periods to calculate quantiles for.
    small_regions_threshold : int, optional
        The threshold for removing small regions. Default is 5.
    l1 : xr.DataArray, optional
        First L-moment (location) values. L-moment can be specified for ungauged catchments.
        If `None`, values are taken from ds_moments_iter.

    Returns
    -------
    xarray.DataArray
        Quantiles for each bootstrap sample and group.
    """
    # We select groups for all or one id
    id_dim = da_groups.cf.cf_roles["timeseries_id"][0]
    if bv == "all":
        ds_temp = da_groups.dropna("region_id", how="all")
    else:
        ds_temp = da_groups.sel(**{id_dim: bv}).dropna("region_id", how="all")
    ds_mom = []

    # For each group, we find which id are in it
    for region_id in ds_temp.region_id.values:
        id_list = da_groups.sel(region_id=region_id).dropna(id_dim, how="all").id.values
        # We use moments with ressample previously done, and we create ds_moment_group with iterations
        ds_mom.append(
            da_moments_iter.sel(**{id_dim: id_list})
            .assign_coords(region_id=region_id)
            .expand_dims("region_id")
        )

    # Concat along region_id
    ds_moments_groups = xr.concat(ds_mom, dim="region_id")
    da_groups = da_groups.sel(region_id=ds_moments_groups.region_id).dropna(
        dim="id", how="all"
    )
    # With obs and moments  of same dims, we calculate
    qt = calculate_return_period(
        da_groups.to_dataset(),
        ds_moments_groups.to_dataset(),
        return_period=return_period,
        l1=l1,
    )
    qt = remove_small_regions(qt, thresh=small_regions_threshold)
    # For each station we stack regions et bootstrap
    if bv == "all":
        return (
            qt.rename({"samples": "obs_samples"})
            .stack(samples=["region_id", "obs_samples"])
            .to_dataarray()
            .squeeze()
        )
    else:
        return (
            qt.rename({"samples": "obs_samples"})
            .stack(samples=["region_id", "obs_samples"])
            .sel(**{id_dim: bv})
            .to_dataarray()
            .squeeze()
        )


def calc_q_iter(
    bv: str,
    groups: xr.DataArray | xr.Dataset,
    moments_iter: xr.DataArray | xr.Dataset,
    return_period: np.array,
    small_regions_threshold: int | None = 5,
    l1: xr.DataArray | None = None,
    return_periods: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Calculate quantiles for each bootstrap sample and group.

    Parameters
    ----------
    bv : str
        The basin identifier or 'all' to proceed on all basins (needed for ungauged).
        The associated dimension must have a 'cf_role: timeseries_id' attribute.
    groups : xr.DataArray or xr.Dataset
        The grouped data.
    moments_iter : xr.DataArray or xr.Dataset
        The L-moments for each bootstrap sample.
    return_period : array-like
        The return periods to calculate quantiles for.
    small_regions_threshold : int, optional
        The threshold for removing small regions. Default is 5.
    l1 : xr.DataArray, optional
        First L-moment (location) values. L-moment can be specified for ungauged catchments.
        If `None`, values are taken from ds_moments_iter.
    return_periods :  float or list of float
        Kept as an option for retrocompatibility, defaulting it to None when return_period exists.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Quantiles for each bootstrap sample and group. Returns a Dataset if input groups
        and moments_iter are Datasets, otherwise returns a DataArray.
    """
    warnings.warn(
        "This function is deprecated and will be removed in xhydro v0.7.0. Use calculate_quantiles_over_boostraped_groups instead.",
        FutureWarning,
    )
    return calculate_quantiles_over_boostraped_groups(
        bv,
        groups,
        moments_iter,
        return_period,
        small_regions_threshold,
        l1,
        return_periods,
    )


def calculate_quantiles_over_boostraped_groups(
    bv: str,
    groups: xr.DataArray | xr.Dataset,
    moments_iter: xr.DataArray | xr.Dataset,
    return_period: np.array,
    small_regions_threshold: int | None = 5,
    l1: xr.DataArray | None = None,
    return_periods: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Calculate quantiles for each bootstrap sample and group.

    Parameters
    ----------
    bv : str
        The basin identifier or 'all' to proceed on all basins (needed for ungauged).
        The associated dimension must have a 'cf_role: timeseries_id' attribute.
    groups : xr.DataArray or xr.Dataset
        The grouped data.
    moments_iter : xr.DataArray or xr.Dataset
        The L-moments for each bootstrap sample.
    return_period : array-like
        The return periods to calculate quantiles for.
    small_regions_threshold : int, optional
        The threshold for removing small regions. Default is 5.
    l1 : xr.DataArray, optional
        First L-moment (location) values. L-moment can be specified for ungauged catchments.
        If `None`, values are taken from ds_moments_iter.
    return_periods :  float or list of float
        Kept as an option for retrocompatibility, defaulting it to None when return_period exists.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Quantiles for each bootstrap sample and group. Returns a Dataset if input groups
        and moments_iter are Datasets, otherwise returns a DataArray.
    """
    if return_periods is not None:
        warnings.warn(
            "The 'return_periods' parameter has been renamed to 'return_period' and will be dropped in xHydro v0.7.0.",
            FutureWarning,
        )
        return_period = return_periods
    if all(isinstance(input, xr.DataArray) for input in [groups, moments_iter]):
        ds = False
    elif all(isinstance(input, xr.Dataset) for input in [groups, moments_iter]):
        ds = True
    else:
        raise TypeError(
            "groups and moments_iter must be both xr.DataArray or xr.Dataset"
        )

    if ds:
        ds = xr.Dataset()
        groups_var = list(groups.keys())
        if groups_var != list(moments_iter.keys()):
            raise Exception("Variables in groups and moments_iter must be the same")
        for var in groups_var:
            ds[var] = _calc_q_iter_da(
                bv,
                groups[var],
                moments_iter[var],
                return_period=return_period,
                small_regions_threshold=small_regions_threshold,
                l1=l1,
            ).expand_dims("id")
        return ds
    else:
        return _calc_q_iter_da(
            bv,
            groups,
            moments_iter,
            return_period=return_period,
            small_regions_threshold=small_regions_threshold,
            l1=l1,
        ).expand_dims("id")


def generate_combinations(da: xr.DataArray, *, n: int) -> list:
    """
    Generate combinations of indices omitting up to N indices.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray.
    n : int
        Number of indices to omit in each combination.

    Returns
    -------
    list
        List of all combinations.
    """
    # Get the list of indices
    for ids in da.indexes:
        if ids != "components":
            indices = da[ids].values

    # Variable to store all combinations
    all_combinations = []

    # Generate combinations for each case
    for i in range(0, n + 1):
        # Generate combinations of indices omitting i indices
        index_combinations = list(combinations(indices, len(indices) - i))

        # Add each combination to the variable
        all_combinations.extend(index_combinations)

    return all_combinations
