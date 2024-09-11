"""
Uncertainty analysis module for hydrological frequency analysis.

This module provides functions for bootstrap sampling, distribution fitting,
and uncertainty estimation in regional frequency analysis. It includes tools
for generating bootstrap samples from observed data and fitted distributions,
calculating L-moments, and estimating quantiles with uncertainty bounds.

Functions:
    boostrap_obs: Generate bootstrap samples from observed data.
    boostrap_dist: Generate bootstrap samples from a fitted distribution.
    fit_boot_dist: Fit distributions to bootstrap samples.
    calc_moments_iter: Calculate L-moments for each bootstrap sample.
    calc_q_iter: Calculate quantiles for each bootstrap sample and group.
    generate_combinations: Generate combinations of indices for sensitivity analysis.
"""

from itertools import combinations
from typing import Optional

import numpy as np
import xarray as xr
import xclim
from scipy import stats

import xhydro.frequency_analysis as xhfa

from .regional import calc_moments, calculate_rp_from_afr, remove_small_regions


def boostrap_obs(
    obs: xr.DataArray, n_samples: int, seed: int | None = None
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


def boostrap_dist(
    ds_obs: xr.Dataset, ds_params: xr.Dataset, n_samples: int
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


def calc_q_iter(
    bv: str,
    var: str,
    ds_groups: xr.Dataset,
    ds_moments_iter: xr.Dataset,
    return_periods: np.array,
    small_regions_threshold: int | None = 5,
) -> xr.DataArray:
    """
    Calculate quantiles for each bootstrap sample and group.

    Parameters
    ----------
    bv : str
        The basin identifier.
    var : str
        The variable name.
    ds_groups : xarray.Dataset
        The grouped data.
    ds_moments_iter : xarray.Dataset
        The L-moments for each bootstrap sample.
    return_periods : array-like
        The return periods to calculate quantiles for.
    small_regions_threshold : int, optional
        The threshold for removing small regions. Default is 5.

    Returns
    -------
    xarray.DataArray
        Quantiles for each bootstrap sample and group.
    """
    # We select groups for only one id
    ds_temp = ds_groups[[var]].sel(id=bv).dropna("group_id", "all")
    ds_mom = []

    # For each group, we find which id are in it
    for group_id in ds_temp.group_id.values:
        id_list = ds_groups.sel(group_id=group_id).dropna("id", how="all").id.values
        # We use moments with ressample previously done, and we create ds_moment_group with iterations
        ds_mom.append(
            ds_moments_iter[[var]]
            .sel(id=id_list)
            .assign_coords(group_id=group_id)
            .expand_dims("group_id")
        )

    # Concat along group_id
    ds_moments_groups = xr.concat(ds_mom, dim="group_id")
    ds_groups = (
        ds_groups[[var]]
        .sel(group_id=ds_moments_groups.group_id.values)
        .dropna(dim="id", how="all")
    )
    # With obs and moments  of same dims, we calculate
    qt = calculate_rp_from_afr(ds_groups, ds_moments_groups, return_periods)
    qt = remove_small_regions(qt, thresh=small_regions_threshold)
    # For each station we stack regions et bootstrap
    return (
        qt.rename({"samples": "obs_samples"})
        .stack(samples=["group_id", "obs_samples"])
        .squeeze()
    )


def generate_combinations(da: xr.DataArray, n: int) -> list:
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
