"""Module to compute climate change statistics using xscen functions."""

from datetime import datetime

import numpy as np
import xarray as xr

# Special imports from xscen
from xscen import (
    clean_up,
    climatological_op,
    compute_deltas,
    ensemble_stats,
    produce_horizon,
)

__all__ = [
    "climatological_op",
    "compute_deltas",
    "ensemble_stats",
    "produce_horizon",
    "sampled_indicators",
    "weighted_random_sampling",
]


def weighted_random_sampling(
    ds: xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    include_dims: list[str] | None = None,
    n: int = 5000,
    seed: int | None = None,
) -> xr.Dataset:
    """Sample from a dataset using random sampling.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to sample from.
    weights : xr.DataArray, optional
        Weights to use when sampling the dataset, for dimensions other than 'percentile' or 'quantile'.
        See Notes for more information on special cases for 'percentile', 'quantile', 'time', and 'horizon' dimensions.
    include_dims : list of str, optional
        List of dimensions to include when sampling the dataset, in addition to the 'percentile' or 'quantile' dimensions and those with weights.
        These dimensions will be sampled uniformly.
    n : int
        Number of samples to generate.
    seed : int, optional
        Seed to use for the random number generator.

    Returns
    -------
    xr.Dataset
        Dataset containing the 'n' samples, stacked along a 'sample' dimension.

    Notes
    -----
    If the dataset contains a "percentile" [0, 100] or "quantile" [0, 1] dimension, the percentiles will be sampled accordingly as to
    account for the uneven spacing between percentiles and maintain the distribution's shape.

    Weights along the 'time' or 'horizon' dimensions are supported, but behave differently than other dimensions. They will not
    be stacked alongside other dimensions in the new 'sample' dimension. Rather, a separate sampling will be done for each time/horizon,
    """
    # Prepare weights
    include_dims = include_dims or []

    if weights is not None:
        was_weighted = True
        # Add weights along all dimensions not in 'exclude_dims'
        weights = weights.expand_dims(
            {dim: ds[dim] for dim in include_dims if dim not in weights.dims}
        )
    else:
        was_weighted = False
        # Uniform sampling for all dimensions in 'include_dims'
        weights = xr.DataArray(
            np.ones([ds.sizes[dim] for dim in include_dims]),
            coords={dim: ds[dim] for dim in include_dims},
            dims=include_dims,
        )

    # Add the percentile weights
    if any([dim in ds.dims for dim in ["percentile", "quantile"]]):
        percentile_weights = _percentile_weights(ds)

        weights = (
            percentile_weights.expand_dims({dim: weights[dim] for dim in weights.dims})
            * weights
        )

    if len(weights.dims) == 0:
        raise ValueError("No weights provided for sampling.")

    def _add_sampling_attrs(d, prefix, history, ds_for_attrs=None):
        for var in d.data_vars:
            if ds_for_attrs is not None:
                d[var].attrs = ds_for_attrs[var].attrs
            d[var].attrs["sampling_n"] = n
            d[var].attrs["sampling_seed"] = seed
            d[var].attrs[
                "description"
            ] = f"{prefix} of {d[var].attrs.get('long_name', var)} constructed using random sampling."
            d[var].attrs[
                "long_name"
            ] = f"{prefix} of {d[var].attrs.get('long_name', var)}"
            old_history = d[var].attrs.get("history", "")
            d[var].attrs[
                "history"
            ] = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {history}\n {old_history}"

    # Sample the distributions
    ds_dist = (
        _weighted_sampling(ds, weights, n=n, seed=seed)
        .drop_vars("sample", errors="ignore")
        .assign_coords({"sample": np.arange(n)})
    )
    _add_sampling_attrs(
        ds_dist,
        "Sampled distribution",
        f"{'Weighted' if was_weighted else 'Uniform'} random sampling "
        f"applied using 'xhydro.sampled_indicators' and {n} samples.",
        ds_for_attrs=ds,
    )

    return ds_dist


def sampled_indicators(  # noqa: C901
    ds_dist: xr.Dataset,
    deltas_dist: xr.Dataset,
    *,
    delta_kind: str | dict | None = None,
    percentiles: xr.DataArray | xr.Dataset | list[int] | np.ndarray | None = None,
) -> xr.Dataset | tuple[xr.Dataset, xr.Dataset]:
    """Reconstruct indicators using a perturbation approach and random sampling.

    Parameters
    ----------
    ds_dist : xr.Dataset
        Dataset containing the sampled reference distribution, stacked along a 'sample' dimension.
        Typically generated using 'xhydro.cc.weighted_random_sampling'.
    deltas_dist : xr.Dataset
        Dataset containing the sampled deltas, stacked along a 'sample' dimension.
        Typically generated using 'xhydro.cc.weighted_random_sampling'.
    delta_kind : str or dict, optional
        Type of delta provided. Recognized values are: ['absolute', 'abs.', '+'], ['percentage', 'pct.', '%'].
        If a dict is provided, it should map the variable names to their respective delta type.
        If None, the variables should have a 'delta_kind' attribute.
    percentiles : xr.DataArray or xr.Dataset or list or np.ndarray, optional
        Percentiles to compute in the future distribution.
        This will change the output of the function to a tuple containing the future distribution and the future percentiles.
        If given a Dataset, it should contain a 'percentile' or 'quantile' dimension.
        If given a list or np.ndarray, it should contain percentiles [0, 100].

    Returns
    -------
    fut_dist : xr.Dataset
        The future distribution, stacked along the 'sample' dimension.
    fut_pct : xr.Dataset
        Dataset containing the future percentiles.

    Notes
    -----
    The future percentiles are computed as follows:
    1. Sample 'n' values from the reference distribution.
    2. Sample 'n' values from the delta distribution.
    3. Create the future distribution by applying the sampled deltas to the sampled historical distribution, element-wise.
    4. Compute the percentiles of the future distribution (optional).
    """
    # Prepare the operation to apply on the variables
    if delta_kind is None:
        try:
            delta_kind = {
                var: deltas_dist[var].attrs["delta_kind"]
                for var in deltas_dist.data_vars
            }
        except KeyError:
            raise KeyError(
                "The 'delta_kind' argument is None, but the variables within the 'deltas' Dataset are missing a 'delta_kind' attribute."
            )
    elif isinstance(delta_kind, str):
        delta_kind = {var: delta_kind for var in deltas_dist.data_vars}
    elif isinstance(delta_kind, dict):
        if not all([var in delta_kind for var in deltas_dist.data_vars]):
            raise ValueError(
                f"If 'delta_kind' is a dict, it should contain all the variables in 'deltas'. Missing variables: "
                f"{list(set(deltas_dist.data_vars) - set(delta_kind))}."
            )

    def _add_sampling_attrs(d, prefix, history, ds_for_attrs=None):
        for var in d.data_vars:
            if ds_for_attrs is not None:
                d[var].attrs = ds_for_attrs[var].attrs
            d[var].attrs["delta_kind"] = delta_kind[var]
            var_name = (
                d[var]
                .attrs.get("long_name", var)
                .replace("Sampled distribution of ", "")
            )
            d[var].attrs[
                "description"
            ] = f"{prefix} of {var_name} constructed from a perturbation approach and random sampling."
            d[var].attrs["long_name"] = f"{prefix} of {var_name}"
            old_history = d[var].attrs.get("history", "")
            d[var].attrs[
                "history"
            ] = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {history}\n {old_history}"

    # Apply the deltas element-wise to the historical distribution
    fut_dist = xr.Dataset()
    for var in deltas_dist.data_vars:
        with xr.set_options(keep_attrs=True):
            if delta_kind[var] in ["absolute", "abs.", "+"]:
                fut_dist[var] = ds_dist[var] + deltas_dist[var]
            elif delta_kind[var] in ["percentage", "pct.", "%"]:
                fut_dist[var] = ds_dist[var] * (1 + deltas_dist[var] / 100)
            else:
                raise ValueError(
                    f"Unknown operation '{delta_kind[var]}', expected one of ['absolute', 'abs.', '+'], ['percentage', 'pct.', '%']."
                )
            _add_sampling_attrs(
                fut_dist,
                "Reconstructed distribution",
                f"Reconstructed distribution using a perturbation approach and random sampling.",
                ds_for_attrs=ds_dist,
            )

    # Build the attributes based on common attributes between the historical and delta distributions
    fut_dist = clean_up(fut_dist, common_attrs_only=[ds_dist, deltas_dist])

    # Compute future percentiles
    if percentiles is not None:
        if isinstance(percentiles, list | np.ndarray):
            if isinstance(percentiles, list):
                percentiles = np.array(percentiles)
            pdim = "percentile"
            mult = 100
        else:
            _, pdim, mult = _perc_or_quantile(percentiles)

        with xr.set_options(keep_attrs=True):
            fut_pct = fut_dist.quantile(percentiles / mult, dim="sample")
        _add_sampling_attrs(
            fut_pct,
            "Reconstructed percentiles",
            f"Reconstructed percentiles computed from the reconstructeddistribution using 'xhydro.sampled_indicators'.",
            ds_for_attrs=ds_dist,
        )

        if pdim == "percentile":
            fut_pct = fut_pct.rename({"quantile": "percentile"})
            fut_pct["percentile"] = percentiles

        return fut_dist, fut_pct

    return fut_dist


def _percentile_weights(da: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Compute the weights associated with percentiles, with support for unevenly spaced percentiles.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        DataArray or Dataset containing the percentiles to use when sampling.
        The percentiles are expected to be stored in either a dimension called "percentile" [0, 100] or "quantile" [0, 1].

    Returns
    -------
    p : xr.DataArray
        DataArray containing the weights associated with the percentiles.
    """
    pct, pdim, multiplier = _perc_or_quantile(da)

    # Temporarily add a 0 and 100th percentile
    p0 = xr.DataArray([0], coords={pdim: [0]}, dims=[pdim])
    p1 = xr.DataArray([multiplier], coords={pdim: [multiplier]}, dims=[pdim])
    p = xr.concat([p0, pct, p1], dim=pdim)
    p = p.diff(pdim) / 2
    p[0] = (
        p[0] * 2
    )  # The first and last weights need to be doubled to account for the fact that the first and last percentiles are not centered
    p[-1] = p[-1] * 2
    p = p.rolling({pdim: 2}, center=True).sum().shift({pdim: -1})[:-1]

    return p


def _weighted_sampling(
    ds: xr.Dataset, weights: xr.DataArray, n: int = 5000, seed: int | None = None
) -> xr.Dataset:
    """Sample from a distribution with weights.
    In the case of weights on a 'time' or 'horizon' dimension, the sampling is done separately for each time/horizon, but with the same seed.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the distribution to sample from.
    weights : xr.DataArray
        Weights to use when sampling.
    n : int
        Number of samples to generate.
    seed : int, optional
        Seed to use for the random number generator.

    Returns
    -------
    ds_dist : xr.Dataset
        Dataset containing the 'n' samples, stacked along the 'sample' dimension.
    """
    # Prepare the weights
    if weights.isnull().any():
        raise ValueError("The weights contain NaNs.")

    # Weights on the time dimension need to be handled differently
    time_dim = [dim for dim in weights.dims if dim in ["time", "horizon"]]
    if len(time_dim) > 1:
        raise NotImplementedError(
            "Weights on multiple time dimensions are not supported."
        )
    time_dim = time_dim[0] if time_dim else ""
    other_dims = list(set(weights.dims).difference([time_dim]))

    # Must equal 1
    weights = weights / weights.sum(other_dims)

    # For performance reasons, remove the chunking along impacted dimensions
    ds = ds.chunk({dim: -1 for dim in weights.dims})
    weights = weights.chunk({dim: -1 for dim in weights.dims})
    # Load coordinates and weights to avoid issues with dask
    [ds[c].load() for c in ds.coords]
    weights = weights.compute()

    # Stack the dimensions containing weights
    ds = ds.stack({"sample": sorted(other_dims)})
    weights = weights.stack({"sample": sorted(other_dims)})
    weights = weights.reindex_like(ds)
    ds = ds.drop_vars(["sample"] + other_dims).assign_coords(
        {"sample": np.arange(len(ds.sample))}
    )
    weights = weights.drop_vars(["sample"] + other_dims).assign_coords(
        {"sample": np.arange(len(weights.sample))}
    )

    # Sample the dimensions with weights
    rng = np.random.default_rng(seed=seed)

    # Perform the sampling
    if not time_dim:
        idx = rng.choice(len(weights.sample), size=n, p=weights)
        ds_dist = ds.chunk({"sample": -1}).reindex({"sample": idx})
        ds_dist["sample"] = np.arange(len(ds_dist.sample))
        ds_dist = ds_dist.chunk({"sample": -1})

    else:
        ds_dist = []
        for time in ds[time_dim].values:
            idx = rng.choice(
                len(weights.sample), size=n, p=weights.sel({time_dim: time})
            )
            ds_dist_tmp = (
                ds.sel({time_dim: time}).chunk({"sample": -1}).reindex({"sample": idx})
            )
            ds_dist_tmp["sample"] = np.arange(len(ds_dist_tmp.sample))
            ds_dist_tmp = ds_dist_tmp.chunk({"sample": -1})
            ds_dist.append(ds_dist_tmp)
        ds_dist = xr.concat(ds_dist, dim=time_dim)

    return ds_dist


def _perc_or_quantile(da: xr.DataArray | xr.Dataset) -> tuple[xr.DataArray, str, int]:
    """Return 'percentile' or 'quantile' depending on the name of the percentile dimension."""
    if isinstance(da, xr.DataArray):
        if len(da.dims) != 1:
            raise ValueError(
                f"DataArray has more than one dimension: received {da.dims}."
            )
        pdim = str(da.dims[0])
        pct = da
        if pdim not in ["percentile", "quantile"]:
            raise ValueError(
                f"DataArray has no 'percentile' or 'quantile' dimension: received {pdim}."
            )
    else:
        pdim = [dim for dim in da.dims if dim in ["percentile", "quantile"]]
        if len(pdim) != 1:
            raise ValueError(
                "The Dataset should contain one of ['percentile', 'quantile']."
            )
        pdim = pdim[0]
        pct = da[pdim]

    multiplier = 100 if pdim == "percentile" else 1
    if (pct.min() < 0 or pct.max() > multiplier) or (
        pdim == "percentile" and pct.max() <= 1
    ):
        raise ValueError(
            f"The {pdim} values do not seem to be in the [0, {multiplier}] range."
        )

    return pct, pdim, multiplier
