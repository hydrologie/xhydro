"""Module to compute climate change statistics using xscen functions."""

from datetime import datetime

import numpy as np
import xarray as xr
import xscen as xs

# Special imports from xscen
from xscen import climatological_op, compute_deltas, ensemble_stats, produce_horizon

__all__ = [
    "climatological_op",
    "compute_deltas",
    "ensemble_stats",
    "produce_horizon",
    "sampled_indicators",
]


def sampled_indicators(  # noqa: C901
    ds: xr.Dataset,
    deltas: xr.Dataset,
    *,
    delta_kind: str | dict | None = None,
    ds_weights: xr.DataArray | None = None,
    delta_weights: xr.DataArray | None = None,
    n: int = 50000,
    seed: int | None = None,
    return_dist: bool = False,
) -> xr.Dataset | tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """Compute future indicators using a perturbation approach and random sampling.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the historical indicators. The indicators are expected to be represented by a distribution of pre-computed percentiles.
        The percentiles should be stored in either a dimension called "percentile" [0, 100] or "quantile" [0, 1].
    deltas : xr.Dataset
        Dataset containing the future deltas to apply to the historical indicators.
    delta_kind : str or dict, optional
        Type of delta provided. Recognized values are: ['absolute', 'abs.', '+'], ['percentage', 'pct.', '%'].
        If a dict is provided, it should map the variable names to their respective delta type.
        If None, the variables should have a 'delta_kind' attribute.
    ds_weights : xr.DataArray, optional
        Weights to use when sampling the historical indicators, for dimensions other than 'percentile'/'quantile'.
        Dimensions not present in this Dataset, or if None, will be sampled uniformly unless they are shared with 'deltas'.
    delta_weights : xr.DataArray, optional
        Weights to use when sampling the deltas, such as along the 'realization' dimension.
        Dimensions not present in this Dataset, or if None, will be sampled uniformly unless they are shared with 'ds'.
    n : int
        Number of samples to generate.
    seed : int, optional
        Seed to use for the random number generator.
    return_dist : bool
        Whether to return the full distributions (ds, deltas, fut) or only the percentiles.

    Returns
    -------
    fut_pct : xr.Dataset
        Dataset containing the future percentiles.
    ds_dist : xr.Dataset
        The historical distribution, stacked along the 'sample' dimension.
    deltas_dist : xr.Dataset
        The delta distribution, stacked along the 'sample' dimension.
    fut_dist : xr.Dataset
        The future distribution, stacked along the 'sample' dimension.

    Notes
    -----
    Weights along the 'time' or 'horizon' dimensions are supported, but behave differently than other dimensions. They will not
    be stacked alongside other dimensions in the new 'sample' dimension. Rather, a separate sampling will be done for each time/horizon,

    The future percentiles are computed as follows:
    1. Sample 'n' values from the historical distribution, weighting the percentiles by their associated coverage.
    2. Sample 'n' values from the delta distribution, using the provided weights.
    3. Create the future distribution by applying the sampled deltas to the sampled historical distribution, element-wise.
    4. Compute the percentiles of the future distribution.
    """
    # Prepare weights
    shared_dims = set(ds.dims).intersection(set(deltas.dims))
    exclude_dims = ["time", "horizon"]
    percentile_weights = _percentile_weights(ds)
    if ds_weights is not None:
        percentile_weights = (
            percentile_weights.expand_dims(
                {dim: ds_weights[dim] for dim in ds_weights.dims}
            )
            * ds_weights
        )
    percentile_weights = percentile_weights.expand_dims(
        {
            dim: ds[dim]
            for dim in set(ds.dims).difference(
                list(shared_dims) + list(percentile_weights.dims) + exclude_dims
            )
        }
    )
    deltas_weighted = True
    if delta_weights is None:
        deltas_weighted = False
        dims = set(deltas.dims).difference(list(shared_dims) + exclude_dims)
        delta_weights = xr.DataArray(
            np.ones([deltas.sizes[dim] for dim in dims]),
            coords={dim: deltas[dim] for dim in dims},
            dims=dims,
        )
    delta_weights = delta_weights.expand_dims(
        {
            dim: deltas[dim]
            for dim in set(deltas.dims).difference(
                list(shared_dims) + list(delta_weights.dims) + exclude_dims
            )
        }
    )

    unique_dims = set(percentile_weights.dims).symmetric_difference(
        set(delta_weights.dims)
    )
    if any([dim in shared_dims for dim in unique_dims]):
        problem_dims = [dim for dim in unique_dims if dim in shared_dims]
        raise ValueError(
            f"Dimension(s) {problem_dims} is shared between 'ds' and 'deltas', but not between 'ds_weights' and 'delta_weights'."
        )

    # Prepare the operation to apply on the variables
    if delta_kind is None:
        try:
            delta_kind = {
                var: deltas[var].attrs["delta_kind"] for var in deltas.data_vars
            }
        except KeyError:
            raise KeyError(
                "The 'delta_kind' argument is None, but the variables within the 'deltas' Dataset are missing a 'delta_kind' attribute."
            )
    elif isinstance(delta_kind, str):
        delta_kind = {var: delta_kind for var in deltas.data_vars}
    elif isinstance(delta_kind, dict):
        if not all([var in delta_kind for var in deltas.data_vars]):
            raise ValueError(
                f"If 'delta_kind' is a dict, it should contain all the variables in 'deltas'. Missing variables: "
                f"{list(set(deltas.data_vars) - set(delta_kind))}."
            )

    def _add_sampling_attrs(d, prefix, history, ds_for_attrs=None):
        for var in d.data_vars:
            if ds_for_attrs is not None:
                d[var].attrs = ds_for_attrs[var].attrs
            d[var].attrs["sampling_kind"] = delta_kind[var]
            d[var].attrs["sampling_n"] = n
            d[var].attrs["sampling_seed"] = seed
            d[var].attrs[
                "description"
            ] = f"{prefix} of {d[var].attrs.get('long_name', var)} constructed from a perturbation approach and random sampling."
            d[var].attrs[
                "long_name"
            ] = f"{prefix} of {d[var].attrs.get('long_name', var)}"
            old_history = d[var].attrs.get("history", "")
            d[var].attrs[
                "history"
            ] = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {history}\n {old_history}"

    # Sample the distributions
    _, pdim, mult = _perc_or_quantile(ds)
    ds_dist = (
        _weighted_sampling(ds, percentile_weights, n=n, seed=seed)
        .drop_vars("sample", errors="ignore")
        .assign_coords({"sample": np.arange(n)})
    )
    _add_sampling_attrs(
        ds_dist,
        "Historical distribution",
        f"{'Weighted' if ds_weights is not None else 'Uniform'} random sampling "
        f"applied to the historical distribution using 'xhydro.sampled_indicators' "
        f"and {n} samples.",
        ds_for_attrs=ds,
    )

    deltas_dist = (
        _weighted_sampling(deltas, delta_weights, n=n, seed=seed)
        .drop_vars("sample", errors="ignore")
        .assign_coords({"sample": np.arange(n)})
    )
    _add_sampling_attrs(
        deltas_dist,
        "Future delta distribution",
        f"{'Weighted' if deltas_weighted else 'Uniform'} random sampling "
        f"applied to the delta distribution using 'xhydro.sampled_indicators' "
        f"and {n} samples.",
        ds_for_attrs=deltas,
    )

    # Apply the deltas element-wise to the historical distribution
    fut_dist = xr.Dataset()
    for var in deltas.data_vars:
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
                "Future distribution",
                f"Future distribution constructed from a perturbation approach and random sampling with {n} samples.",
                ds_for_attrs=ds,
            )

    # Build the attributes based on common attributes between the historical and delta distributions
    fut_dist = xs.clean_up(fut_dist, common_attrs_only=[ds, deltas])

    # Compute future percentiles
    with xr.set_options(keep_attrs=True):
        fut_pct = fut_dist.quantile(ds.percentile / mult, dim="sample")
    _add_sampling_attrs(
        fut_pct,
        "Future percentiles",
        f"Future percentiles computed from the future distribution using 'xhydro.sampled_indicators' and {n} samples.",
        ds_for_attrs=ds,
    )

    if pdim == "percentile":
        fut_pct = fut_pct.rename({"quantile": "percentile"})
        fut_pct["percentile"] = ds.percentile

    if return_dist:
        return fut_pct, ds_dist, deltas_dist, fut_dist
    else:
        return fut_pct


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
    ds: xr.Dataset, weights: xr.DataArray, n: int = 50000, seed: int | None = None
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

    # Stack the dimensions containing weights
    ds = ds.stack({"sample": sorted(other_dims)})
    weights = weights.stack({"sample": sorted(other_dims)})
    weights = weights.reindex_like(ds)

    # Sample the dimensions with weights
    rng = np.random.default_rng(seed=seed)

    if not time_dim:
        idx = rng.choice(len(weights.sample), size=n, p=weights)
        # Create the distribution dataset
        ds_dist = ds.isel({"sample": idx}).chunk({"sample": -1})
    else:
        # 2D weights are not supported by np.random.choice, so we need to loop over the time dimension
        ds_tmp = [
            ds.sel({time_dim: time}).isel(
                {
                    "sample": rng.choice(
                        len(weights.sample), size=n, p=weights.sel({time_dim: time})
                    )
                }
            )
            for time in ds[time_dim].values
        ]
        # Remove the coordinates from the sample dimension to allow concatenation
        for i, ds_ in enumerate(ds_tmp):
            ds_tmp[i] = ds_.reset_index("sample", drop=True)
        ds_dist = xr.concat(ds_tmp, dim=time_dim).chunk({time_dim: -1, "sample": -1})

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
