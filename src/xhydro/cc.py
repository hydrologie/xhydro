"""Module to compute climate change statistics using xscen functions."""

from typing import Optional, Union

import numpy as np
import xarray as xr

# Special imports from xscen
from xscen import climatological_op, compute_deltas, ensemble_stats, produce_horizon

__all__ = [
    "climatological_op",
    "compute_deltas",
    "ensemble_stats",
    "produce_horizon",
    "sampled_indicators",
]


def sampled_indicators(
    ds: xr.Dataset,
    deltas: xr.Dataset,
    delta_type: str,
    *,
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
    delta_type : str
        Type of delta provided. Must be one of ['absolute', 'percentage'].
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
    if delta_weights is None:
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

    # Sample the distributions
    _, pdim, mult = _perc_or_quantile(ds)
    ds_dist = (
        _weighted_sampling(ds, percentile_weights, n=n, seed=seed)
        .drop_vars(["sample", *percentile_weights.dims])
        .assign_coords({"sample": np.arange(n)})
    )
    deltas_dist = (
        _weighted_sampling(deltas, delta_weights, n=n, seed=seed)
        .drop_vars(["sample", *delta_weights.dims])
        .assign_coords({"sample": np.arange(n)})
    )

    # Element-wise multiplication of the ref_dist and ens_dist
    if delta_type == "percentage":
        fut_dist = ds_dist * (1 + deltas_dist / 100)
    elif delta_type == "absolute":
        fut_dist = ds_dist + deltas_dist
    else:
        raise ValueError(
            f"Unknown operation '{delta_type}', expected one of ['absolute', 'percentage']."
        )

    # Compute future percentiles
    fut_pct = fut_dist.quantile(ds.percentile / mult, dim="sample")

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
    weights = weights / weights.sum()  # Must equal 1

    # For performance reasons, remove the chunking along impacted dimensions
    ds = ds.chunk({dim: -1 for dim in weights.dims})
    weights = weights.chunk({dim: -1 for dim in weights.dims})

    # Stack the dimensions containing weights
    ds = ds.stack({"sample": sorted(list(weights.dims))})
    weights = weights.stack({"sample": sorted(list(weights.dims))})
    weights = weights.reindex_like(ds)

    # Sample the dimensions with weights
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(weights.size, size=n, p=weights)

    # Create the distribution dataset
    ds_dist = ds.isel({"sample": idx}).chunk({"sample": -1})

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
