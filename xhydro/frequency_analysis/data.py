"""Data processing functions to support the frequency analysis."""
import xarray as xr
import xclim as xc
import xscen as xs  # FIXME: Replace with xhydro once xscen.compute_indicators is merged

import xhydro as xh

__all__ = ["get_yearly_op"]


def get_yearly_op(
    ds,
    op,
    *,
    freq: str = "YS",  # TODO: Test if this works with other frequencies
    window: int = 1,
    indexers: dict = None,
    missing: str = "skip",
    missing_options: dict = None,
    volume_method: str = "mean",
):
    """
    Compute a series of operations on a streamflow dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Streamflow dataset. Must contain a streamflow variable, or a volume variable if op is "volume".
    op: str
        Operation to compute. One of "max", "min", "mean", "volume".
        If op=="volume" and the dataset does not contain a volume variable, it will be computed on the fly using :py:func:`xhydro.compute_volume` with out_units="m3".
        If you want the volume to have a different unit, you must either call :py:func:`xhydro.compute_volume` before calling this function, or call :py:func:`xclim.units.convert_units` after calling this function.
    freq: str
        Resampling frequency. Defaults to "YS" (annual).
    window: int
        Size of the rolling window.
    indexers: dict
        Dictionary of indexers for the operation. Keys are the name of the indexers (e.g. "Winter", "Summer", "Annual") and the values are the indexers.
        See :py:func:`xclim.indices.generic.select_time` for more information.
        If op=="volume", the indexers must be either None, "doy_bounds" or "date_bounds".
    missing: str
        How to handle missing values. One of "skip", "any", "at_least_n", "pct", "wmo".
        See :py:func:`xclim.core.missing` for more information.
    missing_options: dict
        Dictionary of options for the missing values' method. See :py:func:`xclim.core.missing` for more information.
    volume_method: str
        Method to compute the volume. One of ["mean", "sum"]. Defaults to "mean", which multiplies the mean streamflow by the number of days in the indexer.

    Returns
    -------
    xr.Dataset
        Dataset containing the computed operations. The name of the variables is the name of the operation followed by the name of the indexers.

    Notes
    -----
    Volume operations are computed by multiplying the mean streamflow by the number of days in the indexer, instead of summing the streamflow.
    This is done to avoid the issue of missing values in the streamflow/volume variable, which would lead to an underestimation of the volume.

    """
    missing_options = missing_options or {}
    indexers = indexers or {"Annual": {}}

    # Check for variables
    if op != "volume" and "streamflow" not in ds:
        raise ValueError("The streamflow variable is missing.")
    if op == "volume" and "volume" not in ds:
        if "streamflow" not in ds:
            raise ValueError("Cannot compute a volume without a streamflow variable.")
        ds["volume"] = xh.compute_volume(ds, out_units="m3")

    # Add a volume variable to xclim if required
    if (op == "volume") and ("volume" not in xc.core.utils.VARIABLES):
        xc.core.utils.VARIABLES["volume"] = {
            "long_name": "Volume of water",
            "canonical_units": "m3",
            "cell_methods": "time: sum",
            "description": "Volume of water",
        }

    # rolling window
    if window > 1:
        if op != "volume":
            ds["streamflow"] = (
                ds["streamflow"]
                .rolling(dim={"time": window}, min_periods=window)
                .mean()
            )
        else:
            raise ValueError("Cannot use a rolling window with a volume operation.")

    indicators = []
    mult = []
    for i in indexers:
        if len(indexers[i]) > 1:
            raise ValueError("Only one indexer is supported per operation.")

        identifier = f"q{op}_{i.lower()}" if op != "volume" else f"volume_{i.lower()}"
        ind = xc.core.indicator.Indicator.from_dict(
            data={
                "base": "stats",
                "input": {"da": "streamflow" if op != "volume" else "volume"},
                "parameters": {
                    "op": op if op != "volume" else volume_method,
                    "freq": freq,
                    "indexer": indexers[i],
                },
                "missing": missing,
                "missing_options": missing_options,
            },
            identifier=identifier,
            module="fa",
        )
        indicators.append((identifier, ind))
        # Get the multiplier for the volume
        if op == "volume" and volume_method == "mean":
            if "doy_bounds" in indexers[i]:
                mult.append(
                    indexers[i]["doy_bounds"][1] - indexers[i]["doy_bounds"][0] + 1
                )
            elif "date_bounds" in indexers[i]:
                dt_start = ds.time.where(
                    (
                        ds.time.dt.month
                        == int(indexers[i]["date_bounds"][0].split("-")[0])
                    )
                    & (
                        ds.time.dt.day
                        == int(indexers[i]["date_bounds"][0].split("-")[1])
                    ),
                    drop=True,
                ).dt.dayofyear
                dt_end = ds.time.where(
                    (
                        ds.time.dt.month
                        == int(indexers[i]["date_bounds"][1].split("-")[0])
                    )
                    & (
                        ds.time.dt.day
                        == int(indexers[i]["date_bounds"][1].split("-")[1])
                    ),
                    drop=True,
                ).dt.dayofyear
                dt = xr.align(dt_start, dt_end, join="override")[1] - dt_start + 1
                mult.append(dt)
            elif indexers[i] == {}:
                # TODO: To be tested
                dt_start = ds.time.groupby("time.year").min().dt.dayofyear
                dt_end = ds.time.groupby("time.year").max().dt.dayofyear
                dt = xr.align(dt_start, dt_end, join="override")[1] - dt_start + 1
                mult.append(dt)
            else:
                raise NotImplementedError(
                    "Only doy_bounds and date_bounds are supported for volume operations."
                )

    out = xs.compute_indicators(ds, indicators=indicators).popitem()[1]
    if op == "volume" and volume_method == "mean":
        for i, m in zip(indexers, mult):
            identifier = f"volume_{i.lower()}"
            if isinstance(m, xr.DataArray):
                out[identifier] = out[identifier] * xr.align(out, m, join="override")[1]
            else:
                out[identifier] = out[identifier] * m

    return out
