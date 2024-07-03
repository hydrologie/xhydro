"""Module to compute Probable Maximum Precipitation (PMP)"""

import numpy as np
import xarray as xr


def major_precipitation_events(ds, acc_day, quantil=0.9, path=None):
    """Extracts precipitation events that exceed a given quantile
       for a given days accumulation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the precipitation values.
    acc_day: list
        List of precipitation accumulation days.
    quantil: float
        Threshold that limits the events to those that exceed this quantile.
    path: str, optional
        Path where the results will be saved


    Returns
    -------
    da_acc : xr.DataArray
        DataArray containing the accumulated precipitation
    events : xr.DataArray
        Precipitation events that exceed the given quantile for
        the given days accumulation.

    """
    ds_exp = ds.expand_dims(dim={"acc_day": acc_day}).copy(deep=True)
    da_acc = xr.apply_ufunc(
        cumulate_precip,
        ds_exp,
        acc_day,
        input_core_dims=[["time", "y", "x"], []],
        output_core_dims=[["time", "y", "x"]],
        dask="parallelized",
        output_dtypes=[float],
        vectorize=True,
    )

    events = (
        da_acc.chunk(dict(time=-1))
        .groupby(da_acc.time.dt.year)
        .apply(keep_higest_values, quantil=quantil)
    )
    events = events.rename("rainfall_event")

    if path is not None:
        events.to_zarr(path)

    return events


def cumulate_precip(da, acc_day):
    """Accumulates precipitation for a given number of days.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the precipitation values.
    acc_day: Int
        Number of days to be accumulated.

    Returns
    -------
    da_acc : xr.DataArray
        DataArray containing the accumulated precipitation.

    """
    cumsum = np.cumsum(da, axis=0, dtype=float)

    da_acc = np.empty_like(da, dtype=float)
    da_acc[:acc_day, :, :] = cumsum[acc_day - 1, :, :]
    da_acc[acc_day:, :, :] = cumsum[acc_day:, :, :] - cumsum[:-acc_day, :, :]

    return da_acc


def keep_higest_values(da, quantil):
    """
    Mask values that are less than the given quantile.

    Parameters.
    ----------
    da : xr.DataArray
        DataArray containing the values.
    quantil: float.
        Quantile to compute, which must be between 0 and 1 inclusive.

    Returns.
    -------
    da_higest : xr.DataArray
        DataArray containing values greater than the given quantile.

    """
    threshold = da.quantile(quantil, dim="time")
    da_higest = da.where(da > threshold)

    return da_higest
