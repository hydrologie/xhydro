"""
Provides functions for hydrological modeling and data manipulation, specifically focusing on Probable Maximum Flood (PMF) analysis.
The functions include:
- `remove_precip`: Removes precipitation data for specified dates.
- `swap_meteo`: Swaps meteorological data for a given event.
- `separate_pr`: Separates precipitation into rain and snow based on temperature thresholds.
- `two_year_pmp`: Retrieves two years of meteorological data and integrates a PMP event.
- `fix_pmp_year`: Adjusts the year of a PMP event in the data array.
- `place_pmp`: Integrates PMP data into the meteorological dataset.
This module is intended for use in hydrological studies and flood risk assessments.
"""

import numpy as np
import pandas as pd
import xarray as xr

from xhydro.utils import update_history


def remove_precip(da_full, dates, each_year=False):
    """
    Remove precipitation for given dates.

    Parameters
    ----------
    da_full : xarray.DataArray
        Full precipitation data array.
    dates : list
        List of dates to remove precipitation.

    Returns
    -------
    xarray.DataArray
        Data array with precipitation removed for given dates.
    """
    if each_year:
        da = xr.DataArray(data=np.zeros(len(dates)), coords={"time": dates})
        return _swap_meteo(da_full, da)
    else:
        mask = da_full["time"].isin(dates)
        return da_full.where(~mask, other=0)


def _swap_meteo(da_full, da_event):
    """
    Swap meteorological data for given event.

    Parameters
    ----------
    da_full : xarray.DataArray
        Full meteorological data array.
    da_event : xarray.DataArray
        Event data array.

    Returns
    -------
    xarray.DataArray
        Data array with swapped meteorological data.
    """
    return da_full.groupby("time.year").apply(_replace_one_year, args=(da_event,))


def _replace_one_year(full_da, sample_da):
    """
    Replace one year of data with sample data.

    Parameters
    ----------
    full_da : xarray.DataArray
        Full data array.
    sample_da : xarray.DataArray
        Sample data array.

    Returns
    -------
    xarray.DataArray
        Data array with one year replaced by sample data.
    """
    full_doy = full_da.time.dt.dayofyear
    sample_doy = sample_da.time.dt.dayofyear
    mask = xr.where(full_doy.isin(sample_doy), True, False)
    if 1 and 365 in sample_da.time.dt.dayofyear:
        if 366 in full_doy:
            full_leap = True
        else:
            full_leap = False
        if 366 in sample_doy:
            sample_leap = True
        else:
            sample_leap = False
        if len(sample_doy) - sum(mask.values) > 1:
            return full_da
        elif full_leap == sample_leap:
            fill_values = np.empty(mask.shape)
            fill_values[mask.values] = sample_da.values
            return full_da.where(~mask, other=fill_values)
        elif full_leap and not sample_leap:
            sample_doy = sample_doy.to_numpy()
            sample_doy = np.append(sample_doy, 366)
            mask = xr.where(full_doy.isin(sample_doy), True, False)
            fill_values = np.empty(mask.shape)
            fill_values[mask.values] = np.append(sample_da.values, 0)
            return full_da.where(~mask, other=fill_values)
        elif not full_leap and sample_leap:
            sample_doy = sample_doy.to_numpy()
            sample_doy = sample_doy[sample_doy != 366]
            mask = xr.where(full_doy.isin(sample_doy), True, False)
            fill_values = np.empty(mask.shape)
            fill_values[mask.values] = sample_da.where(
                sample_da.time.dt.dayofyear != 366, drop=True
            ).values
            return full_da.where(~mask, other=fill_values)
    else:
        if sum(mask.values) < len(sample_doy):
            return full_da
        else:
            fill_values = np.empty(mask.shape)
            fill_values[mask.values] = sample_da.values
            return full_da.where(~mask, other=fill_values)


def separate_pr(ds, pr, tmin, tmax, t_trans=0, delta_t=4, algo="DINGMAN"):
    """
    Separate precipitation into rain and snow.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing precipitation data.
    pr : str
        Precipitation variable name.
    tmin : str
        Minimum temperature variable name.
    tmax : str
        Maximum temperature variable name.
    t_trans : int, optional
        Transition temperature (default is 0).
    delta_t : int, optional
        Temperature range for transition (default is 4).
    algo : str, optional
        Algorithm to use for separation (default is 'DINGMAN').

    Returns
    -------
    xarray.Dataset
        Dataset with separated rain and snow.
    """
    attrs = ds[pr].attrs | {
        "t_trans": t_trans,
        "history": update_history(f"Separated with algo {algo}"),
    }
    if algo == "DINGMAN":
        snow_fraction = (t_trans - ds[tmin]) / (ds[tmax] - ds[tmin])
    elif algo == "UBC" or algo == "HBV":
        t_ave = (ds[tmax] + ds[tmin]) / 2
        snow_fraction = 0.5 + ((t_trans - t_ave) / delta_t)
        attrs = attrs | {"delta_t": delta_t}
    else:
        raise NotImplementedError

    snow_fraction = snow_fraction.where(snow_fraction <= 1, other=1)
    snow_fraction = snow_fraction.where(snow_fraction >= 0, other=0)
    snow = pr + "_snow"
    rain = pr + "_rain"
    ds[snow] = ds[pr] * snow_fraction
    ds[rain] = ds[pr] * (1 - snow_fraction)

    ds[rain].attrs = attrs
    ds[snow].attrs = attrs
    ds[rain].attrs["long_name"] = "Rainfall"
    ds[snow].attrs["long_name"] = "Snowfall"
    return ds


def fix_pmp_year(da_rain_pmp, year):
    """
    Fix PMP event year.

    Parameters
    ----------
    da_rain_pmp : xarray.DataArray
        PMP event data array.
    year : int
        Year to fix the PMP event.

    Returns
    -------
    xarray.DataArray
        Data array with fixed PMP event year.
    """
    milieu = str(year) + da_rain_pmp.time.min().dt.strftime("-%m-%d").values

    delta = pd.Timestamp(milieu) - da_rain_pmp.time.min().values
    da_rain_pmp["time"] = da_rain_pmp.time + delta
    return da_rain_pmp


def place_pmf_inputs(ds_pmp: xr.Dataset, ds_meteo: xr.Dataset) -> xr.Dataset:
    """
    Place PMP data into the meteorological dataset.

    Parameters
    ----------
    ds_pmp : xr.Dataset
        The PMP dataset.
    ds_meteo : xr.Dataset
        The meteorological dataset.

    Returns
    -------
    xr.Dataset
        The updated meteorological dataset.
    """
    for v in ds_meteo.data_vars:
        mask = xr.where(ds_meteo[v].time.isin(ds_pmp.time), True, False)
        fill_values = np.empty(mask.shape)
        fill_values[mask.values] = ds_pmp[v].values.squeeze()
        ds_meteo[v] = ds_meteo[v].where(~mask, other=fill_values)
    return ds_meteo
