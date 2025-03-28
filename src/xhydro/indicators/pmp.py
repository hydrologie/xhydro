"""Module to compute Probable Maximum Precipitation."""

import warnings
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
import xclim
from xclim.indices.stats import fit, parametric_quantile
from xscen.utils import unstack_dates


def major_precipitation_events(
    da: xr.DataArray, *, windows: list[int], quantile: float = 0.9
):
    """
    Get precipitation events that exceed a given quantile for a given time step accumulation. Based on Clavet-Gaumont et al. (2017).

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the precipitation values.
    windows : list of int
        List of the number of time steps to accumulate precipitation.
    quantile : float
        Threshold that limits the events to those that exceed this quantile. Defaults to 0.9.

    Returns
    -------
    xr.DataArray
        Masked DataArray containing the major precipitation events.

    Notes
    -----
    https://doi.org/10.1016/j.ejrh.2017.07.003
    """
    da_exp = xr.concat(
        [
            da.rolling({"time": window}, center=False)
            .sum(keep_attrs=True)
            .assign_coords({"window": window})
            for window in windows
        ],
        dim="window",
    )

    events = (
        da_exp.chunk(dict(time=-1))
        .groupby("time.year")
        .map(_keep_highest_values, quantile=quantile)
    )

    # Add attributes
    events.name = "rainfall_events"
    events.attrs["long_name"] = "Major precipitation events"
    events.attrs["description"] = (
        f"Major precipitation events defined as the {quantile * 100}% highest precipitation events for the given accumulation days."
    )

    return events


def _keep_highest_values(da: xr.DataArray, quantile: float) -> xr.DataArray:
    """
    Mask values that are less than the given quantile.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the values.
    quantile : float
        Quantile to compute, which must be between 0 and 1 inclusive.

    Returns
    -------
    xr.DataArray
        DataArray containing values greater than the given quantile.
    """
    threshold = da.quantile(quantile, dim="time")
    da_highest = da.where(da > threshold)

    return da_highest


def precipitable_water(
    *,
    hus: xr.DataArray,
    zg: xr.DataArray,
    orog: xr.DataArray,
    windows: list[int] | None = None,
    beta_func: bool = True,
    add_pre_lay: bool = False,
):
    """
    Compute precipitable water based on Clavet-Gaumont et al. (2017) and Rousseau et al (2014).

    Parameters
    ----------
    hus : xr.DataArray
        Specific humidity. Must have a pressure level (plev) dimension.
    zg : xr.DataArray
        Geopotential height. Must have a pressure level (plev) dimension.
    orog : xr.DataArray
        Surface altitude.
    windows : list of int, optional
        Duration of the event in time steps. Defaults to [1].
        Note that an additional time step will be added to the window size to account for antecedent conditions.
    beta_func : bool, optional
        If True, use the beta function proposed by Boer (1982) to get the pressure layers above the topography.
        If False, the surface altitude is used as the lower boundary of the atmosphere assuming that the surface altitude
        and the geopotential height are virtually identical at low altitudes.
    add_pre_lay : bool, optional
        If True, add the pressure layer between the surface and the lowest pressure level (e.g., at sea level).
        If False, only the pressure layers between the lowest and highest pressure level are considered.

    Returns
    -------
    xr.DataArray
        Precipitable water.

    Notes
    -----
    1) The precipitable water of an event is defined as the maximum precipitable water found during the entire duration of the event,
    extending up to one time step before the start of the event. Thus, the rolling operation made using `windows` is a maximum, not a sum.

    2) beta_func = True and add_pre_lay = False follow Clavet-Gaumont et al. (2017) and Rousseau et al (2014).

    3) https://doi.org/10.1016/j.ejrh.2017.07.003
       https://doi.org/10.1016/j.jhydrol.2014.10.053
       https://doi.org/10.1175/1520-0493(1982)110<1801:DEIIC>2.0.CO;2
    """
    windows = windows or [1]

    zg = xclim.core.units.convert_units_to(zg, "m")
    orog = xclim.core.units.convert_units_to(orog, "m")
    if hus.attrs.get("units") not in ["1", "", "kg kg-1", "kg/kg"]:
        warnings.warn(
            "Specific humidity units does not appear to be in kg/kg. Results may be incorrect."
        )

    # Correction of the first zone above the topography to consider the correct fraction.
    da = zg - orog
    da = da.where(da >= 0)

    # The surface altitude is used as lower boundary of the atmosphere assuming that the surface altitude and the geopotential height
    # are virtually identical at low altitudes. This layer will be removed if beta_func=True.
    zg_corr = (da + orog).fillna(orog)

    # Thickness of the pressure layers.
    zg1 = zg_corr.diff("plev", label="upper")
    zg2 = zg_corr.diff("plev", label="lower")
    zg_moy = xr.concat([zg1, zg2], dim="variable").sum("variable") / 2

    # Add the pressure layer below the lowest pressure level when the surface altitude is bewlow it (e.g., at  sea level).
    if add_pre_lay:
        plev_first = float(zg.plev.max())
        zg_moy = zg_moy.where(
            ~((zg_moy == zg_moy.sel(plev=plev_first)) & (~da.isnull())),
            other=(zg + zg_moy).sel(plev=plev_first),
        )

    # Betas computation for zones equal to zero.
    if beta_func:
        db = da.where(da >= 0, 0)
        beta = db.where(db <= 0, 1)
        zg_moy = zg_moy * beta

    # Precipitable water.
    pw = (zg_moy * hus).sum("plev")
    pw.name = "precipitable_water"
    pw.attrs = {
        "long_name": "Precipitable water",
        "description": "Precipitable water computed from the specific humidity and geopotential height.",
        "units": "m",
    }
    pw = xclim.core.units.convert_units_to(pw, "mm")

    # Compute the precipitable water for the given window.
    out = xr.concat(
        [
            pw.rolling({"time": window + 1}, center=False)
            .max(keep_attrs=True)
            .assign_coords({"window": window})
            for window in windows
        ],
        dim="window",
    )
    out["window"].attrs = {
        "description": "Duration of the event, including antecedent conditions (window + 1)."
    }

    return out


def precipitable_water_100y(
    pw: xr.DataArray,
    *,
    dist: str,
    method: str,
    mf: float = 0.2,
    rebuild_time: bool = True,
):
    """Compute the 100-year return period of precipitable water for each month. Based on Clavet-Gaumont et al. (2017).

    Parameters
    ----------
    pw : xr.DataArray
        DataArray containing the precipitable water.
    dist : str
        Probability distributions.
    method : {"ML" or "MLE", "MM", "PWM", "APP"}
        Fitting method, either maximum likelihood (ML or MLE), method of moments (MM) or approximate method (APP).
        Can also be the probability weighted moments (PWM), also called L-Moments, if a compatible `dist` object is passed.
    mf : float
        Maximum majoration factor of the 100-year event compared to the maximum of the timeseries.
        Used as an upper limit for the frequency analysis.
    rebuild_time : bool
        Whether or not to reconstruct a timeseries with the same time dimensions as `pw`.

    Returns
    -------
    xr.DataArray
        Precipitable water for a 100-year return period.

    Notes
    -----
    https://doi.org/10.1016/j.ejrh.2017.07.003
    """
    # Compute max monthly and add a «year» dimension.
    pw_m = pw.resample({"time": "MS"}).max()
    pw_m = unstack_dates(pw_m, seasons={m: m for m in range(1, 13)}, new_dim="month")

    # Fits distribution
    # FIXME: Use xhydro.frequency_analysis.local instead
    params = fit(pw_m, dist=dist, method=method)
    pw100_m = parametric_quantile(params, q=1 - 1 / 100).squeeze()

    # Add a limit to PW100 to limit maximization factors.
    pw100_m = pw100_m.clip(max=(pw_m.max(dim="time") * (1.0 + mf)))

    if rebuild_time:
        hour = np.unique(pw.time.dt.hour)
        if len(hour) > 1:
            raise ValueError("The time dimension must be homogeneous.")
        hour = hour[0]
        pw100_m = pw100_m.expand_dims(dim={"year": np.unique(pw.time.dt.year)})
        pw100_m = pw100_m.stack(stacked_coords=("month", "year"))
        if isinstance(pw.indexes["time"], pd.core.indexes.datetimes.DatetimeIndex):
            time_coord = pd.DatetimeIndex(
                pd.to_datetime(
                    {
                        "year": pw100_m.year,
                        "month": pw100_m.month,
                        "day": 1,
                        "hour": hour,
                    }
                )
            )
        elif isinstance(pw.indexes["time"], xr.coding.cftimeindex.CFTimeIndex):
            time_coord = [
                xclim.core.calendar.datetime_classes[pw.time.dt.calendar](y, m, 1, hour)
                for y, m in zip(
                    pw100_m.year.values,
                    pw100_m.month.values,
                )
            ]
        else:
            raise ValueError("The type of 'time' was not understood.")

        pw100_m = pw100_m.assign_coords(time=("stacked_coords", time_coord))
        pw100_m = pw100_m.swap_dims({"stacked_coords": "time"}).sortby("time")
        pw100_m = (
            pw100_m.reindex_like(pw)
            .ffill(dim="time")
            .drop_vars(["month", "year", "stacked_coords"])
        )

    pw100_m.name = "precipitable_water_monthly_100y"

    return pw100_m


def compute_spring_and_summer_mask(
    snw: xr.DataArray,
    *,
    thresh: str = "1 cm",
    window_wint_start: int = 14,
    window_wint_end: int = 45,
    spr_start: int = 60,
    spr_end: int = 30,
    freq: str = "YS-JUL",
):
    """Create a mask that defines the spring and summer seasons based on the snow water equivalent.

    Parameters
    ----------
    snw : xarray.DataArray
        Snow water equivalent. Must be a length (e.g. "mm") or a mass (e.g. "kg m-2").
    thresh : Quantified
        Threshold snow thickness to define the start and end of winter.
    window_wint_start : int
        Minimum number of days with snow depth above or equal to threshold to define the start of winter.
    window_wint_end : int
        Maximum number of days with snow depth below or equal to threshold to define the end of winter.
    spr_start : int
        Number of days before the end of winter to define the start of spring.
    spr_end : int
        Number of days after the end of winter to define the end of spring.
    freq : str
        Frequency of the time axis (annual frequency). Defaults to "YS-JUL".

    Returns
    -------
    xr.Dataset
        Dataset with two DataArrays (mask_spring and mask_summer), with values of 1 where the
        spring and summer criteria are met and 0 where they are not.
    """
    attrs = deepcopy(snw.attrs)
    snw = xclim.core.units.convert_units_to(snw, "mm", context="hydro")
    # xclim expects precipitation and thus writes wrong attributes.
    snw.attrs.update(attrs)
    snw.attrs["units"] = "mm"

    winter_start = xclim.indices.snd_season_start(
        snd=snw, thresh=thresh, window=window_wint_start, freq=freq
    )
    first_day = int(winter_start.time.dt.dayofyear[0])
    if first_day < 182:
        raise NotImplementedError(
            "Frequencies starting before July 1st are not yet supported."
        )

    # Summer ends when winter starts, or at the end of the year
    summer_end = (
        xr.where(winter_start > first_day, winter_start - 1, 366)
        .assign_coords({"year": winter_start.time.dt.year})
        .swap_dims({"time": "year"})
        .drop_vars("time")
        .sel(year=slice(min(snw.time.dt.year), max(snw.time.dt.year)))
    )

    # YS-JUL and similar freqs shifts the start by a year. Since we are looking for dates in spring and summer, we need to add a year.
    winter_start = winter_start.assign_coords({"year": winter_start.time.dt.year + 1})
    winter_start = winter_start.swap_dims({"time": "year"}).drop_vars("time")
    winter_start = winter_start.sel(
        year=slice(min(snw.time.dt.year), max(snw.time.dt.year))
    )  # The last year is not complete
    mask = xr.where(winter_start.isnull(), np.nan, 1)
    winter_start = (
        winter_start.where(winter_start <= first_day, 1) * mask
    )  # Set to 1 where winter started before January 1st

    winter_end = (
        xclim.indices.snd_season_end(
            snd=snw, thresh=thresh, window=window_wint_end, freq=freq
        )
        - 1
    )  # xclim gives the first day without snow, we want the last day with snow
    winter_end = winter_end.where(
        winter_end < first_day - 2
    )  # If winter never ends, xclim gives the last day of the year

    winter_end = winter_end.assign_coords({"year": winter_end.time.dt.year + 1})
    winter_end = winter_end.swap_dims({"time": "year"}).drop_vars("time")
    winter_end = winter_end.sel(
        year=slice(min(snw.time.dt.year), max(snw.time.dt.year))
    )  # The last year is not complete
    winter_end = xr.where(
        winter_end >= first_day, 1, winter_end
    )  # If winter ends the autumn before, set to 1

    # Sanity check
    if (winter_end.isnull() & winter_start.notnull()).any():
        raise ValueError(
            "Winter starts but never ends. Check your `freq` or `window` parameters."
        )

    winter_end = winter_end.where(
        winter_end < summer_end, summer_end - 1
    )  # Winter can't end after summer
    mask = xr.where(winter_start.isnull(), np.nan, 1)
    winter_end = winter_end * mask

    # Summer starts when winter ends, or at the beginning of the year
    summer_start = xr.where(winter_end.isnull(), 1, winter_end + 1)

    spring_start = winter_end - spr_start
    spring_start = spring_start.where(spring_start >= 1, 1) * mask
    spring_end = winter_end + spr_end

    # Create the masks
    def _create_mask(group, start, end):
        year = group.time.dt.year[0]
        s = start.sel(year=year)
        e = end.sel(year=year)
        year_mask = (group.time.dt.dayofyear >= s) & (group.time.dt.dayofyear <= e)
        return year_mask

    summer_mask = (
        xr.zeros_like(snw)
        .groupby("time.year")
        .map(_create_mask, start=summer_start, end=summer_end)
        .drop_vars("year")
        .astype(int)
    )
    spring_mask = (
        xr.zeros_like(snw)
        .groupby("time.year")
        .map(_create_mask, start=spring_start, end=spring_end)
        .drop_vars("year")
        .astype(int)
    )

    summer_mask.name = "mask_summer"
    summer_mask.attrs = {
        "long_name": "Summer mask",
        "description": f"Mask defining the summer season based on a SWE threshold of {thresh}. "
        f"Winter starts when the SWE is above the threshold for at least "
        f"{window_wint_start} days and ends when it is below the threshold "
        f"for at least {window_wint_end} days.",
    }
    summer_mask.attrs["units"] = ""
    spring_mask.name = "mask_spring"
    spring_mask.attrs = {
        "long_name": "Spring mask",
        "description": f"Mask defining the spring season based on a SWE threshold of {thresh}. "
        f"Winter starts when the SWE is above the threshold for at least "
        f"{window_wint_start} days and ends when it is below the threshold "
        f"for at least {window_wint_end} days. Spring starts {spr_start} days "
        f"before the end of winter and ends {spr_end} days after the end of winter.",
    }
    spring_mask.attrs["units"] = ""

    return xr.Dataset(
        {
            "mask_spring": spring_mask.where(spring_mask == 1),
            "mask_summer": summer_mask.where(summer_mask == 1),
        }
    )


def spatial_average_storm_configurations(da, *, radius):
    """Compute the spatial average for different storm configurations proposed by Clavet-Gaumont et al. (2017).

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the precipitation values.
    radius : float
        Maximum radius of the storm.

    Returns
    -------
    xr.DataSet
        DataSet containing the spatial averages for all the storm configurations. The y and x coordinates indicate
        the location of the storm. This location is determined by n//2, where n is the total number of cells for
        both the rows and columns in the configuration, and // represents floor division.

    Notes
    -----
    https://doi.org/10.1016/j.ejrh.2017.07.003.
    """
    dict_config = {
        "2.1": [[0, 0], [0, 1]],
        "2.2": [[0, 1], [0, 0]],
        "3.1": [[0, 1, 1], [0, 0, 1]],
        "3.2": [[0, 1, 1], [1, 0, 1]],
        "3.3": [[0, 0, 1], [0, 1, 1]],
        "3.4": [[0, 0, 1], [0, 1, 0]],
        "4.1": [[0, 0, 1, 1], [0, 1, 0, 1]],
        "5.1": [[0, 0, 1, 1, 1], [1, 2, 0, 1, 2]],
        "5.2": [[0, 0, 0, 1, 1], [0, 1, 2, 1, 2]],
        "5.3": [[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]],
        "5.4": [[0, 0, 1, 1, 1], [0, 1, 0, 1, 2]],
        "5.5": [[0, 0, 1, 1, 2], [0, 1, 0, 1, 0]],
        "5.6": [[0, 0, 1, 1, 2], [0, 1, 0, 1, 1]],
        "5.7": [[0, 1, 1, 2, 2], [0, 0, 1, 0, 1]],
        "5.8": [[0, 1, 1, 2, 2], [1, 0, 1, 0, 1]],
        "6.1": [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
        "6.2": [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        "7.1": [[0, 0, 0, 1, 1, 1, 2], [0, 1, 2, 0, 1, 2, 1]],
        "7.2": [[0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 0, 1]],
        "7.3": [[0, 1, 1, 1, 2, 2, 2], [1, 0, 1, 2, 0, 1, 2]],
        "7.4": [[0, 0, 1, 1, 1, 2, 2], [1, 2, 0, 1, 2, 1, 2]],
        "8.1": [[0, 0, 0, 1, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2, 1, 2]],
        "8.2": [[0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 0, 1, 2, 0, 1, 2]],
        "8.3": [[0, 0, 1, 1, 1, 2, 2, 2], [1, 2, 0, 1, 2, 0, 1, 2]],
        "8.4": [[0, 0, 0, 1, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1]],
        "9.1": [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        "10.1": [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2], [0, 1, 2, 3, 0, 1, 2, 3, 1, 2]],
        "10.2": [[0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [1, 2, 0, 1, 2, 3, 0, 1, 2, 3]],
        "10.3": [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 1, 0, 1, 2, 0, 1, 2, 0, 1]],
        "10.4": [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [1, 2, 0, 1, 2, 0, 1, 2, 1, 2]],
        "12.1": [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ],
        "12.2": [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        ],
        "14.1": [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2],
        ],
        "14.2": [
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
        ],
        "14.3": [
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ],
        "14.4": [
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
        ],
        "16.1": [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ],
        "18.1": [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3],
        ],
        "18.2": [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3],
            [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3, 4],
        ],
        "18.3": [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2],
        ],
        "18.4": [
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            [1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ],
        "20.1": [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        ],
        "20.2": [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ],
        "23.1": [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
            [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3],
        ],
        "23.2": [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
            [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3, 4],
        ],
        "23.3": [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3],
        ],
        "23.4": [
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            [1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        ],
        "24.1": [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        ],
        "24.2": [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        ],
        "25.1": [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        ],
    }

    x = da.cf.axes["X"][0]
    y = da.cf.axes["Y"][0]

    # Pixel size
    dy = (da[y][1] - da[y][0]).values
    dx = (da[x][1] - da[x][0]).values

    r_max = dy * max(dict_config["24.2"][0]) / 2
    if r_max < radius:
        warnings.warn(
            f"The chosen `radius` exceeds the maximum storm radius that can be calculated. As a result, the maximum storm radius will be {r_max}."
        )

    if (dy > radius) or (dx > radius):
        raise ValueError(
            "The chosen `radius` is smaller than the grid size. Please choose a larger `radius`."
        )

    # Number of pixels in da
    npy_da = len(da[y])
    npx_da = len(da[x])

    da_stacked = da.stack(stacked_coords=("y", "x"))
    confi_lst = []
    for name, confi in dict_config.items():

        conf_y = confi[0]
        conf_x = confi[1]

        # Number of pixels in the configuration
        npy_confi = np.abs(conf_y).max() + 1
        npx_confi = np.abs(conf_x).max() + 1

        # Checks that the configuration size is within the desired storn size.
        if (dy * npy_confi / 2 > radius) or (dx * npx_confi / 2 > radius):
            break

        # Checks that the configuration fits in the domain.
        if (npy_confi > npy_da) or (npx_confi > npx_da):
            break

        # Number of times a configuration can be shifted in each axis
        ny = len(da[y]) - npy_confi + 1
        nx = len(da[x]) - npx_confi + 1

        # List with the configuration duplicated as many times as indicated by nx and nx.
        conf_y_ex = np.reshape(np.array(conf_y * ny), (ny, len(conf_y)))
        conf_x_ex = np.reshape(np.array(conf_x * nx), (nx, len(conf_x)))

        # List with the incrementes from 0 to nx and ny
        inc_y = np.ones(conf_y_ex.shape) * [[i] for i in range(ny)]
        inc_x = np.ones(conf_x_ex.shape) * [[i] for i in range(nx)]

        # Shifted configurations
        pos_y = (conf_y_ex + inc_y).astype(int)
        pos_x = (conf_x_ex + inc_x).astype(int)

        # List of all the combinations of the shifted configurations
        shifted_confi = list(product(pos_y, pos_x))

        list_mean = []
        for confi_shifted in shifted_confi:

            matrix_mask = np.full((len(da[y]), len(da[x])), np.nan)

            cen_shift_y = (min(confi_shifted[0]) + max(confi_shifted[0])) // 2
            cen_shift_x = (min(confi_shifted[1]) + max(confi_shifted[1])) // 2

            matrix_mask[(confi_shifted[0], confi_shifted[1])] = 1
            da_mask = da * matrix_mask
            da_mean = (
                da_mask.mean(dim=[x, y])
                .expand_dims(
                    dim={
                        "y": [da[y][cen_shift_y].values],
                        "x": [da[x][cen_shift_x].values],
                    }
                )
                .stack(stacked_coords=("y", "x"))
            )

            list_mean.append(da_mean)

        confi_lst.append(
            xr.concat(list_mean, dim="stacked_coords")
            .reindex_like(da_stacked, fill_value=np.nan)
            .unstack("stacked_coords")
            .expand_dims(dim={"conf": [name]})
        )

    confi_ds = xr.concat(confi_lst, dim="conf")

    if "units" in da.attrs:
        confi_ds.attrs["units"] = da.attrs["units"]

    return confi_ds
