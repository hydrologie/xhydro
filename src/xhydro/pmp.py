"""Module to compute Probable Maximum Precipitation."""

import sys
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
import xclim
from xclim.indices.stats import fit, parametric_quantile


def major_precipitation_events(da, acc_day, quantil=0.9, path=None):
    """Extract precipitation events that exceed a given quantile for a given days accumulation.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the precipitation values.
    acc_day : list
        List of precipitation accumulation days.
    quantil : float
        Threshold that limits the events to those that exceed this quantile.
    path : str, optional
        Path where the results will be saved.

    Returns
    -------
    xr.DataArray
        Precipitation events that exceed the given quantile for
        the given days accumulation.
    """
    if "y" and "x" in da.coords:
        core_dims = ["time", "y", "x"]

    if "conf" in da.coords:
        core_dims = ["time", "conf"]

    ds_exp = da.expand_dims(dim={"acc_day": acc_day}).copy(deep=True)
    da_acc = xr.apply_ufunc(
        cumulate_precip,
        ds_exp,
        acc_day,
        input_core_dims=[core_dims, []],
        output_core_dims=[core_dims],
        dask="parallelized",
        output_dtypes=[float],
        vectorize=True,
    )

    events = (
        da_acc.chunk(dict(time=-1))
        .groupby(da_acc.time.dt.year)
        .map(keep_higest_values, quantil=quantil)
    )

    if path is not None:
        events.to_zarr(path)

    return events.rename("rainfall_event")


def cumulate_precip(a, acc_day):
    """Accumulate precipitation for a given number of days.

    Parameters
    ----------
    a : array_like
       Precipitation values.
    acc_day : Int
        Number of days to be accumulated.

    Returns
    -------
    array_like
        Array containing the accumulated precipitation.
    """
    cumsum = np.cumsum(a, axis=0, dtype=float)
    a_acc = np.empty_like(a, dtype=float)

    if a.ndim == 3:
        a_acc[:acc_day, :, :] = cumsum[acc_day - 1, :, :]
        a_acc[acc_day:, :, :] = cumsum[acc_day:, :, :] - cumsum[:-acc_day, :, :]
    else:
        a_acc[:acc_day] = cumsum[acc_day - 1]
        a_acc[acc_day:] = cumsum[acc_day:] - cumsum[:-acc_day]

    return a_acc


def keep_higest_values(da, quantil):
    """
    Mask values that are less than the given quantile.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the values.
    quantil : float
        Quantile to compute, which must be between 0 and 1 inclusive.

    Returns
    -------
    xr.DataArray
        DataArray containing values greater than the given quantile.
    """
    threshold = da.quantile(quantil, dim="time")
    da_higest = da.where(da > threshold)

    return da_higest


def precipitable_water(ds, ds_fx, acc_day=[1], path=None):
    """Compute the precipitable water for the antecedent conditions given the accumulations days.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the Specific humidity (hus) and the Geopotential Height (zg).
    ds_fx : xr.Dataset
        Dataset containing the Surface altitude (orog).
    acc_day : list
        List of precipitation accumulation days.
    path : str, optional
        Path where the results will be saved.

    Returns
    -------
    xr.DataArray
        Precipitable water.
    """
    # Correction of the first zone above the topography to consider the correct fraction.
    da = ds["zg"] - ds_fx["orog"]
    da = da.where(da >= 0)
    ds["zg_corr"] = (da + ds_fx["orog"]).fillna(ds_fx["orog"])

    # Betas computation for zones equal to zero.
    db = da.where(da >= 0, 0)
    ds["beta"] = db.where(db <= 0, 1)

    # Evaluation of average elevation for each pressure level, including beta coefficient
    ds["zg1"] = ds.zg_corr.diff("plev", label="upper")
    ds["zg2"] = ds.zg_corr.diff("plev", label="lower")
    ds["zg_moy"] = ds[["zg1", "zg2"]].to_array().sum("variable") / 2
    ds["zg_moy"] = ds["zg_moy"] * ds["beta"]

    # Precipitable water. You can then juxtapose with Pevent to return only PWevent (max).
    ds["pt"] = (ds["zg_moy"] * ds["hus"]).sum("plev")
    pw = ds["pt"].rename("precipitable_water")

    pw_exp = (
        pw.expand_dims(dim={"acc_day": acc_day}).chunk({"time": -1}).copy(deep=True)
    )

    da_pw = xr.apply_ufunc(
        rolling_max,
        pw_exp,
        (np.array(acc_day) + 1),
        input_core_dims=[["time", "y", "x"], []],
        output_core_dims=[["time", "y", "x"]],
        dask="parallelized",
        output_dtypes=[float],
        vectorize=True,
    )

    if path is not None:
        da_pw.to_zarr(path)

    return da_pw


def rolling_max(arr, window_size):
    """Compute the maximum rolling value.

    Parameters
    ----------
    arr : array_like
        Array to create the sliding window.
    window_size : xr.Dataset
        Size of window over each axis that takes part in the sliding window.

    Returns
    -------
    ndarray
        Rolling maximum of the array. The function includes NaN values at the beginning of the array to keep the same length of the input array.
    """
    windowed = np.lib.stride_tricks.sliding_window_view(arr, window_size, axis=0)

    roll_max = np.maximum.reduce(windowed, axis=-1)

    roll_max_out = np.full(arr.shape, np.nan)
    roll_max_out[window_size - 1 :, :, :] = roll_max

    return roll_max_out


def precipitable_water_100y(da_pw, dist, mf=0.2, path=None):
    """Compute the 100-year return period of precipitable water for each month of the year.

    Parameters
    ----------
    da_pw : xr.DataArray
        Dataset containing the precipitable water.
    dist : lmoments3 distribution object
        Probability distributions.
    mf : float
        The annual maximums of the precipitable water plus a porcentage (mf) are used as a upper limit.
    path : str, optional
        Path where the results will be saved.

    Returns
    -------
    xr.DataArray
        Precipitable water for a 100-year return period.
        It has the same dimensions as da_pw.
    """
    # Compute max monthly and add a «year» dimension.
    da_pw_m = da_pw.resample({"time": "ME"}).max()
    year = da_pw_m.time.dt.year
    month = da_pw_m.time.dt.month
    da_pw_m = da_pw_m.assign_coords(
        year=("time", year.data), month=("time", month.data)
    )
    da_pw_m = da_pw_m.set_index(time=("year", "month")).unstack("time")
    da_pw_m = da_pw_m.rename({"year": "time"}).squeeze()

    # Fits distribution
    params = fit(da_pw_m, dist=dist, method="pwm")
    pw100_m = parametric_quantile(params, q=1 - 1 / 100).squeeze().rename("pw100")

    pw_mm = da_pw_m.rename("precipitable_water_monthly")
    pw_mm = pw_mm.rename({"time": "year"})

    # Add a limit to PW100 to limit maximization factors.
    pw_mm_mf = (pw_mm.max(dim="year") * (1.0 + mf)).squeeze()

    pw100_m = pw100_m.where(pw100_m < pw_mm_mf, other=pw_mm_mf)
    pw100_m = pw100_m.expand_dims(dim={"year": np.unique(year.values)})
    pw100_m = pw100_m.stack(stacked_coords=("month", "year"))
    date_index = pd.DatetimeIndex(
        pd.to_datetime(
            pd.DataFrame(
                {
                    "year": pw100_m.year.data,
                    "month": pw100_m.month.data,
                    "day": (xr.ones_like(pw100_m.month).data),
                    "hour": (xr.ones_like(pw100_m.month).data) * 12,
                }
            )
        )
    )
    pw100_m = pw100_m.assign_coords(time=("stacked_coords", date_index))

    pw100_m = pw100_m.swap_dims({"stacked_coords": "time"}).sortby("time")
    pw100_m = pw100_m.convert_calendar("noleap")
    pw100_m = pw100_m.rename("pw100").to_dataset()

    if "x" and "y" in pw100_m.coords:
        pw100_m = pw100_m.transpose("time", "y", "x")
        da_pw = da_pw.drop_vars(["x", "y"])

    da_pw100 = xr.merge([pw100_m, da_pw]).ffill(dim="time")

    da_pw100 = da_pw100.pw100.squeeze().drop_vars(["month", "year", "stacked_coords"])

    if path is not None:
        da_pw100.to_zarr(path)

    return da_pw100


def compute_spring_and_summer_mask(
    snt,
    thresh="1 cm",
    window_wint_start=14,
    window_wint_end=90,
    spr_start=60,
    spr_end=30,
):
    """Create a mask that defines the spring and summer seasons based on the snow thickness.

    Parameters
    ----------
    snt : xarray.DataArray
        Surface snow thickness.

    thresh : Quantified
        Threshold snow thickness.

    window_wint_start : int
        Minimum number of days with snow depth above or equal to threshold to define the start of winter.

    window_wint_end : int
        Maximum number of days with snow depth above or equal to threshold to define the end of winter.

    spr_start : int
        Number of days before the end of winter to define the start of spring.

    spr_end : int
        Number of days after the end of winter to define the end of spring.

    Returns
    -------
    xr.Dataset
        Dataset with two DataArrays (mask_spring and mask_summer), with values of 1 where the
        spring and summer criteria are met and 0 where they are not.
    """
    if snt.attrs["units"] == "kg m-2":
        snt = snt / 10  # kg/m² == 1mm --> Transformation 1mm = 1cm/10
        snt.attrs["units"] = "cm"
    else:
        sys.exit("snow units are not in kg m-2")

    winter_start = xclim.indices.snd_season_start(
        snd=snt, thresh=thresh, window=window_wint_start, freq="YS-JUL"
    )
    winter_start = winter_start.assign_coords({"year": winter_start.time.dt.year})
    winter_start = winter_start.swap_dims({"time": "year"})

    winter_end = xclim.indices.snd_season_end(
        snd=snt, thresh=thresh, window=window_wint_end, freq="YS"
    )
    winter_end = winter_end.assign_coords({"year": winter_end.time.dt.year})
    winter_end = winter_end.swap_dims({"time": "year"})

    spring_start = winter_end - spr_start
    spring_start = spring_start.where(spring_start >= 1, 1)
    spring_end = winter_end + spr_end

    mask = xr.where(winter_start.isnull(), np.nan, 1)
    mask = mask.drop_sel(year=max(winter_start.time.dt.year))
    mask["year"] = mask.year + 1
    winter_end = winter_end * mask
    array_winter_end = xr.ones_like(snt).groupby(snt.time.dt.year) * (winter_end)
    condition1 = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        winter_start.drop_vars("time")
    )

    # Set NaN when the winter start before july 1rst
    mask = xr.where(condition1 >= 182, 1, np.nan)
    condition1 = condition1 * mask

    winter_start["year"] = winter_start.year + 1
    condition2 = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        winter_start.drop_vars("time")
    )

    # Set NaN when the winter start after july 1rst
    mask = xr.where(condition2 < 182, 1, np.nan)
    condition2 = condition2 * mask

    # Make an array of julian days the same size as «snt»
    array_days = xr.ones_like(snt) * snt.time.dt.dayofyear

    a1 = array_days >= condition1
    b1 = array_days < array_winter_end
    c1 = a1 + b1
    winter_mask_half1 = xr.where(c1 == 1, 1, 0)

    a2 = array_days >= condition2
    a2 = xr.where(a2, 1, 0)
    b2 = array_days < array_winter_end
    b2 = xr.where(b2, 1, 0)
    c2 = a2 + b2
    winter_mask_half2 = xr.where(c2 == 2, 1, 0)

    winter_mask = (winter_mask_half1 + winter_mask_half2).drop_vars("year")
    winter_mask = xr.where(winter_mask >= 1, 1, np.nan)

    summer_mask = xr.where(winter_mask.isnull(), 1, np.nan)

    # Spring mask
    array_days = xr.ones_like(snt) * snt.time.dt.dayofyear
    array_days_end_of_spring = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        spring_end.drop_vars(["time"])
    )
    array_days_start_of_spring = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        spring_start.drop_vars(["time"])
    )

    condition3 = array_days >= array_days_start_of_spring
    condition4 = array_days <= array_days_end_of_spring

    c3 = xr.where(condition3, 1, 0)
    c4 = xr.where(condition4, 1, 0)
    c = c3 + c4
    spring_mask = xr.where(c == 2, 1, np.nan)

    return xr.Dataset({"mask_spring": spring_mask, "mask_summer": summer_mask})


def spatial_average_storm_configurations(da, radius, path=None):
    """Compute the spatial average for different storm configurations (Clavet-Gaumont et al., 2017).

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the precipitation values.
    radius : float
        Maximum radius of the storm.
    path : str, optional
        Path where the results will be saved.

    Returns
    -------
    xr.DataSet
        DataSet contaning the spatial averages for all the storm configurations.

    Notes
    -----
    https://doi.org/10.1016/j.ejrh.2017.07.003.
    """
    dict_config = {
        "1": [[0], [0]],
        "2.1": [[0, 0], [0, 1]],
        "2.2": [[0, 1], [0, 0]],
        "3.1": [[0, 1, 1], [0, 0, 1]],
        "3.2": [[1, 1, 0], [0, 1, 1]],
        "3.3": [[0, 0, 1], [0, 1, 1]],
        "3.4": [[0, 1, 0], [0, 0, 1]],
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

    # Pixel size
    dy = (da.y[1] - da.y[0]).values
    dx = (da.x[1] - da.x[0]).values

    # Number of pixels in da
    npy_da = len(da.y)
    npx_da = len(da.x)

    spt_av_list = []
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
        ny = len(da.y) - npy_confi + 1
        nx = len(da.x) - npx_confi + 1

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

        for shift, confi_shifted in enumerate(shifted_confi):
            matrix_mask = np.full((len(da.y), len(da.x)), np.nan)
            matrix_mask[(confi_shifted[0], confi_shifted[1])] = 1
            da_mask = da * matrix_mask
            da_mean = da_mask.mean(dim=["x", "y"]).expand_dims(
                dim={"conf": [name + "_" + str(shift)]}
            )

            spt_av_list.append(da_mean)

    spt_av = xr.concat(spt_av_list, dim="conf")

    if "units" in da.attrs:
        spt_av.attrs["units"] = da.attrs["units"]

    if path is not None:
        spt_av.to_zarr(path)

    return spt_av
