"""Module to compute Probable Maximum Precipitation (PMP)"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
import xclim
from lmoments3.distr import gev
from xclim.indices.stats import fit, parametric_quantile


def major_precipitation_events(da, acc_day, quantil=0.9, path=None):
    """Extracts precipitation events that exceed a given quantile
       for a given days accumulation.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the precipitation values.
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
    if "conf" in list(da.coords):
        core_dims = ["time", "y", "x", "conf"]
        da = da.expand_dims(dim={"x": [0], "y": [0]}).copy(deep=True)
    else:
        core_dims = ["time", "y", "x"]

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


def precipitable_water(ds_day, ds_fx, acc_day=[1], path=None):
    """Computes the precipitable water for the antecedent conditions
    given the accumulations days.

    Parameters
    ----------
    ds_day : xr.Dataset
        Dataset containing the Specific humidity (hus) and the Geopotential Height (zg).
    ds_fx : xr.Dataset
        Dataset containing the Surface altitude (orog).
    acc_day : list
        List of precipitation accumulation days.
    path: str, optional
        Path where the results will be saved.

    Returns
    -------
    da_pw : xr.DataArray
        Precipitable water.

    """
    # Correction of the first zone above the topography to consider the correct fraction.
    da = ds_day["zg"] - ds_fx["orog"]
    da = da.where(da >= 0)
    ds_day["zg_corr"] = (da + ds_fx["orog"]).fillna(ds_fx["orog"])

    # Betas computation for zones equal to zero.
    db = da.where(da >= 0, 0)
    ds_day["beta"] = db.where(db <= 0, 1)

    # Evaluation of average elevation for each pressure level, including beta coefficient
    ds_day["zg1"] = ds_day.zg_corr.diff("plev", label="upper")
    ds_day["zg2"] = ds_day.zg_corr.diff("plev", label="lower")
    ds_day["zg_moy"] = ds_day[["zg1", "zg2"]].to_array().sum("variable") / 2
    ds_day["zg_moy"] = ds_day["zg_moy"] * ds_day["beta"]

    # This gives precipitable water for the entire history. You can then juxtapose with Pevent to return only PWevent (max).
    ds_day["pt"] = (ds_day["zg_moy"] * ds_day["hus"]).sum("plev")
    pw = ds_day["pt"].rename("precipitable_water")

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

    if da_pw is not None:
        da_pw.to_zarr(path)

    return da_pw


def rolling_max(arr, window_size):
    """Computes the maximum rolling value.

    Parameters
    ----------
    arr : array_like
        Array to create the sliding window view from.
    window_size : xr.Dataset
        Size of window over each axis that takes part in the
        sliding window.

    Returns
    -------
    roll_max_out : ndarray
        Rolling maximum of the array. The function includes NaN values
        at the beginning of the array to keep the same length of
        the input array.

    """
    windowed = np.lib.stride_tricks.sliding_window_view(arr, window_size, axis=0)

    roll_max = np.maximum.reduce(windowed, axis=-1)

    roll_max_out = np.full(arr.shape, np.nan)
    roll_max_out[window_size - 1 :, :, :] = roll_max

    return roll_max_out


def precipitable_water_100y(da_pw, mf=0.2, path=None):
    """Computes the 100-year return period of precipitable water
    for each month of the year. The annual maximums of the precipitable
    water plus a porcentage (mf) is use as a upper linit.

    Parameters
    ----------
    da_pw : xr.DataArray
        Dataset containing the precipitable water.
    mf : float
        Maximization factor.
    path: str, optional
        Path where the results will be saved.

    Returns
    -------
    da_pw100 : xr.DataArray
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
    params = fit(da_pw_m, dist=gev, method="pwm")
    pw100_m = parametric_quantile(params, q=1 - 1 / 100).squeeze().rename("pw100")

    pw_mm = da_pw_m.rename("precipitable_water_monthly")
    pw_mm = pw_mm.rename({"time": "year"})
    pw_mm_mf = (
        pw_mm.max(dim="year") * (1.0 + mf)
    ).squeeze()  # Add a limit to PW100 to limit maximization factors.

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
    pw100_m = pw100_m.rename("pw100").to_dataset().transpose("time", "y", "x")

    expanded_dates = da_pw.drop_vars(["x", "y", "height"])
    da_pw100 = xr.merge([pw100_m, expanded_dates]).ffill(dim="time")

    da_pw100 = da_pw100.pw100.squeeze().drop(
        ["height", "month", "year", "stacked_coords"]
    )

    return da_pw100


def compute_spring_and_summer_mask(snt):
    """Creates a mask that defines the spring and
    summer seasons based on the snow thickness.

    Parameters
    ----------
    snd : xarray.DataArray
        Surface snow thickness.

    Returns
    -------
    mask : xr.Dataset
        Dataset with two DataArrays (mask_spring and mask_summer),
        which have the same snt dimensions with values of 1 where
        the spring and summer criteria are met and 0 where they
        are not.
    """
    if snt.attrs["units"] == "kg m-2":
        snt = snt / 10  # kg/m² == 1mm   --> Transformation en 1mm = 1cm/10
        snt.attrs["units"] = "cm"
    else:
        sys.exit("snow units are not in kg m-2")
        snt = (
            np.nan
        )  # That is put there to make sure the code stops if the previous line does no work...

    winter_start = xclim.indices.snd_season_start(
        snd=snt, thresh="1 cm", window=14, freq="AS-JUL"
    )
    winter_start = winter_start.assign_coords({"year": winter_start.time.dt.year})
    winter_start = winter_start.swap_dims({"time": "year"})
    # 14 days of windows to make sure that the snow stays on the ground at the beginning of winter.
    # freq="AS-JUL" makes the count start in july, so that the start of the year is not a problem. The result is given in julian days.
    # This process adds a year at the beginning of the array, because the first count begins before the first year of the array.
    # The first day of the array is given as one, not as 0 as in python. (We will need to make -1 in the code)

    winter_end = xclim.indices.snd_season_end(
        snd=snt, thresh="1 cm", window=90, freq="YS"
    )
    winter_end = winter_end.assign_coords({"year": winter_end.time.dt.year})
    winter_end = winter_end.swap_dims({"time": "year"})
    spring_start = winter_end - 60
    spring_start = spring_start.where(spring_start >= 1, 1)
    spring_end = winter_end + 30

    # 90 days of window because we want to make sure it is not a thaw.
    # freq='YS' because we make the hypothesis that winter always ends after january.
    # The results give the first days of summer, and not the last days of winter. (We will need to make -2 in the code...)

    array_days = xr.ones_like(snt) * snt.time.dt.dayofyear
    # Make an array of julian days the same size as «snt»

    mask = xr.where(winter_start.isnull(), np.nan, 1)
    mask = mask.drop_sel(year=max(winter_start.time.dt.year))
    mask["year"] = mask.year + 1  # --> Not a good practice, to change ASAP
    winter_end = winter_end * mask
    array_winter_end = xr.ones_like(snt).groupby(snt.time.dt.year) * (winter_end)
    # A winter that never started cannot finish, so we look for nan in the julian_days_start and put put them in the julian_days_end
    # We need to make a +1 year because the winter_start start a year earlier.

    condition1 = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        winter_start.drop_vars("time")
    )
    # Make an array of the same size as «snt» with the julian days of start of winter
    # This array exclude the year that was added at the begining of the array. We call it condition 1.
    mask = xr.where(condition1 >= 182, 1, np.nan)
    condition1 = condition1 * mask
    # Put nan when the winter start before july 1rst

    winter_start["year"] = (
        winter_start.year + 1
    )  # --> Not a good practice, to change ASAP
    condition2 = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        winter_start.drop_vars("time")
    )
    # Make an array of the same size as «snt» with the julian days of start of winter
    # This array exclude the last year. We call it condition 2.
    mask = xr.where(condition2 < 182, 1, np.nan)
    condition2 = condition2 * mask
    # Put nan when the winter start after july 1rst

    a1 = array_days >= condition1
    b1 = array_days < array_winter_end
    c1 = a1 + b1
    winter_mask_half1 = xr.where(c1 == 1, 1, 0)

    a2 = array_days >= condition2
    a2 = xr.where(a2 == True, 1, 0)
    b2 = array_days < array_winter_end
    b2 = xr.where(b2 == True, 1, 0)
    c2 = a2 + b2
    winter_mask_half2 = xr.where(
        c2 == 2, 1, 0
    )  # It's normal that it is 2: consdition a2 and b2 must be met...

    winter_mask = (winter_mask_half1 + winter_mask_half2).drop_vars("year")
    winter_mask = xr.where(
        winter_mask >= 1, 1, np.nan
    )  # It's >=1 because sommetimes conditions 1 and 2 are met at the same time.

    summer_mask = xr.where(winter_mask.isnull(), 1, np.nan)

    # calculer le masque de printemps
    array_days = xr.ones_like(snt) * snt.time.dt.dayofyear
    array_days_end_of_spring = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        spring_end.drop(["time"])
    )
    array_days_start_of_spring = xr.ones_like(snt).groupby(snt.time.dt.year) * (
        spring_start.drop(["time"])
    )

    condition3 = array_days >= array_days_start_of_spring
    condition4 = array_days <= array_days_end_of_spring

    c3 = xr.where(condition3 == True, 1, 0)
    c4 = xr.where(condition4 == True, 1, 0)
    c = c3 + c4
    spring_mask = xr.where(c == 2, 1, 0)

    mask = xr.Dataset({"mask_spring": spring_mask, "mask_summer": summer_mask})

    return mask


def spatial_storm_aggregation(da, center_x, center_y, radius):
    """
    """
    xx, yy = np.meshgrid(da.coords["x"].values, da.coords["y"].values)
    coor_x = xx.flatten()
    coor_y = yy.flatten()

    sq_dist = (coor_x - center_x) ** 2 + (coor_y - center_y) ** 2
    within_r_ind = np.where(sq_dist <= radius**2)

    within_r_x = coor_x[within_r_ind]
    within_r_y = coor_y[within_r_ind]

    da_cut = da.where((da.x.isin(within_r_x)) & (da.y.isin(within_r_y)), drop=True)

    dict_config = {  # y, x
        "1": [[0], [0]],
        "2": [[0, 0], [0, 1]],
        "3": [[0, 0, -1], [0, 1, 0]],
        "4": [[0, 0, -1, -1], [0, 1, 0, 1]],
        "5.a": [[0, 0, -1, -1, 0], [0, 1, 0, 1, -1]],
        "5.b": [[0, 0, 0, 1, 1], [0, 1, -1, 0, 1]],
        "6": [[0, 0, 0, 1, 1, 1], [0, 1, -1, 0, 1, -1]],
        "7": [[0, 0, 0, -1, -1, -1, 1], [0, 1, -1, 0, 1, -1, 0]],
        "8": [[0, 0, 0, -1, -1, -1, 1, 1], [0, 1, -1, 0, 1, -1, 0, 1]],
        "9": [[0, 0, 0, -1, -1, -1, 1, 1, 1], [0, 1, -1, 0, 1, -1, 0, 1, -1]],
        "10.a": [[0, 0, 0, 0, -1, -1, -1, -1, 1, 1], [-1, 0, 1, 2, -1, 0, 1, 2, 0, 1]],
        "10.b": [
            [0, 0, 0, 0, -1, -1, -1, -1, 1, 1],
            [-2, -1, 0, 1, -2, -1, 0, 1, -1, 0],
        ],
        "12": [
            [0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1],
            [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
        ],
        "14.a": [
            [0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2],
            [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, 0, 1],
        ],
        "14.b.": [
            [0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2],
            [-2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -1, 0],
        ],
        "16": [
            [0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2, 2, 2],
            [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
        ],
        "18.a": [
            [0, 0, 0, 0, -2, -2, -2, -2, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2],
            [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, 0, 1],
        ],
        "18.b": [
            [0, 0, 0, 0, -2, -2, -2, -2, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2],
            [-2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -1, 0],
        ],
        "20": [
            [0, 0, 0, 0, -2, -2, -2, -2, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2, 2, 2],
            [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
        ],
        '23': [[-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2],
               [-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-1,0,1]
        ],
        '24.a': [[-2,-2,-2,-2,-1,-1,-1,-1,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],
                 [-1,0,1,2,-1,0,1,2,-1,0,1,2,-1,0,1,2,-1,0,1,2,-1,0,1,2]
        ],
        '24.b': [[-2,-2,-2,-2,-1,-1,-1,-1,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],
                 [-2,-1,0,1,-2,-1,0,1,-2,-1,0,1,-2,-1,0,1,-2,-1,0,1,-2,-1,0,1]
        ],
        '25': [[-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],
               [-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2]
        ],
    }

    da_cut_c = da_cut[0, :, :].copy(deep=True) * np.nan

    da_for_coor = da_cut_c.where(
        ~((da_cut_c.x == center_x) & (da_cut_c.y == center_y)), 1
    )
    centers_pos = np.argwhere(da_for_coor.values == 1)

    ds_mean = []
    for name, confi in dict_config.items():

        cent_pos_y = confi[0] + centers_pos[0][0]
        cent_pos_x = confi[1] + centers_pos[0][1]

        matrix_mask = da_cut_c.values.copy()

        try:
            matrix_mask[cent_pos_y, cent_pos_x] = 1
        except:
            continue

        for rot in range(4):
            matrix_mask_rot = np.rot90(matrix_mask, k=rot)
            da_pr = da_cut * matrix_mask_rot
            ds_mean.append(
                da_pr.mean(dim=["x", "y"]).expand_dims(
                    dim={"conf": [name + "_" + str(rot)]}
                )
            )

            if np.array_equal(np.rot90(matrix_mask, k=1), matrix_mask, equal_nan=True):
                break

    return xr.concat(ds_mean, dim="time")
