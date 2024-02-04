"""

"""

import numpy as np
import xarray as xr


def create_dataset_flexible(filename, dynamic_var_tags, qsim_pos, static_var_tags):
    """
    This function will prepare the arrays of dynamic, static and observed flow
    variables. A few things are absolutely required:
        1. a "watershed" variable that contains the ID of watersheds, such that
           we can preallocate the size of the matrices.
        2. a "Qobs" variable that contains observed flows for the catchments.
        3. The size of the catchments in the "drainage_area" tag. This is used
           to compute scaled streamflow values for regionalization applications

    "dynamic_var_tags" should contain the tags of the timeseries data (dynamic)
    in the form of a list.

    "static_var_tags" should contain the tags of the catchment characteristics
    in the form of a list.

    "Qsim_pos" is an array or list of Booleans on which we want to scale with
    drainage area. Same size as dynamic_var_tags

    # Need to add checks according to possible inputs and required keywords,
    empty arrays, etc.
    """

    # Open the dataset file
    ds = xr.open_dataset(filename)

    # Number of watersheds in the dataset
    n_watersheds = ds.watershed.shape[0]

    # Number of days for each dynamic variables
    # of watersheds.
    n_days = ds.Qobs.values.shape[1]

    # Perform the analysis for Qobs first.
    arr_qobs = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_qobs[:] = np.nan
    arr_qobs = ds.Qobs.values
    arr_qobs = arr_qobs / ds.drainage_area.values[:, np.newaxis] * 86.4

    # Prepare the dynamic data array and set Qobs as the first value
    arr_dynamic = np.empty(
        shape=[n_watersheds, n_days, len(dynamic_var_tags) + 1], dtype=np.float32
    )
    arr_dynamic[:] = np.nan
    arr_dynamic[:, :, 0] = arr_qobs

    for i in range(len(dynamic_var_tags)):

        # Prepare tmp var to store data
        tmp = np.empty([n_watersheds, n_days], dtype=np.float32)
        tmp[:] = np.nan

        # Read the data and put in the formatted tmp var
        tmp = ds[dynamic_var_tags[i]].values

        # If the variable must be scaled, do it
        if qsim_pos[i]:
            tmp = tmp / ds.drainage_area.values[:, np.newaxis] * 86.4

        # Set the data in the main dataset
        arr_dynamic[:, :, i + 1] = tmp

    # Prepare the static dataset
    arr_static = np.empty([n_watersheds, len(static_var_tags)], dtype=np.float32)
    arr_static[:] = np.nan

    # Loop the
    for i in range(len(static_var_tags)):
        arr_static[:, i] = ds[static_var_tags[i]].values

    return arr_dynamic, arr_static, arr_qobs


def create_lstm_dataset(
    arr_dynamic,
    arr_static,
    q_stds,
    window_size,
    watershed_list,
    idx,
    clean_nans=True,
):

    ndata = arr_dynamic.shape[2] - 1
    x = np.empty((0, window_size, ndata))  # 7 is with Qsim as predictor and snow
    x_static = np.empty((0, arr_static.shape[1]))
    x_q_stds = np.empty(0)
    y = np.empty(0)

    for w in watershed_list:
        idx_w = idx[w]
        print("Currently working on watershed no: " + str(w))
        x_w, x_w_static, x_w_q_std, y_w = extract_watershed_block(
            arr_w_dynamic=arr_dynamic[w, idx_w[0] : idx_w[1], :],
            arr_w_static=arr_static[w, :],
            q_std_w=q_stds[w],
            window_size=window_size,
        )

        # Clean nans
        if clean_nans:
            y_w, x_w, x_w_q_std, x_w_static = clean_nans_func(
                y_w, x_w, x_w_q_std, x_w_static
            )

        x = np.vstack([x, x_w])
        x_static = np.vstack([x_static, x_w_static])
        x_q_stds = np.hstack([x_q_stds, x_w_q_std])
        y = np.hstack([y, y_w])

    return x, x_static, x_q_stds, y


def extract_windows_vectorized(array, sub_window_size):

    max_time = array.shape[0]

    # expand_dims are used to convert a 1D array to 2D array.
    sub_windows = (
        np.expand_dims(np.arange(sub_window_size), 0)
        + np.expand_dims(np.arange(max_time - sub_window_size), 0).T
    )

    return array[sub_windows]


def extract_watershed_block(arr_w_dynamic, arr_w_static, q_std_w, window_size):
    """
    This function extracts all series of the desired window length over all
    features for a given watershed. Both dynamic and static variables are
    extracted.
    """
    # Extract all series of the desired window length for all features
    x_w = extract_windows_vectorized(array=arr_w_dynamic, sub_window_size=window_size)

    x_w_static = np.repeat(arr_w_static.reshape(-1, 1), x_w.shape[0], axis=1).T

    x_w_q_std = np.squeeze(np.repeat(q_std_w.reshape(-1, 1), x_w.shape[0], axis=1).T)

    # Get the last value of Qobs from each series for the prediction
    y_w = x_w[:, -1, 0]

    # Remove Qobs from the features
    x_w = np.delete(x_w, 0, axis=2)

    return x_w, x_w_static, x_w_q_std, y_w


def clean_nans_func(y, x, x_q_std, x_static):
    """
    Check for nans in the variable "y" and remove all lines containing those nans in all datasets
    """
    ind_nan = np.isnan(y)
    y = y[~ind_nan]
    x = x[~ind_nan, :, :]
    x_q_stds = x_q_std[~ind_nan]
    x_static = x_static[~ind_nan, :]

    return y, x, x_q_stds, x_static
