"""Tools to create the datasets to be used in LSTM model training and simulation."""

import numpy as np
import xarray as xr


def create_dataset_flexible(
    filename: str,
    dynamic_var_tags: list,
    qsim_pos: list,
    static_var_tags: list,
):
    """Prepare the arrays of dynamic, static and observed flow variables.

    A few things are absolutely required:
        1. a "watershed" variable that contains the ID of watersheds, such that
           we can preallocate the size of the matrices.
        2. a "Qobs" variable that contains observed flows for the catchments.
        3. The size of the catchments in the "drainage_area" tag. This is used
           to compute scaled streamflow values for regionalization applications.

    Parameters
    ----------
    filename : str
        Path to the netcdf file containing the required input and target data for the LSTM. The ncfile must contain a
        dataset named "Qobs" and "Drainage_area" for the code to work, as these are required as target and scaling for
        training, respectively.
    dynamic_var_tags : list of str
        List of dataset variables to use in the LSTM model training. Must be part of the input_data_filename ncfile.
    qsim_pos : list of bool
        List of same length as dynamic_var_tags. Should be set to all False EXCEPT where the dynamic_var_tags refer to
        flow simulations (ex: simulations from a hydrological model such as HYDROTEL). Those should be set to True.
    static_var_tags : list of str
        List of the catchment descriptor names in the input_data_filename ncfile. They need to be present in the ncfile
        and will be used as inputs to the regional model, to help the flow regionalization process.

    Returns
    -------
    arr_dynamic : np.array
        Tensor of size [watersheds x timestep x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    arr_static : np.array
        Tensor of size [watersheds x n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training.
    q_stds : np.array
        Tensor of size [watersheds] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in arr_dynamic and arr_static.
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


def create_dataset_flexible_local(
    filename: str,
    dynamic_var_tags: list,
    qsim_pos: list,
):
    """Prepare the arrays of dynamic and observed flow variables.

    A few things are absolutely required:
        1. a "watershed" variable that contains the ID of watersheds, such that
           we can preallocate the size of the matrices.
        2. a "Qobs" variable that contains observed flows for the catchments.

    Parameters
    ----------
    filename : str
        Path to the netcdf file containing the required input and target data for the LSTM. The ncfile must contain a
        dataset named "Qobs" and "Drainage_area" for the code to work, as these are required as target and scaling for
        training, respectively.
    dynamic_var_tags : list of str
        List of dataset variables to use in the LSTM model training. Must be part of the input_data_filename ncfile.
    qsim_pos : list of bool
        List of same length as dynamic_var_tags. Should be set to all False EXCEPT where the dynamic_var_tags refer to
        flow simulations (ex: simulations from a hydrological model such as HYDROTEL). Those should be set to True.

    Returns
    -------
    arr_dynamic : np.array
        Tensor of size [watersheds x timestep x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=1 is the observed flow.
    arr_qobs : np.array
        Array containing the observed flow vector.
    """
    # Open the dataset file
    ds = xr.open_dataset(filename)

    # Number of days for each dynamic variables
    # of watersheds.
    n_days = ds.Qobs.values.shape[0]

    # Perform the analysis for Qobs first.
    arr_qobs = np.empty([n_days], dtype=np.float32)
    arr_qobs[:] = np.nan
    arr_qobs = ds.Qobs.values

    # Prepare the dynamic data array and set Qobs as the first value
    arr_dynamic = np.empty(shape=[n_days, len(dynamic_var_tags) + 1], dtype=np.float32)
    arr_dynamic[:] = np.nan
    arr_dynamic[:, 0] = arr_qobs

    for i in range(len(dynamic_var_tags)):

        # Prepare tmp var to store data
        tmp = np.empty([n_days], dtype=np.float32)
        tmp[:] = np.nan

        # Read the data and put in the formatted tmp var
        tmp = ds[dynamic_var_tags[i]].values

        # If the variable must be scaled, do it
        if qsim_pos[i]:
            tmp = tmp / ds.drainage_area.values * 86.4

        # Set the data in the main dataset
        arr_dynamic[:, i + 1] = tmp

    return arr_dynamic, arr_qobs


def create_lstm_dataset(
    arr_dynamic: np.array,
    arr_static: np.array,
    q_stds: np.array,
    window_size: int,
    watershed_list: list,
    idx: np.array,
    clean_nans: bool = True,
):
    """Create the LSTM dataset and shape the data using look-back windows and preparing all data for training.

    Parameters
    ----------
    arr_dynamic : np.array
        Tensor of size [watersheds x timestep x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    arr_static : np.array
        Tensor of size [watersheds x n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training.
    q_stds : np.array
        Tensor of size [watersheds] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in arr_dynamic and arr_static.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow. Usually
        set to 365 days to get one year of previous data. This makes the model heavier and longer to train but can improve
        results.
    watershed_list : list
        List of the total number of watersheds that will be used for training and simulation. Corresponds to the watershed
        in the input file, i.e. in the arr_dynamic array axis 0.
    idx : np.array
        2-element array of indices of the beginning and end of the desired period for which the LSTM model should be simulated.
    clean_nans : bool
        Flag indicating that the NaN values associated to the observed streamflow should be removed. Required for training
        but can be kept to False for simulation to ensure simulation on the entire period.

    Returns
    -------
    x : np.array
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e. time-
        series) variables that will be used during training.
    x_static : np.array
        Tensor of size [(timesteps * watersheds) x n_static_variables] that contains the static (i.e. catchment descriptors)
        variables that will be used during training.
    x_q_stds : np.array
        Tensor of size [(timesteps * watersheds)] that contains the standard deviation of scaled streamflow values for
        the catchment associated to the data in x and x_static. Each data point could come from any catchment and this
        q_std variable helps scale the objective function.
    y : np.array
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x,
        x_static and x_q_stds. Usually the observed streamflow for the day associated to each of the training points.
    """
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


def create_lstm_dataset_local(
    arr_dynamic: np.array,
    window_size: int,
    idx: np.array,
    clean_nans: bool = True,
):
    """Create the local LSTM dataset and shape the data using look-back windows and preparing all data for training.

    Parameters
    ----------
    arr_dynamic : np.array
        Tensor of size [watersheds x timestep x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow. Usually
        set to 365 days to get one year of previous data. This makes the model heavier and longer to train but can improve
        results.
    idx : np.array
        2-element array of indices of the beginning and end of the desired period for which the LSTM model should be simulated.
    clean_nans : bool
        Flag indicating that the NaN values associated to the observed streamflow should be removed. Required for training
        but can be kept to False for simulation to ensure simulation on the entire period.

    Returns
    -------
    x : np.array
        Tensor of size [timesteps x window_size x n_dynamic_variables] that contains the dynamic (i.e. time-
        series) variables that will be used during training.
    y : np.array
        Tensor of size [timesteps] containing the target variable for the same time point as in x,
        x_static and x_q_stds. Usually the observed streamflow for the day associated to each of the training points.
    """
    x, y = extract_watershed_block_local(
        arr_dynamic=arr_dynamic[idx[0] : idx[1], :],
        window_size=window_size,
    )

    # Clean nans
    if clean_nans:
        y, x = clean_nans_func_local(y, x)

    return x, y


def extract_windows_vectorized(
    array: np.array,
    sub_window_size: int,
):
    """Create the array where each day contains data from a previous period (look-back period).

    Parameters
    ----------
    array : np.array
        The array of dynamic variables for a single catchment.
    sub_window_size : int
        Size of the look-back window.

    Returns
    -------
    data_array
        Array of dynamic data processed into a 3D tensor for LSTM model training.
    """
    max_time = array.shape[0]

    # expand_dims are used to convert a 1D array to 2D array.
    sub_windows = (
        np.expand_dims(np.arange(sub_window_size), 0)
        + np.expand_dims(np.arange(max_time - sub_window_size), 0).T
    )
    data_array = array[sub_windows]

    return data_array


def extract_watershed_block(
    arr_w_dynamic: np.array, arr_w_static: np.array, q_std_w: np.array, window_size: int
):
    """Extract all series of the desired window length over all features for a given watershed.

    Create the LSTM tensor format of data from the regular input arrays. Both dynamic and static variables are
    extracted.

    Parameters
    ----------
    arr_w_dynamic : np.array
        Tensor of size [timestep x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training for the current catchment. The first element in axis=1 is the
        observed flow.
    arr_w_static : np.array
        Tensor of size [n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training for the current catchment.
    q_std_w : np.array
        Tensor of size [1] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the current catchment.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow. Usually
        set to 365 days to get one year of previous data. This makes the model heavier and longer to train but can improve
        results.

    Returns
    -------
    x_w : np.array
        Tensor of size [timesteps x window_size x n_dynamic_variables] that contains the dynamic (i.e. time-series)
        variables that will be used during training for a single processed catchment.
    x_w_static : np.array
        Tensor of size [timesteps x n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training for a single processed catchment.
    x_w_q_stds : np.array
        Tensor of size [timesteps] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in x and x_static for a single processed catchment.
    y_w : np.array
        Tensor of size [timesteps] containing the target variable for the same time point as in x_w, x_w_static and
        x_w_q_stds. Usually the observed streamflow for the day associated to each of the training points for the
        currently processed catchment.
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


def extract_watershed_block_local(arr_dynamic: np.array, window_size: int):
    """Extract all series of the desired window length over all features for a given watershed.

    Create the LSTM tensor format of data from the regular input arrays. Both dynamic and static variables are
    extracted.

    Parameters
    ----------
    arr_dynamic : np.array
        Tensor of size [timestep x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training for the current catchment. The first element in axis=1 is the
        observed flow.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow. Usually
        set to 365 days to get one year of previous data. This makes the model heavier and longer to train but can improve
        results.

    Returns
    -------
    x : np.array
        Tensor of size [timesteps x window_size x n_dynamic_variables] that contains the dynamic (i.e. time-series)
        variables that will be used during training for a the catchment.
    y : np.array
        Tensor of size [timesteps] containing the target variable for the same time point as in 'x'. Usually the
        observed streamflow for the day associated to each of the training points.
    """
    # Extract all series of the desired window length for all features
    x = extract_windows_vectorized(array=arr_dynamic, sub_window_size=window_size)

    # Get the last value of Qobs from each series for the prediction
    y = x[:, -1, 0]

    # Remove Qobs from the features
    x = np.delete(x, 0, axis=1)

    return x, y


def clean_nans_func(y: np.array, x: np.array, x_q_std: np.array, x_static: np.array):
    """Check for nans in the variable "y" and remove all lines containing those nans in all datasets.

    Parameters
    ----------
    y : np.array
        Array of target variables for training, that might contain NaNs.
    x : np.array
        Array of dynamic variables for LSTM model training and simulation.
    x_q_std : np.array
        Array of observed streamflow standard deviations for catchments in regional LSTM models.
    x_static : np.array
        Array of static variables for LSTM model training and simulation, specifically for regional LSTM models.

    Returns
    -------
    y : np.array
        Array of target variables for training, cleaned of any NaNs.
    x : np.array
        Array of dynamic variables for LSTM model training and simulation, with values associated to NaN "y" values
        removed.
    x_q_std : np.array
        Array of observed streamflow standard deviations for catchments in regional LSTM models, with values associated
        to NaN "y" values removed.
    x_static : np.array
        Array of static variables for LSTM model training and simulation, specifically for regional LSTM models, with
        values associated to NaN "y" values removed.
    """
    ind_nan = np.isnan(y)
    y = y[~ind_nan]
    x = x[~ind_nan, :, :]
    x_q_stds = x_q_std[~ind_nan]
    x_static = x_static[~ind_nan, :]

    return y, x, x_q_stds, x_static


def clean_nans_func_local(y: np.array, x: np.array):
    """Check for nans in the variable "y" and remove all lines containing those nans in all datasets.

    Parameters
    ----------
    y : np.array
        Array of target variables for training, that might contain NaNs.
    x : np.array
        Array of dynamic variables for LSTM model training and simulation.

    Returns
    -------
    y : np.array
        Array of target variables for training, cleaned of any NaNs.
    x : np.array
        Array of dynamic variables for LSTM model training and simulation, with values associated to NaN "y" values
        removed.
    """
    ind_nan = np.isnan(y)
    y = y[~ind_nan]
    x = x[~ind_nan, :, :]

    return y, x
