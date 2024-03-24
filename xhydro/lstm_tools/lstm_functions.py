"""Collection of functions required to process LSTM models and their required data."""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# TODO: Cannot use star imports, pre-commit screams.
from xhydro.lstm_tools.create_datasets import (
    create_dataset_flexible,
    create_dataset_flexible_local,
    create_lstm_dataset,
    create_lstm_dataset_local,
    remove_nans_func,
    remove_nans_func_local,
)
from xhydro.lstm_tools.LSTM_static import (
    TrainingGenerator,
    TrainingGeneratorLocal,
    get_list_of_LSTM_models,
    run_trained_model,
    run_trained_model_local,
)

__all__ = [
    "perform_initial_train",
    "perform_initial_train_local",
    "run_model_after_training",
    "run_model_after_training_local",
    "scale_dataset",
    "scale_dataset_local",
    "split_dataset",
    "split_dataset_local",
]


def scale_dataset(
    input_data_filename: str,
    dynamic_var_tags: list,
    qsim_pos: list,
    static_var_tags: list,
    train_pct: int,
    valid_pct: int,
):
    """Scale the datasets using training data to normalize all inputs, ensuring weighting is unbiased.

    Parameters
    ----------
    input_data_filename : str
        Path to the netcdf file containing the required input and target data for the LSTM. The ncfile must contain a
        dataset named "qobs" and "drainage_area" for the code to work, as these are required as target and scaling for
        training, respectively.
    dynamic_var_tags : list of str
        List of dataset variables to use in the LSTM model training. Must be part of the input_data_filename ncfile.
    qsim_pos : list of bool
        List of same length as dynamic_var_tags. Should be set to all False EXCEPT where the dynamic_var_tags refer to
        flow simulations (ex: simulations from a hydrological model such as HYDROTEL). Those should be set to True.
    static_var_tags : list of str
        List of the catchment descriptor names in the input_data_filename ncfile. They need to be present in the ncfile
        and will be used as inputs to the regional model, to help the flow regionalization process.
    train_pct : int
        Percentage of days from the dataset to use as training. The higher, the better for model training skill, but it
        is important to keep a decent amount for validation and testing.
    valid_pct : int
        Percentage of days from the dataset to use as validation. The sum of train_pct and valid_pct needs to be less
        than 100, such that the remainder can be used for testing. A good starting value is 20%. Validation is used as
        the stopping criteria during training. When validation stops improving, then the model is overfitting and
        training is stopped.

    Returns
    -------
    watershed_areas : np.ndarray
        Area of the watershed, in square kilometers, as taken from the training dataset initial input ncfile.
    watersheds_ind : np.ndarray
        List of watershed indices to use during training.
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    arr_static : np.array
        Tensor of size [time_steps x n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training.
    q_stds : np.ndarray
        Tensor of size [time_steps] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in arr_dynamic and arr_static.
    train_idx : np.ndarray
        Indices of the training period from the complete period. Contains 2 values per watershed: start and end indices.
    valid_idx : np.ndarray
        Indices of the validation period from the complete period. Contains 2 values per watershed: start and end
        indices.
    test_idx : np.ndarray
        Indices of the testing period from the complete period. Contains 2 values per watershed: start and end indices.
    all_idx : np.ndarray
        Indices of the full period. Contains 2 values per watershed: start and end indices.
    """
    # Create the dataset
    arr_dynamic, arr_static, arr_qobs = create_dataset_flexible(
        input_data_filename, dynamic_var_tags, qsim_pos, static_var_tags
    )

    # TODO : This should be a user-selected option
    # Filter catchments with too many NaNs.
    # Find which catchments have less than 10 years of data
    # (20% of 10 = 2 years for valid/testing) and delete them
    for i in reversed(range(0, arr_qobs.shape[0])):
        if np.count_nonzero(~np.isnan(arr_qobs[i, :])) < 10 * 365:
            arr_dynamic = np.delete(arr_dynamic, i, 0)
            arr_static = np.delete(arr_static, i, 0)
            arr_qobs = np.delete(arr_qobs, i, 0)

    # Get the indexes of the train, test and valid periods of each catchment.
    train_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)
    valid_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)
    test_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)
    all_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)

    for i in range(0, arr_dynamic.shape[0]):
        jj = np.argwhere(~np.isnan(arr_qobs[i, :]))
        total_number_days = np.shape(jj)[0]
        number_training_days = int(np.floor(total_number_days * train_pct / 100))
        number_valid_days = int(np.floor(total_number_days * valid_pct / 100))

        train_idx[i, 0] = int(jj[0])  # From this index...
        train_idx[i, 1] = int(jj[number_training_days])  # to this index.
        valid_idx[i, 0] = int(jj[number_training_days])  # From this index...
        valid_idx[i, 1] = int(
            jj[number_training_days + number_valid_days]
        )  # to this index.
        test_idx[i, 0] = int(
            jj[number_training_days + number_valid_days]
        )  # From this index...
        test_idx[i, 1] = int(jj[total_number_days - 1])  # to this index.
        all_idx[i, 0] = int(0)
        all_idx[i, 1] = int(jj[total_number_days - 1])

    # Get watershed numbers list
    watersheds_ind = np.arange(0, arr_dynamic.shape[0])

    # Get catchment areas, required for testing later to transform flows back into m3/s
    watershed_areas = arr_static[:, 0]

    # Get the standard deviation of the streamflow for all watersheds
    q_stds = np.nanstd(arr_qobs, axis=1)

    # Scale the dynamic feature variables (not the target variable)
    scaler_dynamic = StandardScaler()  # Use standardization by mean and std

    # Prepare data for scaling. Must rebuild a vector from the training data length instead of constants here due to
    # differing lengths of data per catchment. See Matlab code in this folder to see how it was performed.
    n_data = arr_dynamic.shape[2] - 1
    dynamic_data = np.empty((0, n_data))

    for tmp in range(0, arr_dynamic.shape[0]):
        dynamic_data = np.vstack(
            [dynamic_data, arr_dynamic[tmp, train_idx[tmp, 0] : train_idx[tmp, 1], 1:]]
        )

    # Fit the scaler using only the training watersheds
    _ = scaler_dynamic.fit_transform(dynamic_data)

    # Scale all watersheds data
    for w in watersheds_ind:
        arr_dynamic[w, :, 1:] = scaler_dynamic.transform(arr_dynamic[w, :, 1:])

    # Scale the static feature variables
    scaler_static = MinMaxScaler()  # Use normalization between 0 and 1
    _ = scaler_static.fit_transform(arr_static)
    arr_static = scaler_static.transform(arr_static)

    return (
        watershed_areas,
        watersheds_ind,
        arr_dynamic,
        arr_static,
        q_stds,
        train_idx,
        valid_idx,
        test_idx,
        all_idx,
    )


def scale_dataset_local(
    input_data_filename: str,
    dynamic_var_tags: list,
    qsim_pos: list,
    train_pct: int,
    valid_pct: int,
):
    """Scale the datasets using training data to normalize all inputs, ensuring weighting is unbiased.

    Parameters
    ----------
    input_data_filename : str
        Path to the netcdf file containing the required input and target data for the LSTM. The ncfile must contain a
        dataset named "qobs" and "drainage_area" for the code to work, as these are required as target and scaling for
        training, respectively.
    dynamic_var_tags : list of str
        List of dataset variables to use in the LSTM model training. Must be part of the input_data_filename ncfile.
    qsim_pos : list of bool
        List of same length as dynamic_var_tags. Should be set to all False EXCEPT where the dynamic_var_tags refer to
        flow simulations (ex: simulations from a hydrological model such as HYDROTEL). Those should be set to True.
    train_pct : int
        Percentage of days from the dataset to use as training. The higher, the better for model training skill, but it
        is important to keep a decent amount for validation and testing.
    valid_pct : int
        Percentage of days from the dataset to use as validation. The sum of train_pct and valid_pct needs to be less
        than 100, such that the remainder can be used for testing. A good starting value is 20%. Validation is used as
        the stopping criteria during training. When validation stops improving, then the model is overfitting and
        training is stopped.

    Returns
    -------
    watershed_areas : np.ndarray
        Area of the watershed, in square kilometers, as taken from the training dataset initial input ncfile.
    watersheds_ind : np.ndarray
        List of watershed indices to use during training.
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    train_idx : np.ndarray
        Indices of the training period from the complete period. Contains 2 values per watershed: start and end indices.
    valid_idx : np.ndarray
        Indices of the validation period from the complete period. Contains 2 values per watershed: start and end
        indices.
    test_idx : np.ndarray
        Indices of the testing period from the complete period. Contains 2 values per watershed: start and end indices.
    all_idx : np.ndarray
        Indices of the full period. Contains 2 values per watershed: start and end indices.
    """
    # Create the dataset
    arr_dynamic, arr_qobs = create_dataset_flexible_local(
        input_data_filename, dynamic_var_tags, qsim_pos
    )

    # Get the indexes of the train, test and valid periods of each catchment.
    train_idx = np.empty([2], dtype=int)
    valid_idx = np.empty([2], dtype=int)
    test_idx = np.empty([2], dtype=int)
    all_idx = np.empty([2], dtype=int)

    # Count how may non-nan qobs data exists.
    jj = np.argwhere(~np.isnan(arr_qobs))
    total_number_days = np.shape(jj)[0]
    number_training_days = int(np.floor(total_number_days * train_pct / 100))
    number_valid_days = int(np.floor(total_number_days * valid_pct / 100))

    train_idx[0] = int(jj[0])  # From this index...
    train_idx[1] = int(jj[number_training_days])  # to this index.
    valid_idx[0] = int(jj[number_training_days])  # From this index...
    valid_idx[1] = int(jj[number_training_days + number_valid_days])  # to this index.
    test_idx[0] = int(
        jj[number_training_days + number_valid_days]
    )  # From this index...
    test_idx[1] = int(jj[total_number_days - 1])  # to this index.
    all_idx[0] = int(0)
    all_idx[1] = arr_qobs.shape[0]

    dynamic_data = arr_dynamic[train_idx[0] : train_idx[1], 1:]

    # Fit the scaler using only the training watersheds
    scaler_dynamic = StandardScaler()  # Use standardization by mean and std
    _ = scaler_dynamic.fit_transform(dynamic_data)

    # Scale all watersheds data
    arr_dynamic[:, 1:] = scaler_dynamic.transform(arr_dynamic[:, 1:])

    return (
        arr_dynamic,
        train_idx,
        valid_idx,
        test_idx,
        all_idx,
    )


def split_dataset(
    arr_dynamic: np.array,
    arr_static: np.array,
    q_stds: np.array,
    watersheds_ind: np.array,
    train_idx: np.array,
    window_size: int,
    valid_idx: np.array,
):
    """Extract only the required data from the entire dataset according to the desired period.

    Parameters
    ----------
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    arr_static : np.ndarray
        Tensor of size [time_steps x n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training.
    q_stds : np.ndarray
        Tensor of size [time_steps] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in arr_dynamic and arr_static.
    watersheds_ind : np.ndarray
        List of watershed indices to use during training.
    train_idx : np.ndarray
        Indices of the training period from the complete period. Contains 2 values per watershed: start and end indices.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    valid_idx : np.ndarray
        Indices of the validation period from the complete period. Contains 2 values per watershed: start and end
        indices.

    Returns
    -------
    x_train : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during training.
    x_train_static : np.ndarray
        Tensor of size [(timesteps * watersheds) x n_static_variables] that contains the static (i.e. catchment
        descriptors) variables that will be used during training.
    x_train_q_stds : np.ndarray
        Tensor of size [(timesteps * watersheds)] that contains the standard deviation of scaled streamflow values for
        the catchment associated to the data in x_train and x_train_static. Each data point could come from any
        catchment and this x_train_q_std variable helps scale the objective function.
    y_train : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_train,
        x_train_static and x_train_q_stds. Usually the observed streamflow for the day associated to each of the
        training points.
    x_valid : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during validation.
    x_valid_static : np.ndarray
        Tensor of size [(timesteps * watersheds) x n_static_variables] that contains the static (i.e. catchment
        descriptors) variables that will be used during validation.
    x_valid_q_stds : np.ndarray
        Tensor of size [(timesteps * watersheds)] that contains the standard deviation of scaled streamflow values for
        the catchment associated to the data in x_valid and x_valid_static. Each data point could come from any
        catchment and this x_valid_q_std variable helps scale the objective function for the validation points.
    y_valid : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_valid,
        x_valid_static and x_valid_q_stds. Usually the observed streamflow for the day associated to each of the
        validation points.
    """
    # Training dataset
    x_train, x_train_static, x_train_q_stds, y_train = create_lstm_dataset(
        arr_dynamic=arr_dynamic,
        arr_static=arr_static,
        q_stds=q_stds,
        window_size=window_size,
        watershed_list=watersheds_ind,
        idx=train_idx,
    )

    # Remove Nans
    y_train, x_train, x_train_q_stds, x_train_static = remove_nans_func(
        y_train, x_train, x_train_q_stds, x_train_static
    )

    # Validation dataset
    x_valid, x_valid_static, x_valid_q_stds, y_valid = create_lstm_dataset(
        arr_dynamic=arr_dynamic,
        arr_static=arr_static,
        q_stds=q_stds,
        window_size=window_size,
        watershed_list=watersheds_ind,
        idx=valid_idx,
    )

    # Remove nans
    y_valid, x_valid, x_valid_q_stds, x_valid_static = remove_nans_func(
        y_valid, x_valid, x_valid_q_stds, x_valid_static
    )

    return (
        x_train,
        x_train_static,
        x_train_q_stds,
        y_train,
        x_valid,
        x_valid_static,
        x_valid_q_stds,
        y_valid,
    )


def split_dataset_local(
    arr_dynamic: np.ndarray,
    train_idx: np.ndarray,
    window_size: int,
    valid_idx: np.ndarray,
):
    """Extract only the required data from the entire dataset according to the desired period.

    Parameters
    ----------
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    train_idx : np.ndarray
        Indices of the training period from the complete period. Contains 2 values per watershed: start and end indices.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    valid_idx : np.ndarray
        Indices of the validation period from the complete period. Contains 2 values per watershed: start and end
        indices.

    Returns
    -------
    x_train : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during training.
    y_train : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_train,
        x_train_static and x_train_q_stds. Usually the observed streamflow for the day associated to each of the
        training points.
    x_valid : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during validation.
    y_valid : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_valid,
        x_valid_static and x_valid_q_stds. Usually the observed streamflow for the day associated to each of the
        validation points.
    """
    # Training dataset
    x_train, y_train = create_lstm_dataset_local(
        arr_dynamic=arr_dynamic, window_size=window_size, idx=train_idx
    )

    # Remove nans
    y_train, x_train = remove_nans_func_local(y_train, x_train)

    # Validation dataset
    x_valid, y_valid = create_lstm_dataset_local(
        arr_dynamic=arr_dynamic, window_size=window_size, idx=valid_idx
    )

    # Remove nans
    y_valid, x_valid = remove_nans_func_local(y_valid, x_valid)

    return x_train, y_train, x_valid, y_valid


def perform_initial_train(
    model_structure: str,
    use_parallel: bool,
    window_size: int,
    batch_size: int,
    epochs: int,
    x_train: np.ndarray,
    x_train_static: np.ndarray,
    x_train_q_stds: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    x_valid_static: np.ndarray,
    x_valid_q_stds: np.ndarray,
    y_valid: np.ndarray,
    name_of_saved_model: str,
    training_func: str,
    use_cpu: bool = False,
):
    """Train the LSTM model using preprocessed data.

    Parameters
    ----------
    model_structure : str
        The version of the LSTM model that we want to use to apply to our data. Must be the name of a function that
        exists in LSTM_static.py.
    use_parallel : bool
        Flag to make use of multiple GPUs to accelerate training further. Models trained on multiple GPUs can have
        larger batch_size values as different batches can be run on different GPUs in parallel. Speedup is not linear as
        there is overhead related to the management of datasets, batches, the gradient merging and other steps. Still
        very useful and should be used when possible.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    epochs : int
        Number of training evaluations. Larger number of epochs means more model iterations and deeper training. At some
        point, training will stop due to a stop in validation skill improvement.
    x_train : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during training.
    x_train_static : np.ndarray
        Tensor of size [(timesteps * watersheds) x n_static_variables] that contains the static (i.e. catchment
        descriptors) variables that will be used during training.
    x_train_q_stds : np.ndarray
        Tensor of size [(timesteps * watersheds)] that contains the standard deviation of scaled streamflow values for
        the catchment associated to the data in x_train and x_train_static. Each data point could come from any
        catchment and this x_train_q_std variable helps scale the objective function.
    y_train : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_train,
        x_train_static and x_train_q_stds. Usually the observed streamflow for the day associated to each of the
        training points.
    x_valid : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during validation.
    x_valid_static : np.ndarray
        Tensor of size [(timesteps * watersheds) x n_static_variables] that contains the static (i.e. catchment
        descriptors) variables that will be used during validation.
    x_valid_q_stds : np.ndarray
        Tensor of size [(timesteps * watersheds)] that contains the standard deviation of scaled streamflow values for
        the catchment associated to the data in x_valid and x_valid_static. Each data point could come from any
        catchment and this x_valid_q_std variable helps scale the objective function for the validation points.
    y_valid : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_valid,
        x_valid_static and x_valid_q_stds. Usually the observed streamflow for the day associated to each of the
        validation points.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.
    training_func : str
        Name of the objective function used for training. For a regional model, it is highly recommended to use the
        scaled nse_loss variable that uses the standard deviation of streamflow as inputs.
    use_cpu : bool
        Flag to force the training and simulations to be performed on the CPU rather than on the GPU(s). Must be
        performed on a CPU that has AVX and AVX2 instruction sets, or tensorflow will fail. CPU training is very slow
        and should only be used as a last resort (such as for CI testing and debugging).
    """
    success = 0
    while success == 0:
        k.clear_session()  # Reset the model
        # Define multi-GPU training
        if use_parallel and not use_cpu:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Number of devices: {strategy.num_replicas_in_sync}")
            with strategy.scope():
                model_func = get_list_of_LSTM_models(model_structure)
                model_lstm, callback = model_func(
                    window_size=window_size,
                    n_dynamic_features=x_train.shape[2],
                    n_static_features=x_train_static.shape[1],
                    training_func=training_func,
                    checkpoint_path=name_of_saved_model,
                )
                print("USING MULTIPLE GPU SETUP")
        else:
            model_func = get_list_of_LSTM_models(model_structure)
            model_lstm, callback = model_func(
                window_size=window_size,
                n_dynamic_features=x_train.shape[2],
                n_static_features=x_train_static.shape[1],
                training_func=training_func,
                checkpoint_path=name_of_saved_model,
            )
            if use_cpu:
                print("USING CPU")
            else:
                print("USING SINGLE GPU")

        if use_cpu:
            tf.config.set_visible_devices([], "GPU")

        h = model_lstm.fit(
            TrainingGenerator(
                x_train,
                x_train_static,
                x_train_q_stds,
                y_train,
                batch_size=batch_size,
            ),
            epochs=epochs,
            validation_data=TrainingGenerator(
                x_valid,
                x_valid_static,
                x_valid_q_stds,
                y_valid,
                batch_size=batch_size,
            ),
            callbacks=[callback],
            verbose=1,
        )

        if not np.isnan(h.history["loss"][-1]):
            success = 1


def perform_initial_train_local(
    model_structure: str,
    use_parallel: bool,
    window_size: int,
    batch_size: int,
    epochs: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    name_of_saved_model: str,
    training_func: str,
    use_cpu: bool = False,
):
    """Train the LSTM model using preprocessed data on a local catchment.

    Parameters
    ----------
    model_structure : str
        The version of the LSTM model that we want to use to apply to our data. Must be the name of a function that
        exists in LSTM_static.py.
    use_parallel : bool
        Flag to make use of multiple GPUs to accelerate training further. Models trained on multiple GPUs can have
        larger batch_size values as different batches can be run on different GPUs in parallel. Speedup is not linear as
        there is overhead related to the management of datasets, batches, the gradient merging and other steps. Still
        very useful and should be used when possible.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    epochs : int
        Number of training evaluations. Larger number of epochs means more model iterations and deeper training. At some
        point, training will stop due to a stop in validation skill improvement.
    x_train : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during training.
    y_train : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_train,
        x_train_static and x_train_q_stds. Usually the observed streamflow for the day associated to each of the
        training points.
    x_valid : np.ndarray
        Tensor of size [(timesteps * watersheds) x window_size x n_dynamic_variables] that contains the dynamic (i.e.
        timeseries) variables that will be used during validation.
    y_valid : np.ndarray
        Tensor of size [(timesteps * watersheds)] containing the target variable for the same time point as in x_valid,
        x_valid_static and x_valid_q_stds. Usually the observed streamflow for the day associated to each of the
        validation points.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.
    training_func : str
        Name of the objective function used for training. For a regional model, it is highly recommended to use the
        scaled nse_loss variable that uses the standard deviation of streamflow as inputs.
    use_cpu : bool
        Flag to force the training and simulations to be performed on the CPU rather than on the GPU(s). Must be
        performed on a CPU that has AVX and AVX2 instruction sets, or tensorflow will fail. CPU training is very slow
        and should only be used as a last resort (such as for CI testing and debugging).

    Returns
    -------
    code
        Adding this just because linter will not let me put nothing. Exits with 0 if all is normal.
    """
    success = 0
    while success == 0:
        k.clear_session()  # Reset the model
        # Define multi-GPU training
        if use_parallel and not use_cpu:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Number of devices: {strategy.num_replicas_in_sync}")
            with strategy.scope():
                model_func = get_list_of_LSTM_models(model_structure)
                model_lstm, callback = model_func(
                    window_size=window_size,
                    n_dynamic_features=x_train.shape[2],
                    training_func=training_func,
                    checkpoint_path=name_of_saved_model,
                )
                print("USING MULTIPLE GPU SETUP")
        else:
            model_func = get_list_of_LSTM_models(model_structure)
            model_lstm, callback = model_func(
                window_size=window_size,
                n_dynamic_features=x_train.shape[2],
                training_func=training_func,
                checkpoint_path=name_of_saved_model,
            )
            if use_cpu:
                print("USING CPU")
            else:
                print("USING SINGLE GPU")

        if use_cpu:
            tf.config.set_visible_devices([], "GPU")

        h = model_lstm.fit(
            TrainingGeneratorLocal(
                x_train,
                y_train,
                batch_size=batch_size,
            ),
            epochs=epochs,
            validation_data=TrainingGeneratorLocal(
                x_valid,
                y_valid,
                batch_size=batch_size,
            ),
            callbacks=[callback],
            verbose=1,
        )

        if not np.isnan(h.history["loss"][-1]):
            success = 1

    return 0


def run_model_after_training(
    w: int,
    arr_dynamic: np.ndarray,
    arr_static: np.ndarray,
    q_stds: np.ndarray,
    window_size: int,
    train_idx: np.ndarray,
    batch_size: int,
    watershed_areas: np.ndarray,
    name_of_saved_model: str,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    all_idx: np.ndarray,
    simulation_phases: list,
):
    """Simulate streamflow on given input data for a user-defined number of periods.

    Parameters
    ----------
    w : int
        Number of the watershed from the list of catchments that will be simulated.
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    arr_static : np.ndarray
        Tensor of size [time_steps x n_static_variables] that contains the static (i.e. catchment descriptors) variables
        that will be used during training.
    q_stds : np.ndarray
        Tensor of size [time_steps] that contains the standard deviation of scaled streamflow values for the catchment
        associated to the data in arr_dynamic and arr_static.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    train_idx : np.ndarray
        Indices of the training period from the complete period. Contains 2 values per watershed: start and end indices.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    watershed_areas : np.ndarray
        Area of the watershed, in square kilometers, as taken from the training dataset initial input ncfile.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.
    valid_idx : np.ndarray
        Indices of the validation period from the complete period. Contains 2 values per watershed: start and end
        indices.
    test_idx : np.ndarray
        Indices of the testing period from the complete period. Contains 2 values per watershed: start and end indices.
    test_idx : np.ndarray
        Indices of the testing period from the complete period. Contains 2 values per watershed: start and end indices.
    all_idx : np.ndarray
        Indices of the full period. Contains 2 values per watershed: start and end indices.
    simulation_phases : list of str
        List of periods to generate the simulations. Can contain ['train','valid','test','full'], corresponding to the
        training, validation, testing and complete periods, respectively.

    Returns
    -------
    kge : list
        A list of size 4, with one float per period in ['train','valid','test','all']. Each KGE value is comupted
        between observed and simulated flows for the watershed of interest and for all specified periods. Unrequested
        periods return None.
    flows : list
        A list of np.ndarray objects of size 4, with one 2D np.ndarray per period in ['train','valid','test','all'].
        Observed (column 1) and simulated (column 2) streamflows are computed for the watershed of interest and for all
        specified periods. Unrequested periods return None.
    """
    # Run trained model on the training period and save outputs
    if "train" in simulation_phases:
        kge_train, flows_train = run_trained_model(
            arr_dynamic,
            arr_static,
            q_stds,
            window_size,
            w,
            train_idx,
            batch_size,
            watershed_areas,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_train = None
        flows_train = None

    if "valid" in simulation_phases:
        # Run trained model on the validation period and save outputs
        kge_valid, flows_valid = run_trained_model(
            arr_dynamic,
            arr_static,
            q_stds,
            window_size,
            w,
            valid_idx,
            batch_size,
            watershed_areas,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_valid = None
        flows_valid = None

    if "test" in simulation_phases:
        # Run the trained model on the testing period and save outputs
        kge_test, flows_test = run_trained_model(
            arr_dynamic,
            arr_static,
            q_stds,
            window_size,
            w,
            test_idx,
            batch_size,
            watershed_areas,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_test = None
        flows_test = None

    if "full" in simulation_phases:
        # Run the trained model on the full period and save outputs
        kge_full, flows_full = run_trained_model(
            arr_dynamic,
            arr_static,
            q_stds,
            window_size,
            w,
            all_idx,
            batch_size,
            watershed_areas,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_full = None
        flows_full = None

    kge = [kge_train, kge_valid, kge_test, kge_full]
    flows = [flows_train, flows_valid, flows_test, flows_full]

    return kge, flows


def run_model_after_training_local(
    arr_dynamic: np.ndarray,
    window_size: int,
    train_idx: np.ndarray,
    batch_size: int,
    name_of_saved_model: str,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    all_idx: np.ndarray,
    simulation_phases: list,
):
    """Simulate streamflow on given input data for a user-defined number of periods.

    Parameters
    ----------
    arr_dynamic : np.ndarray
        Tensor of size [time_steps x window_size x (n_dynamic_variables+1)] that contains the dynamic (i.e. time-series)
        variables that will be used during training. The first element in axis=2 is the observed flow.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    train_idx : np.ndarray
        Indices of the training period from the complete period. Contains 2 values per watershed: start and end indices.
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.
    valid_idx : np.ndarray
        Indices of the validation period from the complete period. Contains 2 values per watershed: start and end
        indices.
    test_idx : np.ndarray
        Indices of the testing period from the complete period. Contains 2 values per watershed: start and end indices.
    test_idx : np.ndarray
        Indices of the testing period from the complete period. Contains 2 values per watershed: start and end indices.
    all_idx : np.ndarray
        Indices of the full period. Contains 2 values per watershed: start and end indices.
    simulation_phases : list of str
        List of periods to generate the simulations. Can contain ['train','valid','test','full'], corresponding to the
        training, validation, testing and complete periods, respectively.

    Returns
    -------
    kge : list
        A list of floats of size 4, with one float per period in ['train','valid','test','all']. Each KGE value is
        comupted between observed and simulated flows for the watershed of interest and for all specified periods.
        Unrequested periods return None.
    flows : list
        A list of np.ndarray objects of size 4, with one 2D np.ndarray per period in ['train','valid','test','all'].
        Observed (column 1) and simulated (column 2) streamflows are computed for the watershed of interest and for all
        specified periods. Unrequested periods return None.
    """
    # Run trained model on the training period and save outputs
    if "train" in simulation_phases:
        kge_train, flows_train = run_trained_model_local(
            arr_dynamic,
            window_size,
            train_idx,
            batch_size,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_train = None
        flows_train = None

    if "valid" in simulation_phases:
        # Run trained model on the validation period and save outputs
        kge_valid, flows_valid = run_trained_model_local(
            arr_dynamic,
            window_size,
            valid_idx,
            batch_size,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_valid = None
        flows_valid = None

    if "test" in simulation_phases:
        # Run the trained model on the testing period and save outputs
        kge_test, flows_test = run_trained_model_local(
            arr_dynamic,
            window_size,
            test_idx,
            batch_size,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_test = None
        flows_test = None

    if "full" in simulation_phases:
        # Run the trained model on the full period and save outputs
        kge_full, flows_full = run_trained_model_local(
            arr_dynamic,
            window_size,
            all_idx,
            batch_size,
            name_of_saved_model=name_of_saved_model,
            remove_nans=False,
        )
    else:
        kge_full = None
        flows_full = None

    kge = [kge_train, kge_valid, kge_test, kge_full]
    flows = [flows_train, flows_valid, flows_test, flows_full]

    return kge, flows
