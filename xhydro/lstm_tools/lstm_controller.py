"""Control the LSTM training and simulation tools to make clean workflows."""

import os
import tempfile
from pathlib import Path

from xhydro.lstm_tools.lstm_functions import (
    perform_initial_train,
    perform_initial_train_local,
    run_model_after_training,
    run_model_after_training_local,
    scale_dataset,
    scale_dataset_local,
    split_dataset,
    split_dataset_local,
)


def control_regional_lstm_training(
    input_data_filename: str,
    dynamic_var_tags: list,
    qsim_pos: list,
    static_var_tags: list,
    batch_size: int = 32,
    epochs: int = 200,
    window_size: int = 365,
    train_pct: int = 60,
    valid_pct: int = 20,
    use_cpu: bool = True,
    use_parallel: bool = False,
    do_train: bool = True,
    model_structure: str = "dummy_regional_lstm",
    do_simulation: bool = True,
    training_func: str = "nse_scaled",
    filename_base: str = "LSTM_results",
    simulation_phases: list = None,
    name_of_saved_model: str = None,
):
    """Control the regional LSTM model training and simulation.

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
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    epochs : int
        Number of training evaluations. Larger number of epochs means more model iterations and deeper training. At some
        point, training will stop due to a stop in validation skill improvement.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    train_pct : int
        Percentage of days from the dataset to use as training. The higher, the better for model training skill, but it
        is important to keep a decent amount for validation and testing.
    valid_pct : int
        Percentage of days from the dataset to use as validation. The sum of train_pct and valid_pct needs to be less
        than 100, such that the remainder can be used for testing. A good starting value is 20%. Validation is used as
        the stopping criteria during training. When validation stops improving, then the model is overfitting and
        training is stopped.
    use_cpu : bool
        Flag to force the training and simulations to be performed on the CPU rather than on the GPU(s). Must be
        performed on a CPU that has AVX and AVX2 instruction sets, or tensorflow will fail. CPU training is very slow
        and should only be used as a last resort (such as for CI testing and debugging).
    use_parallel : bool
        Flag to make use of multiple GPUs to accelerate training further. Models trained on multiple GPUs can have
        larger batch_size values as different batches can be run on different GPUs in parallel. Speedup is not linear as
        there is overhead related to the management of datasets, batches, the gradient merging and other steps. Still
        very useful and should be used when possible.
    do_train : bool
        Indicate that the code should perform the training step. This is not required as a pre-trained model could be
        used to perform a simulation by passing an existing model in "name_of_saved_model".
    model_structure : str
        The version of the LSTM model that we want to use to apply to our data. Must be the name of a function that
        exists in LSTM_static.py.
    do_simulation : bool
        Indicate that simulations should be performed to obtain simulated streamflow and KGE metrics on the watersheds
        of interest, using the "name_of_saved_model" pre-trained model. If set to True and 'do_train' is True, then the
        new trained model will be used instead.
    training_func : str
        For a regional model, it is highly recommended to use the scaled nse_loss variable that uses the standard
        deviation of streamflow as inputs. For a local model, the "kge" function is preferred. Defaults to "nse_scaled"
        if unspecified by the user. Can be one of ["kge", "nse_scaled"].
    filename_base : str
        Name of the trained model that will be trained if it does not already exist. Do not add the ".h5" extension, it
        will be added automatically.
    simulation_phases : list of str
        List of periods to generate the simulations. Can contain ['train','valid','test','full'], corresponding to the
        training, validation, testing and complete periods, respectively.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.

    Returns
    -------
    kge_results : array-like
        Kling-Gupta Efficiency metric values for each of the watersheds in the input_data_filename ncfile after running
        in simulation mode (thus after training). Contains n_watersheds items, each containing 4 values representing the
        KGE values in training, validation, testing and full period, respectively. If one or more simulation phases are
        not requested, the items will be set to None.
    flow_results : array-like
        Streamflow simulation values for each of the watersheds in the input_data_filename ncfile after running
        in simulation mode (thus after training). Contains n_watersheds items, each containing 4x 2D-arrays representing
        the observed and simulation series in training, validation, testing and full period, respectively. If one or
        more simulation phases are not requested, the items will be set to None.
    name_of_saved_model : str
        Path to the model that has been trained, or to the pre-trained model if it already existed.
    """
    if simulation_phases is None:
        simulation_phases = ["train", "valid", "test", "full"]

    # If we want to use CPU only, deactivate GPU for memory allocation. The order of operations MUST be preserved.
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    tf.get_logger().setLevel("INFO")

    if name_of_saved_model is None:
        if not do_train:
            raise ValueError(
                "Model training is set to False and no existing model is provided. Please provide a "
                'trained model or set "do_train" to True.'
            )
        tmpdir = tempfile.mkdtemp()
        # This needs to be a string for the ModelCheckpoint Callback to work. a PosixPath fails.
        name_of_saved_model = str(Path(tmpdir) / f"{filename_base}.h5")

    # Import and scale dataset
    (
        watershed_areas,
        watersheds_ind,
        arr_dynamic,
        arr_static,
        q_stds,
        train_idx,
        valid_idx,
        test_idx,
        all_idx,
    ) = scale_dataset(
        input_data_filename,
        dynamic_var_tags,
        qsim_pos,
        static_var_tags,
        train_pct,
        valid_pct,
    )

    if do_train:
        # Split into train and valid
        (
            x_train,
            x_train_static,
            x_train_q_stds,
            y_train,
            x_valid,
            x_valid_static,
            x_valid_q_stds,
            y_valid,
        ) = split_dataset(
            arr_dynamic,
            arr_static,
            q_stds,
            watersheds_ind,
            train_idx,
            window_size,
            valid_idx,
        )

        # Do the main large-scale training
        perform_initial_train(
            model_structure,
            use_parallel,
            window_size,
            batch_size,
            epochs,
            x_train,
            x_train_static,
            x_train_q_stds,
            y_train,
            x_valid,
            x_valid_static,
            x_valid_q_stds,
            y_valid,
            name_of_saved_model,
            training_func=training_func,
            use_cpu=use_cpu,
        )

    if do_simulation:
        # Do the model simulation on all watersheds after training
        kge_results = [None] * len(watersheds_ind)
        flow_results = [None] * len(watersheds_ind)

        for w in watersheds_ind:
            kge_results[w], flow_results[w] = run_model_after_training(
                w,
                arr_dynamic,
                arr_static,
                q_stds,
                window_size,
                train_idx,
                batch_size,
                watershed_areas,
                name_of_saved_model,
                valid_idx,
                test_idx,
                all_idx,
                simulation_phases,
            )

        return kge_results, flow_results, name_of_saved_model

    else:
        return None, None, name_of_saved_model


def control_local_lstm_training(
    input_data_filename: str,
    dynamic_var_tags: list,
    qsim_pos: list,
    batch_size: int = 32,
    epochs: int = 200,
    window_size: int = 365,
    train_pct: int = 60,
    valid_pct: int = 20,
    use_cpu: bool = True,
    use_parallel: bool = False,
    do_train: bool = True,
    model_structure: str = "dummy_local_lstm",
    do_simulation: bool = True,
    training_func: str = "kge",
    filename_base: str = "LSTM_results",
    simulation_phases: list = None,
    name_of_saved_model: str = None,
):
    """Control the regional LSTM model training and simulation.

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
    batch_size : int
        Number of data points to use in training. Datasets are often way too big to train in a single batch on a single
        GPU or CPU, meaning that the dataset must be divided into smaller batches. This has an impact on the training
        performance and final model skill, and should be handled accordingly.
    epochs : int
        Number of training evaluations. Larger number of epochs means more model iterations and deeper training. At some
        point, training will stop due to a stop in validation skill improvement.
    window_size : int
        Number of days of look-back for training and model simulation. LSTM requires a large backwards-looking window to
        allow the model to learn from long-term weather patterns and history to predict the next day's streamflow.
        Usually set to 365 days to get one year of previous data. This makes the model heavier and longer to train but
        can improve results.
    train_pct : int
        Percentage of days from the dataset to use as training. The higher, the better for model training skill, but it
        is important to keep a decent amount for validation and testing.
    valid_pct : int
        Percentage of days from the dataset to use as validation. The sum of train_pct and valid_pct needs to be less
        than 100, such that the remainder can be used for testing. A good starting value is 20%. Validation is used as
        the stopping criteria during training. When validation stops improving, then the model is overfitting and
        training is stopped.
    use_cpu : bool
        Flag to force the training and simulations to be performed on the CPU rather than on the GPU(s). Must be
        performed on a CPU that has AVX and AVX2 instruction sets, or tensorflow will fail. CPU training is very slow
        and should only be used as a last resort (such as for CI testing and debugging).
    use_parallel : bool
        Flag to make use of multiple GPUs to accelerate training further. Models trained on multiple GPUs can have
        larger batch_size values as different batches can be run on different GPUs in parallel. Speedup is not linear as
        there is overhead related to the management of datasets, batches, the gradient merging and other steps. Still
        very useful and should be used when possible.
    do_train : bool
        Indicate that the code should perform the training step. This is not required as a pre-trained model could be
        used to perform a simulation by passing an existing model in "name_of_saved_model".
    model_structure : str
        The version of the LSTM model that we want to use to apply to our data. Must be the name of a function that
        exists in LSTM_static.py.
    do_simulation : bool
        Indicate that simulations should be performed to obtain simulated streamflow and KGE metrics on the watersheds
        of interest, using the "name_of_saved_model" pre-trained model. If set to True and 'do_train' is True, then the
        new trained model will be used instead.
    training_func : str
        For a regional model, it is highly recommended to use the scaled nse_loss variable that uses the standard
        deviation of streamflow as inputs. For a local model, the "kge" function is preferred. Defaults to "kge" if
        unspecified by the user. Can be one of ["kge", "nse_scaled"].
    filename_base : str
        Name of the trained model that will be trained if it does not already exist. Do not add the ".h5" extension, it
        will be added automatically.
    simulation_phases : list of str
        List of periods to generate the simulations. Can contain ['train','valid','test','full'], corresponding to the
        training, validation, testing and complete periods, respectively.
    name_of_saved_model : str
        Path to the model that has been pre-trained if required for simulations.

    Returns
    -------
    kge_results : array-like
        Kling-Gupta Efficiency metric values for each of the watersheds in the input_data_filename ncfile after running
        in simulation mode (thus after training). Contains n_watersheds items, each containing 4 values representing the
        KGE values in training, validation, testing and full period, respectively. If one or more simulation phases are
        not requested, the items will be set to None.
    flow_results : array-like
        Streamflow simulation values for each of the watersheds in the input_data_filename ncfile after running
        in simulation mode (thus after training). Contains n_watersheds items, each containing 4x 2D-arrays representing
        the observed and simulation series in training, validation, testing and full period, respectively. If one or
        more simulation phases are not requested, the items will be set to None.
    name_of_saved_model : str
        Path to the model that has been trained, or to the pre-trained model if it already existed.
    """
    if simulation_phases is None:
        simulation_phases = ["train", "valid", "test", "full"]

    # If we want to use CPU only, deactivate GPU for memory allocation. The order of operations MUST be preserved.
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    tf.get_logger().setLevel("INFO")

    if name_of_saved_model is None:
        if not do_train:
            raise ValueError(
                "Model training is set to False and no existing model is provided. Please provide a "
                'trained model or set "do_train" to True.'
            )
        tmpdir = tempfile.mkdtemp()
        # This needs to be a string for the ModelCheckpoint Callback to work. a PosixPath fails.
        name_of_saved_model = str(Path(tmpdir) / f"{filename_base}.h5")

    # Import and scale dataset
    arr_dynamic, train_idx, valid_idx, test_idx, all_idx = scale_dataset_local(
        input_data_filename,
        dynamic_var_tags,
        qsim_pos,
        train_pct,
        valid_pct,
    )

    if do_train:
        # Split into train and valid
        x_train, y_train, x_valid, y_valid = split_dataset_local(
            arr_dynamic, train_idx, window_size, valid_idx
        )

        # Do the main large-scale training
        perform_initial_train_local(
            model_structure,
            use_parallel,
            window_size,
            batch_size,
            epochs,
            x_train,
            y_train,
            x_valid,
            y_valid,
            name_of_saved_model,
            training_func=training_func,
            use_cpu=use_cpu,
        )

    if do_simulation:
        # Do the model simulation on all watersheds after training
        kge_results, flow_results = run_model_after_training_local(
            arr_dynamic,
            window_size,
            train_idx,
            batch_size,
            name_of_saved_model,
            valid_idx,
            test_idx,
            all_idx,
            simulation_phases,
        )

        return kge_results, flow_results, name_of_saved_model

    else:
        return None, None, name_of_saved_model
