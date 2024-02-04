import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xhydro.lstm_tools.create_datasets import (
    clean_nans_func,
    create_dataset_flexible,
    create_lstm_dataset,
)
from xhydro.lstm_tools.LSTM_static import (
    TrainingGenerator,
    define_lstm_model_simple,
    run_trained_model,
)


def scale_dataset(
    input_data_filename,
    dynamic_var_tags,
    qsim_pos,
    static_var_tags,
    train_pct,
    valid_pct,
):
    # %%Load and pre-process the dataset
    """
    Prepare the LSTM features (inputs) and target variable (output):
    """
    # Create the dataset
    arr_dynamic, arr_static, arr_qobs = create_dataset_flexible(
        input_data_filename, dynamic_var_tags, qsim_pos, static_var_tags
    )

    """
    Filter catchments with too many NaNs
    """
    # Find which catchments have less than 10 years of data
    # (20% of 10 = 2 years for valid/testing) and delete them
    for i in reversed(range(0, arr_qobs.shape[0])):
        if np.count_nonzero(~np.isnan(arr_qobs[i, :])) < 10 * 365:
            arr_dynamic = np.delete(arr_dynamic, i, 0)
            arr_static = np.delete(arr_static, i, 0)
            arr_qobs = np.delete(arr_qobs, i, 0)

    """
    Get the indexes of the train, test and valid periods of each catchment.
    """
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


def split_dataset(
    arr_dynamic, arr_static, q_stds, watersheds_ind, train_idx, window_size, valid_idx
):
    # %%
    # Training dataset
    x_train, x_train_static, x_train_q_stds, y_train = create_lstm_dataset(
        arr_dynamic=arr_dynamic,
        arr_static=arr_static,
        q_stds=q_stds,
        window_size=window_size,
        watershed_list=watersheds_ind,
        idx=train_idx,
    )

    # Clean nans
    y_train, x_train, x_train_q_stds, x_train_static = clean_nans_func(
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

    # Clean nans
    y_valid, x_valid, x_valid_q_stds, x_valid_static = clean_nans_func(
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


def perform_initial_train(
    use_parallel,
    window_size,
    batch_size,
    epochs,
    X_train,
    X_train_static,
    X_train_q_stds,
    y_train,
    X_valid,
    X_valid_static,
    X_valid_q_stds,
    y_valid,
    name_of_saved_model,
    use_cpu=False,
):
    """Train the LSTM model using preprocessed data."""
    success = 0
    while success == 0:
        k.clear_session()  # Reset the model
        # Define multi-GPU training
        if use_parallel and not use_cpu:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Number of devices: {strategy.num_replicas_in_sync}")
            with strategy.scope():
                model_lstm, callback = define_lstm_model_simple(
                    window_size=window_size,
                    n_dynamic_features=X_train.shape[2],
                    n_static_features=X_train_static.shape[1],
                    checkpoint_path=name_of_saved_model,
                )
                print("USING MULTIPLE GPU SETUP")
        else:
            model_lstm, callback = define_lstm_model_simple(
                window_size=window_size,
                n_dynamic_features=X_train.shape[2],
                n_static_features=X_train_static.shape[1],
                checkpoint_path=name_of_saved_model,
            )
            print("USING SINGLE GPU")

        if use_cpu:
            tf.config.set_visible_devices([], "GPU")

        h = model_lstm.fit(
            TrainingGenerator(
                X_train,
                X_train_static,
                X_train_q_stds,
                y_train,
                batch_size=batch_size,
            ),
            epochs=epochs,
            validation_data=TrainingGenerator(
                X_valid,
                X_valid_static,
                X_valid_q_stds,
                y_valid,
                batch_size=batch_size,
            ),
            callbacks=[callback],
            verbose=1,
        )

        if not np.isnan(h.history["loss"][-1]):
            success = 1


def run_model_after_training(
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
):

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
            clean_nans=False,
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
            clean_nans=False,
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
            clean_nans=False,
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
            clean_nans=False,
        )
    else:
        kge_full = None
        flows_full = None

    return [kge_train, kge_valid, kge_test, kge_full], [
        flows_train,
        flows_valid,
        flows_test,
        flows_full,
    ]
