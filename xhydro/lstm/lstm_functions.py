import os

import numpy as np

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from utilsS.create_datasets import (
    clean_nans,
    create_dataset_flexible,
    create_LSTM_dataset,
    create_LSTM_dataset_catchment_vary,
)
from utilsS.LSTM_static import (
    define_LSTM_model,
    nse_loss,
    run_trained_model,
    training_generator,
)


def GetNetcdfTags():
    Qsim_pos = [False, False, False, False, False, True]

    dynamic_var_tags = ["tasmax_MELCC", "tasmin_MELCC", "sd", "sf", "rf", "Qsim"]

    static_var_tags = [
        "drainage_area",
        "elevation",
        "slope",
        "water",
        "deciduous_forest",
        "agricultural_pasture",
        "coniferous_forest",
        "impervious",
        "bog",
        "wet_land",
        "loamy_sand",
        "loam",
        "silt",
        "silty_clay_loam",
        "meanPrecip",
        "meanSWE",
        "meanPET",
        "snowFraction",
        "aridity",
        "highPrecipFreq",
        "lowPrecipFreq",
        "highPrecipDuration",
        "lowPrecipDuration",
        "centroid_lat",
        "centroid_lon",
    ]

    return dynamic_var_tags, Qsim_pos, static_var_tags


def ScaleDataset(
    input_data_filename,
    dynamic_var_tags,
    Qsim_pos,
    static_var_tags,
    train_pct,
    valid_pct,
):
    # %%Load and pre-process the dataset
    """
    Prepare the LSTM features (inputs) and target variable (output):
    """
    # Create the dataset
    arr_dynamic, arr_static, arr_Qobs = create_dataset_flexible(
        input_data_filename, dynamic_var_tags, Qsim_pos, static_var_tags
    )

    """
    Filter catchments with too many NaNs
    """
    # Find which catchments have less than 10 years of data
    # (20% of 10 = 2 years for valid/testing) and delete them
    for i in reversed(range(0, arr_Qobs.shape[0])):
        if np.count_nonzero(~np.isnan(arr_Qobs[i, :])) < 10 * 365:
            arr_dynamic = np.delete(arr_dynamic, i, 0)
            arr_static = np.delete(arr_static, i, 0)
            arr_Qobs = np.delete(arr_Qobs, i, 0)

    """
    Get the indexes of the train, test and valid periods of each catchment.
    """
    train_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)
    valid_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)
    test_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)
    all_idx = np.empty([arr_dynamic.shape[0], 2], dtype=int)

    for i in range(0, arr_dynamic.shape[0]):
        jj = np.argwhere(~np.isnan(arr_Qobs[i, :]))
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
    q_stds = np.nanstd(arr_Qobs, axis=1)

    # Scale the dynamic feature variables (not the target variable)
    scaler_dynamic = StandardScaler()  # Use standardization by mean and std

    # Prepare data for scaling. Must rebuild a vector from the training data length instead of constants here due to
    # differing lengths of data per catchment. See Matlab code in this folder to see how it was performed.
    ndata = arr_dynamic.shape[2] - 1
    dynamic_data = np.empty(
        (0, ndata)
    )  # 7 is with including Qsim as predictor and snow

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


def SplitDataset(
    arr_dynamic, arr_static, q_stds, watersheds_ind, train_idx, window_size, valid_idx
):
    # %%
    # Training dataset
    X_train, X_train_static, X_train_q_stds, y_train = (
        create_LSTM_dataset_catchment_vary(
            arr_dynamic=arr_dynamic,
            arr_static=arr_static,
            q_stds=q_stds,
            window_size=window_size,
            watershed_list=watersheds_ind,
            idx=train_idx,
        )
    )

    # Clean nans
    y_train, X_train, X_train_q_stds, X_train_static = clean_nans(
        y_train, X_train, X_train_q_stds, X_train_static
    )

    # Validation dataset
    X_valid, X_valid_static, X_valid_q_stds, y_valid = (
        create_LSTM_dataset_catchment_vary(
            arr_dynamic=arr_dynamic,
            arr_static=arr_static,
            q_stds=q_stds,
            window_size=window_size,
            watershed_list=watersheds_ind,
            idx=valid_idx,
        )
    )

    # Clean nans
    y_valid, X_valid, X_valid_q_stds, X_valid_static = clean_nans(
        y_valid, X_valid, X_valid_q_stds, X_valid_static
    )

    return (
        X_train,
        X_train_static,
        X_train_q_stds,
        y_train,
        X_valid,
        X_valid_static,
        X_valid_q_stds,
        y_valid,
    )


def obj_fun_kge(Qobs, Qsim):
    """
    # This function computes the Kling-Gupta Efficiency (KGE) criterion
    :param Qobs: Observed streamflow
    :param Qsim: Simulated streamflow
    :return: kge: KGE criterion value
    """
    # Remove all nans from both observed and simulated streamflow
    ind_nan = np.isnan(Qobs)
    Qobs = Qobs[~ind_nan]
    Qsim = Qsim[~ind_nan]

    # Compute the dimensionless correlation coefficient
    r = np.corrcoef(Qsim, Qobs)[0, 1]

    # Compute the dimensionless bias ratio b (beta)
    b = np.mean(Qsim) / np.mean(Qobs)

    # Compute the dimensionless variability ratio g (gamma)
    g = (np.std(Qsim) / np.mean(Qsim)) / (np.std(Qobs) / np.mean(Qobs))

    # Compute the Kling-Gupta Efficiency (KGE) modified criterion
    kge = 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)

    # In some cases, the KGE can return nan values which will force some
    # optimization algorithm to crash. Force the worst value possible instead.
    if np.isnan(kge):
        kge = -np.inf

    return kge


def perform_initial_train(
    useParallel,
    window_size,
    batch_size_val,
    epoch_val,
    X_train,
    X_train_static,
    X_train_q_stds,
    y_train,
    X_valid,
    X_valid_static,
    X_valid_q_stds,
    y_valid,
    name_of_saved_model,
):
    success = 0
    while success == 0:
        K.clear_session()  # Reset the model
        # Define multi-GPU training

        if useParallel == True:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Number of devices: {strategy.num_replicas_in_sync}")
            with strategy.scope():
                model_LSTM, callback = define_LSTM_model(
                    window_size=window_size,
                    n_dynamic_features=X_train.shape[2],
                    n_static_features=X_train_static.shape[1],
                    checkpoint_path=name_of_saved_model,
                )
                print("USING MULTIPLE GPU SETUP")
        else:
            model_LSTM, callback = define_LSTM_model(
                window_size=window_size,
                n_dynamic_features=X_train.shape[2],
                n_static_features=X_train_static.shape[1],
                checkpoint_path=name_of_saved_model,
            )
            print("USING SINGLE GPU")

        h = model_LSTM.fit(
            training_generator(
                X_train,
                X_train_static,
                X_train_q_stds,
                y_train,
                batch_size=batch_size_val,
            ),
            epochs=epoch_val,
            validation_data=training_generator(
                X_valid,
                X_valid_static,
                X_valid_q_stds,
                y_valid,
                batch_size=batch_size_val,
            ),
            callbacks=[callback],
            verbose=1,
        )

        if not np.isnan(h.history["loss"][-1]):
            success = 1

    """
    Using the model to generate flows on each individual period + full period
    """
    plot_model(model_LSTM, "AAA_Model_Structure.jpg", show_shapes=True)


def RunModelAfterTraining(
    w,
    arr_dynamic,
    arr_static,
    q_stds,
    window_size,
    train_idx,
    batch_size_val,
    watershed_areas,
    filename_base,
    name_of_saved_model,
    valid_idx,
    test_idx,
    all_idx,
):

    # Run trained model on the training period and save outputs
    run_trained_model(
        arr_dynamic,
        arr_static,
        q_stds,
        window_size,
        w,
        train_idx,
        batch_size_val,
        watershed_areas,
        file_ID="training",
        filename_base=filename_base,
        name_of_saved_model=name_of_saved_model,
        postdate=False,
        cleanNans=True,
    )

    # Run trained model on the validation period and save outputs
    run_trained_model(
        arr_dynamic,
        arr_static,
        q_stds,
        window_size,
        w,
        valid_idx,
        batch_size_val,
        watershed_areas,
        file_ID="validation",
        filename_base=filename_base,
        name_of_saved_model=name_of_saved_model,
        postdate=False,
        cleanNans=True,
    )

    # Run the trained model on the testing period and save outputs
    run_trained_model(
        arr_dynamic,
        arr_static,
        q_stds,
        window_size,
        w,
        test_idx,
        batch_size_val,
        watershed_areas,
        file_ID="testing",
        filename_base=filename_base,
        name_of_saved_model=name_of_saved_model,
        postdate=False,
        cleanNans=True,
    )

    # Run the trained model on the full period and save outputs
    run_trained_model(
        arr_dynamic,
        arr_static,
        q_stds,
        window_size,
        w,
        all_idx,
        batch_size_val,
        watershed_areas,
        file_ID="full-period",
        filename_base=filename_base,
        name_of_saved_model=name_of_saved_model,
        postdate=False,
        cleanNans=False,
    )
