"""
This script allows to calibrate and test an LSTM model on a dataset of 96 watersheds from the INFO-Crue project.
This particular version of the script uses variables from the MELCC gridded dataset as well as the ERA5 reanalysis.
"""

# %% Import packages
import os
import tempfile

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
from .lstm_functions import (
    GetNetcdfTags,
    RunModelAfterTraining,
    ScaleDataset,
    SplitDataset,
    perform_initial_train,
)


# %% Control variables for all experiments
def control_LSTM_training(
    input_data_filename,
    batch_size=32,
    epochs=200,
    window_size=365,
    train_pct=60,
    valid_pct=20,
    use_cpu=True,
    use_parallel=False,
    do_train=True,
    do_simulation=True,
    filename_base="LSTM_results",
    simulation_phases=['train', 'valid', 'test', 'full'],
    name_of_saved_model=None,
):
    """
    All the control variables used to train the LSTM models are predefined here.
    These are consistent from one experiment to the other, but can be modified as
    needed by the user.
    """
    # If we want to use CPU only, deactivate GPU for memory allocation. The order of operations MUST be preserved.
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    import tensorflow as tf
    tf.get_logger().setLevel("INFO")

    if name_of_saved_model is None:
        if not do_train:
            raise ValueError('Model training is set to False and no existing model is provided. Please provide a trained model or set "do_train" to True.')
        tmpdir = tempfile.mkdtemp()
        name_of_saved_model = tmpdir + "/" + filename_base + ".h5"

    # Get NetCDF variables tags and infos to use for model training
    dynamic_var_tags, qsim_pos, static_var_tags = GetNetcdfTags()

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
    ) = ScaleDataset(
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
            X_train,
            X_train_static,
            X_train_q_stds,
            y_train,
            X_valid,
            X_valid_static,
            X_valid_q_stds,
            y_valid,
        ) = SplitDataset(
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
        )

    if do_simulation:
        # Do the model simulation on all watersheds after training
        kge_results = [None] * len(watersheds_ind)
        flow_results = [None] * len(watersheds_ind)

        for w in watersheds_ind:
            kge_results[w], flow_results[w] = RunModelAfterTraining(
                w,
                arr_dynamic,
                arr_static,
                q_stds,
                window_size,
                train_idx,
                batch_size,
                watershed_areas,
                filename_base,
                name_of_saved_model,
                valid_idx,
                test_idx,
                all_idx,
                simulation_phases,
            )

        return kge_results, flow_results, name_of_saved_model

    else:
        return None, None, name_of_saved_model

