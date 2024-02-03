# -*- coding: utf-8 -*-
"""
This script allows to calibrate and test an LSTM model on a dataset of 96 watersheds from the INFO-Crue project.
This particular version of the script uses variables from the MELCC gridded dataset as well as the ERA5 reanalysis.
"""

# %% Import packages
import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import tensorflow as tf
from lstm_utils.create_datasets import plot_boxplot_results
from lstm_functions import (RunModelAfterTraining, 
                            perform_initial_train, 
                            GetNetcdfTags,
                            SplitDataset, 
                            ScaleDataset,
                            )

tf.get_logger().setLevel('INFO')

# %% Control variables for all experiments
def control_LSTM_training(
        input_data_filename,
        batch_size=32,
        epochs=200,
        window_size=365,
        train_pct=60,
        valid_pct=20,
        run_tag='LSTM_MODEL',
        use_parallel=False,
        do_train=True,
        do_simulation=True,
        results_path="./",
        filename_base='LSTM_results_',
        ):
    """
    All the control variables used to train the LSTM models are predefined here.
    These are consistent from one experiment to the other, but can be modified as 
    needed by the user.
    """        
    if do_train:
        # Clean folder of old runs before starting
        for filename in Path(".").glob("*.h5"):
            os.remove(filename)
        for filename in Path(".").glob("*.jpg"):
            os.remove(filename)
        for filename in Path(".").glob("*.txt"):
            os.remove(filename)
        for filename in Path(".").glob("*.png"):
            os.remove(filename)
        
    name_of_saved_model = filename_base + '.h5'

    # Get NetCDF variables tags and infos to use for model training
    dynamic_var_tags, Qsim_pos, static_var_tags = GetNetcdfTags()
    
    # Import and scale dataset
    watershed_areas, watersheds_ind, arr_dynamic, arr_static, q_stds, train_idx, valid_idx, test_idx, all_idx = (
        ScaleDataset(input_data_filename, dynamic_var_tags, Qsim_pos, static_var_tags, train_pct, valid_pct))
    
    if do_train:
        # Split into train and valid
        X_train, X_train_static, X_train_q_stds, y_train, X_valid, X_valid_static, X_valid_q_stds, y_valid=(
            SplitDataset(arr_dynamic, arr_static, q_stds, watersheds_ind, train_idx, window_size, valid_idx))
    
        # Do the main large-scale training
        perform_initial_train(use_parallel, window_size, batch_size, epochs, X_train,
                                                              X_train_static, X_train_q_stds, y_train, X_valid,
                                                              X_valid_static, X_valid_q_stds, y_valid, name_of_saved_model)
    
    if do_simulation:
        # Do the model simulation on all watersheds after training
        for w in watersheds_ind:
            RunModelAfterTraining(w, arr_dynamic, arr_static, q_stds, window_size,
                                                               train_idx, batch_size, watershed_areas, filename_base,
                                                               name_of_saved_model, valid_idx, test_idx, all_idx)
    
    # Load results and make boxplots
    plot_boxplot_results(watersheds_ind, filename_base)
    
    
    
    
    


