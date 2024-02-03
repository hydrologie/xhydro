# -*- coding: utf-8 -*-
"""
This script allows to calibrate and test an LSTM model on a dataset of 96 watersheds from the INFO-Crue project.
This particular version of the script uses variables from the MELCC gridded dataset as well as the ERA5 reanalysis.
"""

# %% Import packages
import numpy as np
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import shutil
from pathlib import Path
import tensorflow as tf
from utilsS.create_datasets import plot_boxplot_results
from AAA_model_functions import (RunModelAfterTraining, RetrainAfterMainTrain, perform_initial_train, GetNetcdfTags,
                                 SplitDataset, ScaleDataset)
tf.get_logger().setLevel('INFO')

# %% Control variables for all experiments
"""
All the control variables used to train the LSTM models are predefined here.
These are consistent from one experiment to the other, but can be modified as 
needed by the user.
"""

batch_size_val = 128  # batch size used in the training - multiple of 32
epoch_val = 200  # Number of epoch to train the LSTM model
window_size = 365  # Number of time step (days) to use in the LSTM model
train_pct = 60  # Percentage of watersheds used for the training
valid_pct = 20  # Percentage of watersheds used for the validation
RUN_TAG = "Lambda_Model_2"
useParallel = True

doTrain = False
doSimul = True
doRetrain = True

path_to_copy_results_to = "/home/ets/HC3 Dropbox/HC3_Lab_Data/DEH_INFO_CRUE_LSTM_SIMULATIONS/"
full_path_to_saved_project = path_to_copy_results_to + RUN_TAG

if os.path.exists(full_path_to_saved_project):
    sys.exit("Project location already exists. Please modify tag to avoid overwriting.")

if doTrain:
    # Clean folder of old runs before starting
    for filename in Path(".").glob("*.h5"):
        os.remove(filename)
    for filename in Path(".").glob("*.jpg"):
        os.remove(filename)
    for filename in Path(".").glob("*.txt"):
        os.remove(filename)
    for filename in Path(".").glob("*.png"):
        os.remove(filename)

input_data_filename = 'INFO_Crue_ds_ERA5_with_Hydrotel_Full.nc'

exten = '.h5'
rootfn = 'INFO_Crue_ERA5_simulation'
name_of_saved_model = 'AAA_' + rootfn + exten
filename_base = rootfn + '_no_retrain'
filename_base_retrain = rootfn + '_with_retrain'

# Get NetCDF variables tags and infos to use for model training
dynamic_var_tags, Qsim_pos, static_var_tags = GetNetcdfTags()

# Import and scale dataset
watershed_areas, watersheds_ind, arr_dynamic, arr_static, q_stds, train_idx, valid_idx, test_idx, all_idx = (
    ScaleDataset(input_data_filename, dynamic_var_tags, Qsim_pos, static_var_tags, train_pct, valid_pct))

if doTrain:
    # Split into train and valid
    X_train, X_train_static, X_train_q_stds, y_train, X_valid, X_valid_static, X_valid_q_stds, y_valid=(
        SplitDataset(arr_dynamic, arr_static, q_stds, watersheds_ind, train_idx, window_size, valid_idx))

    # Do the main large-scale training
    perform_initial_train(useParallel, window_size, batch_size_val, epoch_val, X_train,
                                                          X_train_static, X_train_q_stds, y_train, X_valid,
                                                          X_valid_static, X_valid_q_stds, y_valid, name_of_saved_model)

if doSimul:
    # Do the model simulation on all watersheds after training
    for w in watersheds_ind:
        RunModelAfterTraining(w, arr_dynamic, arr_static, q_stds, window_size,
                                                           train_idx, batch_size_val, watershed_areas, filename_base,
                                                           name_of_saved_model, valid_idx, test_idx, all_idx)


if doRetrain:
    # Do the model retraining on all watersheds after main train and simulations
    RetrainAfterMainTrain(watersheds_ind, window_size, epoch_val, batch_size_val, arr_dynamic, arr_static, q_stds,
                          train_idx, valid_idx, test_idx, all_idx, watershed_areas, name_of_saved_model,
                          filename_base_retrain)


'''
Load results and make boxplots
'''
plot_boxplot_results(watersheds_ind, filename_base, filename_base_retrain)

"""
Move all results to save folder for safekeeping and future reference
"""
shutil.copytree(src="./", dst=full_path_to_saved_project)







