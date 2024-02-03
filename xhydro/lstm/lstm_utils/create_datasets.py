# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import time

def create_dataset_flexible(filename, dynamic_var_tags, Qsim_pos, static_var_tags):
    
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
    # TODO: We should put a check to ensure the size is different than number
    # of watersheds.
    n_days = ds.Qobs.values.shape[1]
    
    # Perform the analysis for Qobs first.
    arr_Qobs = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_Qobs[:] = np.nan
    arr_Qobs = ds.Qobs.values
    arr_Qobs = arr_Qobs / ds.drainage_area.values[:, np.newaxis] * 86.4
    
    # Prepare the dynamic data array and set Qobs as the first value    
    arr_dynamic = np.empty(shape = [n_watersheds, n_days, len(dynamic_var_tags)+1], dtype=np.float32)
    arr_dynamic[:] = np.nan
    arr_dynamic[:, :, 0] = arr_Qobs
    
    for i in range(len(dynamic_var_tags)):
        
        # Prepare tmp var to store data
        tmp = np.empty([n_watersheds, n_days], dtype=np.float32)
        tmp[:] = np.nan
        
        # Read the data and put in the formatted tmp var
        tmp = ds[dynamic_var_tags[i]].values
        
        # If the variable must be scaled, do it
        if Qsim_pos[i]:
            tmp = tmp / ds.drainage_area.values[:, np.newaxis] * 86.4
        
        # Set the data in the main dataset
        arr_dynamic[:, :, i+1] = tmp
        
    # Prepare the static dataset
    arr_static = np.empty([n_watersheds, len(static_var_tags)], dtype=np.float32)
    arr_static[:] = np.nan
    
    # Loop the 
    for i in range(len(static_var_tags)):
        arr_static[:,i] = ds[static_var_tags[i]].values

    return arr_dynamic, arr_static, arr_Qobs


def create_LSTM_dataset(arr_dynamic, arr_static, q_stds, window_size, 
                        watershed_list):
    """
    """
    block_size = arr_dynamic.shape[1] - window_size
    
    # Preallocate the output arrays
    X = np.empty([
        (arr_dynamic.shape[1] - window_size) * watershed_list.shape[0],
        window_size,
        arr_dynamic.shape[2] - 1
        ]    
    )
    X[:] = np.nan
    
    X_static = np.empty([
        block_size * watershed_list.shape[0],
        arr_static.shape[1]
        ]    
    )    
    X_static[:] = np.nan
    
    X_q_stds = np.empty([
        block_size * watershed_list.shape[0]
        ]    
    )    
    X_q_stds[:] = np.nan
    
    y = np.empty([
        block_size * watershed_list.shape[0]
        ]
    )
    y[:] = np.nan

    counter = 0
    for w in watershed_list:
        print('Currently working on watershed no: ' + str(w))
        X_w, X_w_static, X_w_q_std, y_w = extract_watershed_block(
            arr_w_dynamic=arr_dynamic[w, :, :],
            arr_w_static=arr_static[w, :],
            q_std_w=q_stds[w],
            window_size=window_size
            )
            
        X[counter * block_size : (counter + 1) * block_size, :, :] = X_w
        X_static[counter * block_size : (counter + 1) * block_size, :] = X_w_static
        X_q_stds[counter * block_size : (counter + 1) * block_size] = X_w_q_std
        y[counter * block_size : (counter + 1) * block_size] = y_w
        
        counter += 1
        
    return X, X_static, X_q_stds, y


def create_LSTM_dataset_catchment_vary(arr_dynamic, arr_static, q_stds, window_size, watershed_list, idx, top_percent=None, cleanNans=True):

    ndata = arr_dynamic.shape[2] - 1
    X = np.empty((0,window_size, ndata)) # 7 is with Qsim as predictor and snow
    X_static = np.empty((0, arr_static.shape[1]))
    X_q_stds = np.empty((0))
    y = np.empty((0))

    for w in watershed_list:
        idx_w = idx[w]
        print('Currently working on watershed no: ' + str(w))
        X_w, X_w_static, X_w_q_std, y_w = extract_watershed_block(
            arr_w_dynamic=arr_dynamic[w, idx_w[0]:idx_w[1], :],
            arr_w_static=arr_static[w, :],
            q_std_w=q_stds[w],
            window_size=window_size
        )
        
        # Clean nans
        if cleanNans==True:
            y_w, X_w, X_w_q_std, X_w_static = clean_nans(y_w, X_w, X_w_q_std, X_w_static)


        X = np.vstack([X, X_w])
        X_static = np.vstack([X_static, X_w_static])
        X_q_stds = np.hstack([X_q_stds, X_w_q_std])
        y = np.hstack([y, y_w])

    return X, X_static, X_q_stds, y


def extract_windows_vectorized(array, sub_window_size):

    max_time = array.shape[0]
    
    # expand_dims are used to convert a 1D array to 2D array.
    sub_windows = (
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(max_time - sub_window_size), 0).T
    )

    return array[sub_windows]


def extract_watershed_block(arr_w_dynamic, arr_w_static, q_std_w, window_size):
    """
    This function extracts all series of the desired window length over all
    features for a given watershed. Both dynamic and static variables are 
    extracted.
    """
    # Extract all series of the desired window length for all features
    X_w = extract_windows_vectorized(
        array=arr_w_dynamic,
        sub_window_size=window_size
    )

    X_w_static = np.repeat(arr_w_static.reshape(-1,1), X_w.shape[0], axis=1).T

    X_w_q_std = np.squeeze(np.repeat(q_std_w.reshape(-1,1), X_w.shape[0], axis=1).T)

    # Get the last value of Qobs from each series for the prediction
    y_w = X_w[:, -1, 0]

    # Remove Qobs from the features
    X_w = np.delete(X_w, 0, axis=2)

    return X_w, X_w_static, X_w_q_std, y_w


def clean_nans(y, X, X_q_std, X_static):

    """
    Check for nans in the variable "y" and remove all lines containing those nans in all datasets
    """

    ind_nan = np.isnan(y)
    y = y[~ind_nan]
    X = X[~ind_nan, :, :]
    X_q_stds = X_q_std[~ind_nan]
    X_static = X_static[~ind_nan, :]

    return y, X, X_q_stds, X_static


def plot_boxplot_results(watersheds_ind, filename_base, filename_base_retrain):

    """
    After the refactoring code runs, we can plot the results using this code
    """

    # Load results and make boxplots
    kge_regional = np.empty([watersheds_ind.shape[0], 3])
    kge_retrain = np.empty([watersheds_ind.shape[0], 3])

    for w in watersheds_ind:
        kge_regional[w, 0] = np.loadtxt(filename_base + '_KGE_training_' + str(w) + '.txt')
        kge_regional[w, 1] = np.loadtxt(filename_base + '_KGE_validation_' + str(w) + '.txt')
        kge_regional[w, 2] = np.loadtxt(filename_base + '_KGE_testing_' + str(w) + '.txt')
        kge_retrain[w, 0] = np.loadtxt(filename_base_retrain + '_KGE_training_' + str(w) + '.txt')
        kge_retrain[w, 1] = np.loadtxt(filename_base_retrain + '_KGE_validation_' + str(w) + '.txt')
        kge_retrain[w, 2] = np.loadtxt(filename_base_retrain + '_KGE_testing_' + str(w) + '.txt')

    # Plot results
    fig1, ax1 = plt.subplots(1, 1)
    fig1.subplots_adjust(left=0.1,
                         bottom=0.1,
                         right=0.9,
                         top=0.9,
                         wspace=0.4,
                         hspace=0.4)

    ax1.set_title('LSTM model results')
    ax1.boxplot(kge_regional)
    ax1.set_ylim([0.0, 1.0])
    ax1.set_ylabel('KGE')
    ax1.set_xticklabels(['Train', 'Valid', 'Test'])
    ax1.grid(axis='y')


    plt.show(block=False)
    time.sleep(2)
    plt.savefig(fname="AAA_Model_KGE_results.png")
    time.sleep(2)