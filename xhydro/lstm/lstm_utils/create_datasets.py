# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import time



def create_baseline_dataset(ds):
    """
    """
    
    # Number of watersheds in the dataset
    n_watersheds = ds.watershed.shape[0]  
    
    # Number of days for each dynamic variables
    n_days = ds.Qobs.values.shape[1]
    
    # Pre-allocate the dataframes
    arr_Qobs   = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_pr     = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_tasmax = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_tasmin = np.empty([n_watersheds, n_days], dtype=np.float32)
    
    arr_Qobs[:]   = np.nan
    arr_pr[:]     = np.nan
    arr_tasmax[:] = np.nan
    arr_tasmin[:] = np.nan
    
    # Get the data from the xarray dataset within the dataframes
    arr_Qobs   = ds.Qobs.values
    
    # Convert m^3/s to mm/d
    arr_Qobs = arr_Qobs / ds.drainage_area.values[:,np.newaxis] * 86.4
    
    arr_pr     = ds.pr_MELCC.values
    arr_tasmax = ds.tasmax_MELCC.values
    arr_tasmin = ds.tasmin_MELCC.values
    
    # Stack all the dynamic data together
    arr_dynamic = np.dstack((
        arr_Qobs,
        arr_pr,
        arr_tasmax,
        arr_tasmin
        ))
    
    # Extract all 14 watershed descriptors
    arr_static = np.empty([n_watersheds, 14], dtype=np.float32)
    arr_static[:] = np.nan
    
    arr_static[:, 0]  = ds.drainage_area.values
    arr_static[:, 1]  = ds.elevation.values
    arr_static[:, 2]  = ds.slope.values
    arr_static[:, 3]  = ds.water.values
    arr_static[:, 4]  = ds.deciduous_forest.values
    arr_static[:, 5]  = ds.agricultural_pasture.values
    arr_static[:, 6]  = ds.coniferous_forest.values
    arr_static[:, 7]  = ds.impervious.values
    arr_static[:, 8]  = ds.bog.values
    arr_static[:, 9]  = ds.wet_land.values
    arr_static[:, 10] = ds.loamy_sand.values
    arr_static[:, 11] = ds.loam.values
    arr_static[:, 12] = ds.silt.values
    arr_static[:, 13] = ds.silty_clay_loam.values
    
    return arr_dynamic, arr_static, arr_Qobs


def create_ERA5_dataset(ds):
    
    # Number of watersheds in the dataset
    n_watersheds = ds.watershed.shape[0]  
    
    # Number of days for each dynamic variables
    n_days = ds.Qobs.values.shape[1]
    
    # Pre-allocate the dataframes
    arr_Qobs   = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_pr     = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_tasmax = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_tasmin = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_d2m    = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ssr    = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_Qsim   = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_swe     = np.empty([n_watersheds, n_days], dtype=np.float32)

    arr_Qobs[:]   = np.nan
    arr_pr[:]     = np.nan
    arr_tasmax[:] = np.nan
    arr_tasmin[:] = np.nan
    arr_d2m[:]    = np.nan
    arr_ssr[:]    = np.nan
    arr_Qsim[:]   = np.nan
    arr_swe[:]   = np.nan

    # Get the data from the xarray dataset within the dataframes
    arr_Qobs   = ds.Qobs.values
    arr_Qsim   = ds.Qsim.values

    # Convert m^3/s to mm/d
    arr_Qobs = arr_Qobs / ds.drainage_area.values[:, np.newaxis] * 86.4
    arr_Qsim = arr_Qsim / ds.drainage_area.values[:, np.newaxis] * 86.4

    arr_pr     = ds.pr_MELCC.values
    arr_tasmax = ds.tasmax_MELCC.values
    arr_tasmin = ds.tasmin_MELCC.values
    arr_d2m    = ds.d2m_ERA5.values
    arr_ssr    = ds.ssr_ERA5.values
    arr_swe = ds.swe.values


    # Stack all the dynamic data together
    arr_dynamic = np.dstack((
        arr_Qobs,
        arr_pr,
        arr_tasmax,
        arr_tasmin,
        arr_d2m,
        arr_ssr,
        arr_Qsim,
        arr_swe
        ))
    
    # Extract all 14 watershed descriptors
    arr_static = np.empty([n_watersheds, 14], dtype=np.float32)
    arr_static[:] = np.nan
    
    arr_static[:, 0]  = ds.drainage_area.values
    arr_static[:, 1]  = ds.elevation.values
    arr_static[:, 2]  = ds.slope.values
    arr_static[:, 3]  = ds.water.values
    arr_static[:, 4]  = ds.deciduous_forest.values
    arr_static[:, 5]  = ds.agricultural_pasture.values
    arr_static[:, 6]  = ds.coniferous_forest.values
    arr_static[:, 7]  = ds.impervious.values
    arr_static[:, 8]  = ds.bog.values
    arr_static[:, 9]  = ds.wet_land.values
    arr_static[:, 10] = ds.loamy_sand.values
    arr_static[:, 11] = ds.loam.values
    arr_static[:, 12] = ds.silt.values
    arr_static[:, 13] = ds.silty_clay_loam.values
    
    return arr_dynamic, arr_static, arr_Qobs


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



def create_ERA5_dataset_allvars(ds):
    # Number of watersheds in the dataset
    n_watersheds = ds.watershed.shape[0]

    # Number of days for each dynamic variables
    n_days = ds.Qobs.values.shape[1]

    # Pre-allocate the dataframes
    arr_Qobs = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_pr = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_tasmax = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_tasmin = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_Qsim = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5swe = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5smlt = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5snowfall = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5tmin = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5tmax = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5dewpoint = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5precip = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5u10 = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5v10 = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5evap = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5ssr = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5surfpressure = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5rainfall = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_ERA5windspeed = np.empty([n_watersheds, n_days], dtype=np.float32)
    
    arr_Qobs[:] = np.nan
    arr_pr[:] = np.nan
    arr_tasmax[:] = np.nan
    arr_tasmin[:] = np.nan
    arr_Qsim[:] = np.nan
    arr_ERA5swe[:] = np.nan
    arr_ERA5smlt[:] = np.nan
    arr_ERA5snowfall[:] = np.nan
    arr_ERA5tmin[:] = np.nan
    arr_ERA5tmax[:] = np.nan
    arr_ERA5dewpoint[:] = np.nan
    arr_ERA5precip[:] = np.nan
    arr_ERA5u10[:] = np.nan
    arr_ERA5v10[:] = np.nan
    arr_ERA5evap[:] = np.nan
    arr_ERA5ssr[:] = np.nan
    arr_ERA5surfpressure[:] = np.nan
    arr_ERA5rainfall[:] = np.nan
    arr_ERA5windspeed[:] = np.nan
    
    # Get the data from the xarray dataset within the dataframes
    arr_Qobs = ds.Qobs.values
    arr_Qsim = ds.Qsim.values

    # Convert m^3/s to mm/d
    arr_Qobs = arr_Qobs / ds.drainage_area.values[:, np.newaxis] * 86.4
    arr_Qsim = arr_Qsim / ds.drainage_area.values[:, np.newaxis] * 86.4


    arr_pr = ds.pr_MELCC.values
    arr_tasmax = ds.tasmax_MELCC.values
    arr_tasmin = ds.tasmin_MELCC.values
    arr_ERA5swe = ds.sd.values
    arr_ERA5smlt = ds.smlt.values
    arr_ERA5snowfall = ds.sf.values
    arr_ERA5tmin = ds.tmin.values
    arr_ERA5tmax = ds.tmax.values
    arr_ERA5dewpoint = ds.d2m.values
    arr_ERA5precip = ds.tp.values
    arr_ERA5u10 = ds.u10.values
    arr_ERA5v10 = ds.v10.values
    arr_ERA5evap = ds.e.values
    arr_ERA5ssr = ds.ssr.values
    arr_ERA5surfpressure = ds.sp.values
    arr_ERA5rainfall = ds.rf.values
    arr_ERA5windspeed = ds.rf.windspeed



    # Stack all the dynamic data together
    arr_dynamic = np.dstack((
        arr_Qobs,
        arr_pr,
        arr_tasmax,
        arr_tasmin,
        arr_Qsim,
        arr_ERA5swe,
        arr_ERA5smlt,
        arr_ERA5snowfall,
        arr_ERA5tmin,
        arr_ERA5tmax,
        arr_ERA5dewpoint,
        arr_ERA5precip,
        arr_ERA5u10,
        arr_ERA5v10,
        arr_ERA5evap,
        arr_ERA5ssr,
        arr_ERA5surfpressure,
        arr_ERA5rainfall,
        arr_ERA5windspeed,
    ))

    # Extract all 14 watershed descriptors
    arr_static = np.empty([n_watersheds, 16], dtype=np.float32)
    arr_static[:] = np.nan

    arr_static[:, 0] = ds.drainage_area.values
    arr_static[:, 1] = ds.elevation.values
    arr_static[:, 2] = ds.slope.values
    arr_static[:, 3] = ds.water.values
    arr_static[:, 4] = ds.deciduous_forest.values
    arr_static[:, 5] = ds.agricultural_pasture.values
    arr_static[:, 6] = ds.coniferous_forest.values
    arr_static[:, 7] = ds.impervious.values
    arr_static[:, 8] = ds.bog.values
    arr_static[:, 9] = ds.wet_land.values
    arr_static[:, 10] = ds.loamy_sand.values
    arr_static[:, 11] = ds.loam.values
    arr_static[:, 12] = ds.silt.values
    arr_static[:, 13] = ds.silty_clay_loam.values
    arr_static[:, 14] = np.mean(arr_ERA5precip,axis=1)
    arr_static[:, 15] = np.mean(arr_ERA5swe, axis=1)

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


def create_LSTM_dataset_postdated(arr_dynamic, arr_static, q_stds, window_size,
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
        X_w, X_w_static, X_w_q_std, y_w = extract_watershed_block_postdated(
            arr_w_dynamic=arr_dynamic[w, :, :],
            arr_w_static=arr_static[w, :],
            q_std_w=q_stds[w],
            window_size=window_size
        )

        X[counter * block_size: (counter + 1) * block_size, :, :] = X_w
        X_static[counter * block_size: (counter + 1) * block_size, :] = X_w_static
        X_q_stds[counter * block_size: (counter + 1) * block_size] = X_w_q_std
        y[counter * block_size: (counter + 1) * block_size] = y_w

        counter += 1

    return X, X_static, X_q_stds, y


def create_LSTM_dataset_catchment_vary(arr_dynamic, arr_static, q_stds, window_size, watershed_list, idx, top_percent=None, cleanNans=True):

    """
    Removing this part for optimization. Instead, using the brute-force approach that ensures no errors (one block at a
    time and append at each step).
    """

    '''
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
    '''
    counter = 0
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

        if top_percent is not None:
            y_w, X_w, X_w_q_std, X_w_static = get_highest(y_w, X_w, X_w_q_std, X_w_static, top_percent=top_percent)
        """
        X[counter * block_size: (counter + 1) * block_size, :, :] = X_w
        X_static[counter * block_size: (counter + 1) * block_size, :] = X_w_static
        X_q_stds[counter * block_size: (counter + 1) * block_size] = X_w_q_std
        y[counter * block_size: (counter + 1) * block_size] = y_w
        counter += 1
        """

        X = np.vstack([X, X_w])
        X_static = np.vstack([X_static, X_w_static])
        X_q_stds = np.hstack([X_q_stds, X_w_q_std])
        y = np.hstack([y, y_w])

    return X, X_static, X_q_stds, y

def create_LSTM_dataset_catchment_vary_postdated(arr_dynamic, arr_static, q_stds, window_size, watershed_list, idx, top_percent=None):

    counter = 0
    X = np.empty((0,window_size, 5))
    X_static = np.empty((0, arr_static.shape[1]))
    X_q_stds = np.empty((0))
    y = np.empty((0))

    for w in watershed_list:
        idx_w = idx[w]
        print('Currently working on watershed no: ' + str(w))
        X_w, X_w_static, X_w_q_std, y_w = extract_watershed_block_postdated(
            arr_w_dynamic=arr_dynamic[w, idx_w[0]:idx_w[1], :],
            arr_w_static=arr_static[w, :],
            q_std_w=q_stds[w],
            window_size=window_size
        )
        # Clean nans
        y_w, X_w, X_w_q_std, X_w_static = clean_nans(y_w, X_w, X_w_q_std, X_w_static)

        if top_percent is not None:
            y_w, X_w, X_w_q_std, X_w_static = get_highest(y_w, X_w, X_w_q_std, X_w_static, top_percent=top_percent)


        X = np.vstack([X, X_w])
        X_static = np.vstack([X_static, X_w_static])
        X_q_stds = np.hstack([X_q_stds, X_w_q_std])
        y = np.hstack([y, y_w])

    return X, X_static, X_q_stds, y


def extract_windows_vectorized(array, sub_window_size):
    """
    Vectorized sliding window extractor.
    This method is more efficient than the naive sliding window extractor.

    For more info, see:
    https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    :param array: The array from which to extract windows
    :param sub_window_size: The size of the sliding window
    :return: an array of all the sub windows
    """
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


def extract_watershed_block_postdated(arr_w_dynamic, arr_w_static, q_std_w, window_size):
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

    X_w_static = np.repeat(arr_w_static.reshape(-1, 1), X_w.shape[0], axis=1).T

    X_w_q_std = np.squeeze(np.repeat(q_std_w.reshape(-1, 1), X_w.shape[0], axis=1).T)

    # Get the last value of Qobs from each series for the prediction
    y_w = X_w[:, -3, 0]

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
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.subplots_adjust(left=0.1,
                         bottom=0.1,
                         right=0.9,
                         top=0.9,
                         wspace=0.4,
                         hspace=0.4)

    ax1.set_title('Regional model')
    ax1.boxplot(kge_regional)
    ax1.set_ylim([0.5, 1.0])
    ax1.set_ylabel('KGE')
    ax1.set_xticklabels(['Train', 'Valid', 'Test'])
    ax1.grid(axis='y')

    ax2.set_title('Retrain model')
    ax2.set_ylabel('KGE')
    ax2.boxplot(kge_retrain)
    ax2.set_ylim([0.5, 1.0])
    ax2.set_xticklabels(['Train', 'Valid', 'Test'])
    ax2.grid(axis='y')

    plt.show(block=False)
    time.sleep(2)
    plt.savefig(fname="AAA_Model_KGE_results.png")
    time.sleep(2)
    

def get_highest(y, X, X_q_std, X_static, top_percent):

    """
    Delete points that are lower than a given threshold
    """

    ind_del = np.where(y < np.percentile(y, top_percent))
    y = np.delete(y, ind_del)
    X = np.delete(X, ind_del, axis=0)
    X_q_std = np.delete(X_q_std, ind_del)
    X_static = np.delete(X_static, ind_del, axis=0)

    return y, X, X_q_std, X_static


def plot_boxplot_results_peaks():
    """
    After the refactoring code runs, we can plot the results using this code for peak flow per year
    """
    import datetime as dt
    # Load results and make boxplots
    peaks_error_train = np.empty([0])
    peaks_error_val = np.empty([0])
    peaks_error_test = np.empty([0])
    peaks_error_train_retrain = np.empty([0])
    peaks_error_val_retrain = np.empty([0])
    peaks_error_test_retrain = np.empty([0])
    peaks_error_train_retrain_peaks = np.empty([0])
    peaks_error_val_retrain_peaks = np.empty([0])
    peaks_error_test_retrain_peaks = np.empty([0])

    filename_base = "Runs/INFO_Crue_regional_model_512_512_no_retrain"
    filename_base_retrain = 'Runs/INFO_Crue_regional_model_512_512_with_retrain'

    # Not necessary but left here to make code easier to adapt. Will be same results as filename_base_retrain:
    filename_base_retrain_peaks = 'Runs/INFO_Crue_regional_model_512_512_with_retrain'
#    filename_base_retrain = 'INFO_Crue_regional_model_512_512_with_retrain_on_peaks_from_full_regional'
    """
    filename_base_retrain = 'INFO_Crue_regional_model_512_512_with_retrain_on_peaks_from_full_regional.h5'
    filename_base_retrain_peaks = 'INFO_Crue_regional_model_512_512_with_retrain_peaks'
    filename_base = 'INFO_Crue_regional_model_512_512_no_retrain_on_peaks_full_regional'
    """
    watersheds_ind = np.arange(0, 88)
    # Get watershed size
    import xarray as xr
    input_data_filename = 'INFO_Crue_ds_ERA5_with_Hydrotel_with_snow.nc'
    ds = xr.open_dataset(input_data_filename)
    arr_dynamic, arr_static, arr_Qobs = create_ERA5_dataset(ds)
    # Find which catchments have less than 10 years of data (20% of 10 = 2 years for valid/testing) and delete them
    for i in reversed(range(0, arr_Qobs.shape[0])):
        if np.count_nonzero(~np.isnan(arr_Qobs[i, :])) < 10 * 365:
            arr_dynamic = np.delete(arr_dynamic, i, 0)
            arr_static = np.delete(arr_static, i, 0)
            arr_Qobs = np.delete(arr_Qobs, i, 0)

    watershed_areas = arr_static[:, 0]


    for w in watersheds_ind:
        flow_regional_train = np.loadtxt(filename_base + '_FLOW_training_' + str(w) + '.txt', delimiter=",")
        flow_regional_val = np.loadtxt(filename_base + '_FLOW_validation_' + str(w) + '.txt', delimiter=',')
        flow_regional_test = np.loadtxt(filename_base + '_FLOW_testing_' + str(w) + '.txt', delimiter=',')
        flow_specific_train = np.loadtxt(filename_base_retrain + '_FLOW_training_' + str(w) + '.txt', delimiter=',')
        flow_specific_val = np.loadtxt(filename_base_retrain + '_FLOW_validation_' + str(w) + '.txt', delimiter=',')
        flow_specific_test = np.loadtxt(filename_base_retrain + '_FLOW_testing_' + str(w) + '.txt', delimiter=',')
        flow_specific_peaks_train = np.loadtxt(filename_base_retrain_peaks + '_FLOW_training_' + str(w) + '.txt', delimiter=',')
        flow_specific_peaks_val = np.loadtxt(filename_base_retrain_peaks + '_FLOW_validation_' + str(w) + '.txt', delimiter=',')
        flow_specific_peaks_test = np.loadtxt(filename_base_retrain_peaks + '_FLOW_testing_' + str(w) + '.txt', delimiter=',')

        days = pd.date_range("19810101", periods=flow_regional_train.shape[0])

        df_train = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_regional_train.shape[0]), 'Obs': flow_regional_train[:, 0], 'Sim': flow_regional_train[:, 1]})
        df_val = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_regional_val.shape[0]), 'Obs': flow_regional_val[:, 0], 'Sim': flow_regional_val[:, 1]})
        df_test = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_regional_test.shape[0]), 'Obs': flow_regional_test[:, 0], 'Sim': flow_regional_test[:, 1]})
        df_train_retrain = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_specific_train.shape[0]), 'Obs': flow_specific_train[:, 0], 'Sim': flow_specific_train[:, 1]})
        df_val_retrain = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_specific_val.shape[0]), 'Obs': flow_specific_val[:, 0], 'Sim': flow_specific_val[:, 1]})
        df_test_retrain = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_specific_test.shape[0]), 'Obs': flow_specific_test[:, 0], 'Sim': flow_specific_test[:, 1]})
        df_train_retrain_peaks = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_specific_peaks_train.shape[0]), 'Obs': flow_specific_peaks_train[:, 0],'Sim': flow_specific_peaks_train[:, 1]})
        df_val_retrain_peaks = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_specific_peaks_val.shape[0]), 'Obs': flow_specific_peaks_val[:, 0],'Sim': flow_specific_peaks_val[:, 1]})
        df_test_retrain_peaks = pd.DataFrame({'date': pd.date_range("19810101", periods=flow_specific_peaks_test.shape[0]), 'Obs': flow_specific_peaks_test[:, 0],'Sim': flow_specific_peaks_test[:, 1]})

        deltas_train = df_train.groupby(df_train.date.dt.year)['Obs'].max() - df_train.groupby(df_train.date.dt.year)['Sim'].max()
        deltas_val = df_val.groupby(df_val.date.dt.year)['Obs'].max() - df_val.groupby(df_val.date.dt.year)['Sim'].max()
        deltas_test = df_test.groupby(df_test.date.dt.year)['Obs'].max() - df_test.groupby(df_test.date.dt.year)['Sim'].max()
        deltas_train_retrain = df_train_retrain.groupby(df_train_retrain.date.dt.year)['Obs'].max() - df_train_retrain.groupby(df_train_retrain.date.dt.year)['Sim'].max()
        deltas_val_retrain = df_val_retrain.groupby(df_val_retrain.date.dt.year)['Obs'].max() - df_val_retrain.groupby(df_val_retrain.date.dt.year)['Sim'].max()
        deltas_test_retrain = df_test_retrain.groupby(df_test_retrain.date.dt.year)['Obs'].max() - df_test_retrain.groupby(df_test_retrain.date.dt.year)['Sim'].max()
        deltas_train_retrain_peaks = df_train_retrain.groupby(df_train_retrain_peaks.date.dt.year)['Obs'].max() - df_train_retrain_peaks.groupby(df_train_retrain_peaks.date.dt.year)['Sim'].max()
        deltas_val_retrain_peaks = df_val_retrain.groupby(df_val_retrain_peaks.date.dt.year)['Obs'].max() - df_val_retrain_peaks.groupby(df_val_retrain_peaks.date.dt.year)['Sim'].max()
        deltas_test_retrain_peaks = df_test_retrain.groupby(df_test_retrain_peaks.date.dt.year)['Obs'].max() - df_test_retrain_peaks.groupby(df_test_retrain_peaks.date.dt.year)['Sim'].max()

        deltas_train = deltas_train.to_numpy()
        deltas_val = deltas_val.to_numpy()
        deltas_test = deltas_test.to_numpy()
        deltas_train_retrain = deltas_train_retrain.to_numpy()
        deltas_val_retrain = deltas_val_retrain.to_numpy()
        deltas_test_retrain = deltas_test_retrain.to_numpy()
        deltas_train_retrain_peaks = deltas_train_retrain_peaks.to_numpy()
        deltas_val_retrain_peaks = deltas_val_retrain_peaks.to_numpy()
        deltas_test_retrain_peaks = deltas_test_retrain_peaks.to_numpy()

        deltas_train = deltas_train / np.nanmean(arr_Qobs[w,:]) #watershed_areas[w]
        deltas_val = deltas_val / np.nanmean(arr_Qobs[w,:]) #watershed_areas[w]
        deltas_test = deltas_test / np.nanmean(arr_Qobs[w,:]) #watershed_areas[w]
        deltas_train_retrain = deltas_train_retrain / np.nanmean(arr_Qobs[w,:]) #watershed_areas[w]
        deltas_val_retrain = deltas_val_retrain / np.nanmean(arr_Qobs[w,:]) #watershed_areas[w]
        deltas_test_retrain = deltas_test_retrain / np.nanmean(arr_Qobs[w,:]) #watershed_areas[w]
        deltas_train_retrain_peaks = deltas_train_retrain_peaks / np.nanmean(arr_Qobs[w, :])  # watershed_areas[w]
        deltas_val_retrain_peaks = deltas_val_retrain_peaks / np.nanmean(arr_Qobs[w, :])  # watershed_areas[w]
        deltas_test_retrain_peaks = deltas_test_retrain_peaks / np.nanmean(arr_Qobs[w, :])  # watershed_areas[w]

        peaks_error_train = np.hstack([peaks_error_train, deltas_train])
        peaks_error_val = np.hstack([peaks_error_val, deltas_val])
        peaks_error_test = np.hstack([peaks_error_test, deltas_test])
        peaks_error_train_retrain = np.hstack([peaks_error_train_retrain, deltas_train_retrain])
        peaks_error_val_retrain = np.hstack([peaks_error_val_retrain, deltas_val_retrain])
        peaks_error_test_retrain = np.hstack([peaks_error_test_retrain, deltas_test_retrain])
        peaks_error_train_retrain_peaks = np.hstack([peaks_error_train_retrain_peaks, deltas_train_retrain_peaks])
        peaks_error_val_retrain_peaks = np.hstack([peaks_error_val_retrain_peaks, deltas_val_retrain_peaks])
        peaks_error_test_retrain_peaks = np.hstack([peaks_error_test_retrain_peaks, deltas_test_retrain_peaks])

    # Plot results
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig1.subplots_adjust(left=0.1,
                         bottom=0.1,
                         right=0.9,
                         top=0.9,
                         wspace=0.4,
                         hspace=0.4)

    ax1.set_title('Training')
    ttt = np.vstack([peaks_error_train, peaks_error_train_retrain, peaks_error_train_retrain_peaks]).T
    mask = ~np.isnan(ttt)
    ttt = [d[m] for d, m in zip(ttt.T, mask.T)]
    ax1.boxplot(ttt)
    ax1.set_ylabel('Specific Flow error (m³/s)/(m³/s)')
    ax1.set_ylim([-700, 700])
    ax1.set_xticklabels(['Regional', 'Specific', 'Specific with peaks-regional'])
    ax1.grid(axis='y')

    ax2.set_title('Validation')
    ttt = np.vstack([peaks_error_val, peaks_error_val_retrain, peaks_error_val_retrain_peaks]).T
    mask = ~np.isnan(ttt)
    ttt = [d[m] for d, m in zip(ttt.T, mask.T)]
    ax2.boxplot(ttt)
    ax2.set_ylabel('Specific Flow error (m³/s)/(m³/s)')
    ax2.set_ylim([-700, 700])
    ax2.set_xticklabels(['Regional', 'Specific', 'Specific with peaks-regional'])
    ax2.grid(axis='y')

    ax3.set_title('Testing')
    ttt = np.vstack([peaks_error_test, peaks_error_test_retrain, peaks_error_test_retrain_peaks]).T
    mask = ~np.isnan(ttt)
    ttt = [d[m] for d, m in zip(ttt.T, mask.T)]
    ax3.boxplot(ttt)
    ax3.set_ylabel('Specific Flow error (m³/s)/(m³/s)')
    ax3.set_ylim([-700, 700])
    ax3.set_xticklabels(['Regional', 'Specific', 'Specific with peaks-regional'])
    ax3.grid(axis='y')

    plt.show(block=False)


def plot_boxplot_weights():

    """
    After the weights are generated, we can use this to plot results
    """

    weights_regional = np.loadtxt('Results_regional_model_basic_importance.txt', delimiter=',')
    weights_specific = np.loadtxt('Results_specific_model_peak_train_after_full_importance.txt', delimiter=',')
    weights_specific = np.loadtxt('Results_specific_model_peak_train_after_full_importance_scale_1p0.txt',delimiter=',')

    # Plot results
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.subplots_adjust(left=0.1,
                         bottom=0.1,
                         right=0.9,
                         top=0.9,
                         wspace=0.4,
                         hspace=0.4)

    ax1.set_title('Regional model')
    ax1.boxplot(weights_regional)
    ax1.set_ylim([0.0, 5.0])
    ax1.set_ylabel('Importance relative des entrées')
    ax1.set_xticklabels(['Précip','Tmin','Tmax','Hum.Rel.','Radiation'])
    ax1.grid(axis='y')

    ax2.set_title('Retrained/specific model')
    ax2.set_ylabel('Importance relative des entrées')
    ax2.boxplot(weights_specific)
    ax2.set_ylim([0.0, 5.0])
    ax2.set_xticklabels(['Précip','Tmin','Tmax','Hum.Rel.','Radiation'])
    ax2.grid(axis='y')

    plt.show(block=True)

