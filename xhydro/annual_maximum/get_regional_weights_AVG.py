import numpy as np
import os
import scipy.io

# Set paths
work_folder = os.getcwd()
data_folder = 'C:/Users/AR21010/Dropbox/xhydro/INFO-Crue/0_data/2_extracted_data'
results_folder = 'C:/Users/AR21010/Dropbox/xhydro/INFO-Crue/0_data/3_results'
save_folder = 'C:/Users/AR21010/Dropbox/xhydro/INFO-Crue/0_data/3_results'

# Type of simulations to use
simulation_type = 'HYDROTEL_6'  # DEH 6 HYDROTEL simulations
obs_stations_to_select = 'selection'

# Load data
mat_data = scipy.io.loadmat(
    os.path.join(data_folder, f'INFO_Crue_obs_stations_{simulation_type}_data_{obs_stations_to_select}.mat'))
obs_stations_data = mat_data['obs_stations_data']
time_sim = mat_data['time_sim']
drainage_area_obs = mat_data['drainage_area_obs']

# There should be 259 watersheds (all) or 96 watersheds (selection)
n_watersheds = obs_stations_data.shape[0]
# Number of simulations in total used in the multi-model averaging approach
n_models = obs_stations_data.shape[1]

# Pre-process the data
all_data = []
for i in range(n_watersheds):
    dates_sim = time_sim.flatten()

    # Get all observed and simulated streamflow
    Qobs = obs_stations_data[i, 0]['Qobs'][:len(dates_sim)]
    Qsim = obs_stations_data[i, 0]['Qsim'][:len(dates_sim), :]

    # Skip the first year because of the model warm-up period
    ind_first_year = dates_sim[:, 0] == dates_sim[0, 0]
    dates_sim[ind_first_year, :] = []
    Qobs[ind_first_year] = []
    Qsim[ind_first_year, :] = []

    # Remove the last year if it is not complete
    if dates_sim[-1, 1] != 12 or dates_sim[-1, 2] != 31:
        ind_last_year = dates_sim[:, 0] == dates_sim[-1, 0]
        dates_sim[ind_last_year, :] = []
        Qobs[ind_last_year] = []
        Qsim[ind_last_year, :] = []

    # Remove all February 29th to compute Qx1day
    ind_29feb = np.where((dates_sim[:, 1] == 2) & (dates_sim[:, 2] == 29))[0]
    dates_sim = np.delete(dates_sim, ind_29feb, axis=0)
    Qobs = np.delete(Qobs, ind_29feb)
    Qsim = np.delete(Qsim, ind_29feb, axis=0)

    # Get the pre-processed data for the current watershed
    all_data.append({'Qobs': Qobs, 'Qsim': Qsim})

# Combine all Qobs and Qsim together
Qobs = np.concatenate([all_data[i]['Qobs'] / drainage_area_obs[i] for i in range(n_watersheds)])
Qsim = np.concatenate([all_data[i]['Qsim'] / drainage_area_obs[i] for i in range(n_watersheds)])

# Apply random white noise around 1 (needed because some simulations have the same values which can generate problems)
a = np.random.randn(Qsim.shape[0], Qsim.shape[1]) / 100
a = a + 1
Qsim = Qsim * a

# Remove values where no observed streamflows are available
ind_nan = np.isnan(Qobs)
Qobs = Qobs[~ind_nan, :]
Qsim = Qsim[~ind_nan, :]

# Remove values where no simulated streamflows are available
row, _ = np.where(np.isnan(Qsim))
Qobs = np.delete(Qobs, row, axis=0)
Qsim = np.delete(Qsim, row, axis=0)

# Simple Average (AVG)
# A simple average where all weights are equal
print('Currently calculating weights for the AVG...')
regional_weights = np.ones((1, n_models)) / n_models  # AVG weights

# Compute the metrics for the multi-model averaged series
Qobs = np.empty((0,))
Qsim = np.empty((0,))

# Get the results for each watershed (no validation)
for i in range(n_watersheds):
    # Get observed streamflow for the current watershed
    Qobs = all_data[i]['Qobs']

    # Compute the annual maximum observed streamflow series
    Qx1day_obs = np.max(Qobs.reshape((365, -1)), axis=0)

    # Remove values where no observed streamflows are available
    ind_nan = np.isnan(Qobs)
    Qobs = Qobs[~ind_nan]

    # Sort all Qobs values to get percentiles
    Qobs_sort = np.sort(Qobs, axis=None, kind='mergesort')

    # Compute the metrics if there are observed streamflows for the period
    if Qobs.size > 0:
        # Get the AVG averaged streamflow
        Qsim = all_data[i]['Qsim']
        Qsim = np.sum(Qsim * regional_weights, axis=1)

        # Compute the annual maximum AVG streamflow series
        Qx1day_sim = np.max(Qsim.reshape((365, -1)), axis=0)

        # Remove values where no observed streamflows are available
        Qsim = Qsim[~ind_nan]

        # Sort all Qobs values to get percentiles
        Qsim_sort = np.sort(Qsim, axis=None, kind='mergesort')

        # Metrics
        KGE = obj_fun_KGE(Qobs, Qsim)
        NSE = obj_fun_NSE(Qobs, Qsim)
        NRMSE_90_PCTL = obj_fun_NRMSE(Qobs_sort[int(len(Qobs_sort) * 0.90):], Qsim_sort[int(len(Qsim_sort) * 0.90):])
        NRMSE_95_PCTL = obj_fun_NRMSE(Qobs_sort[int(len(Qobs_sort) * 0.95):], Qsim_sort[int(len(Qsim_sort) * 0.95):])
        NRMSE_99_PCTL = obj_fun_NRMSE(Qobs_sort[int(len(Qobs_sort) * 0.99):], Qsim_sort[int(len(Qsim_sort) * 0.99):])
        NRMSE_Qx1day = obj_fun_NRMSE(Qx1day_obs, Qx1day_sim)

    else:  # Otherwise, put NaNs for all metrics
        KGE, NSE, NRMSE_90_PCTL, NRMSE_95_PCTL, NRMSE_99_PCTL, NRMSE_Qx1day = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Save results
    regional_weights_results = {
        'KGE': KGE,
        'NSE': NSE,
        'NRMSE_90_PCTL': NRMSE_90_PCTL,
        'NRMSE_95_PCTL': NRMSE_95_PCTL,
        'NRMSE_99_PCTL': NRMSE_99_PCTL,
        'NRMSE_Qx1day': NRMSE_Qx1day
    }

    # Save the AVG results
    save_path = os.path.join(save_folder, f'AVG_{simulation_type}_{obs_stations_to_select}_weights.mat')
    scipy.io.savemat(save_path,
                     {'regional_weights': regional_weights, 'regional_weights_results': regional_weights_results},
                     format='5', long_field_names=True)
