import os
import numpy as np
import scipy.io

# Set working directories
work_folder = os.getcwd()
data_folder = 'C:\\Users\\AR21010\\Dropbox\\xhydro\\INFO-Crue\\0_data\\2_extracted_data'
results_folder = 'C:\\Users\\AR21010\\Dropbox\\xhydro\\INFO-Crue\\0_data\\3_results'
save_folder = 'C:\\Users\\AR21010\\Dropbox\\xhydro\\INFO-Crue\\0_data\\3_results'

# Type of simulations to use
simulation_type = 'HYDROTEL_150'  # ETS 144 + 6 DEH HYDROTEL simulations

# Take all observed watersheds or only a selection of them (made by DEH)
obs_stations_to_select = 'selection'

# Determine the chosen period on which to compute the weights
chosen_period = 'month_by_month'

os.chdir(data_folder)
mat_data = scipy.io.loadmat(f'INFO_Crue_obs_stations_{simulation_type}_data_{obs_stations_to_select}.mat')
os.chdir(work_folder)

obs_stations_data = mat_data['obs_stations_data'][0]
time_sim = mat_data['time_sim'][0]
drainage_area_obs = mat_data['drainage_area_obs'][0]

# There should be 259 watersheds (all) or 96 watersheds (selection)
n_watersheds = len(obs_stations_data)
# Number of simulations in total used in the multi-model averaging approach
n_models = obs_stations_data[0]['Qsim'].shape[1]

# Pre-process the data data
all_data = []
for i in range(n_watersheds):
    dates_sim = time_sim  # Start from the full vector date

    # Get all observed and simulated streamflow
    Qobs = obs_stations_data[i]['Qobs'][:len(dates_sim)]
    Qsim = obs_stations_data[i]['Qsim'][:len(dates_sim), :]

    # Skip the first year because for the model warm-up period
    ind_first_year = dates_sim[:, 0] == dates_sim[0, 0]
    dates_sim = np.delete(dates_sim, np.where(ind_first_year), axis=0)
    Qobs = np.delete(Qobs, np.where(ind_first_year))
    Qsim = np.delete(Qsim, np.where(ind_first_year), axis=0)

    # Remove the last year if it is not complete
    if dates_sim[-1, 1] != 12 or dates_sim[-1, 2] != 31:
        ind_last_year = dates_sim[:, 0] == dates_sim[-1, 0]
        dates_sim = np.delete(dates_sim, np.where(ind_last_year), axis=0)
        Qobs = np.delete(Qobs, np.where(ind_last_year))
        Qsim = np.delete(Qsim, np.where(ind_last_year), axis=0)

    # Remove all February 29th to compute Qx1day
    ind_29feb = np.where((dates_sim[:, 1] == 2) & (dates_sim[:, 2] == 29))[0]
    dates_sim = np.delete(dates_sim, ind_29feb, axis=0)
    Qobs = np.delete(Qobs, ind_29feb)
    Qsim = np.delete(Qsim, ind_29feb, axis=0)

    # Get the pre-processed data for the current watershed
    all_data.append({'Qobs': Qobs, 'Qsim': Qsim})
