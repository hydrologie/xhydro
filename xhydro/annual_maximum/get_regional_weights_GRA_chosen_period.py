import os
import numpy as np
import scipy.io
from scipy.stats import rankdata


def obj_fun_KGE(obs, sim):
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    obs_skew = np.mean((obs - obs_mean) ** 3) / obs_std ** 3
    sim_skew = np.mean((sim - sim_mean) ** 3) / sim_std ** 3
    obs_corr = np.corrcoef(obs[:-1], obs[1:])[0, 1]
    sim_corr = np.corrcoef(sim[:-1], sim[1:])[0, 1]

    kge = 1 - np.sqrt((obs_corr - 1) ** 2 + (sim_corr - 1) ** 2 + (obs_std / sim_std - 1) ** 2)
    return kge


def obj_fun_NSE(obs, sim):
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    obs_var = np.sum((obs - obs_mean) ** 2)
    nse = 1 - np.sum((sim - obs) ** 2) / obs_var
    return nse


def obj_fun_NRMSE(obs, sim):
    rmse = np.sqrt(np.mean((obs - sim) ** 2))
    nrmse = rmse / (np.max(obs) - np.min(obs))
    return nrmse


# Set working directories
work_folder = os.getcwd()
data_folder = 'C:\\Users\\AR21010\\Dropbox\\xhydro\\INFO-Crue\\0_data\\2_extracted_data'
results_folder = 'C:\\Users\\AR21010\\Dropbox\\xhydro\\INFO-Crue\\0_data\\3_results'
save_folder = 'C:\\Users\\AR21010\\Dropbox\\xhydro\\INFO-Crue\\0_data\\3_results'

# Type of simulations to use
# simulation_type = 'HYDROTEL_6_OI';  # Optimal interpolation (OI)
# simulation_type = 'HYDROTEL_6';  # DEH 6 HYDROTEL simulations
# simulation_type = 'HYDROTEL_144';  # ETS 144 HYDROTEL simulations
simulation_type = 'HYDROTEL_150';  # ETS 144 + 6 DEH HYDROTEL simulations

# Take all observed watersheds or only a selection of them (made by DEH)
# obs_stations_to_select = 'all';
obs_stations_to_select = 'selection';

# Determine the chosen period on which to compute the weights
chosen_period = 'year_by_year'
# chosen_period = 'season_by_season'
# chosen_period = 'month_by_month'

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

# Pre-define the structures to store the results
all_Qsim = []
regional_weights = np.zeros((len(set(time_sim[:, 0])), n_models))
regional_weights_results = []

# Combine all Qobs and Qsim together based on the chosen period
for i, period in enumerate(['year_by_year', 'season_by_season', 'month_by_month']):
    if chosen_period != period:
        continue

    ind_time = None
    if period == 'year_by_year':
        _, ind_time, _ = np.unique(dates_sim[:, 0], return_index=True)
    elif period == 'season_by_season':
        dates_season = dates_sim[:, :2]
        # Winter (JFM)
        ind = dates_season[:, 1] <= 3
        dates_season[ind, 1] = 1
        # Spring (AMJ)
        ind = (dates_season[:, 1] >= 4) & (dates_season[:, 1] <= 6)
        dates_season[ind, 1] = 2
        # Summer (JAS)
        ind = (dates_season[:, 1] >= 7) & (dates_season[:, 1] <= 9)
        dates_season[ind, 1] = 3
        # Autumn (OND)
        ind = (dates_season[:, 1] >= 10) & (dates_season[:, 1] <= 12)
        dates_season[ind, 1] = 4
        _, ind_time, _ = np.unique(dates_season, axis=0, return_index=True)
    elif period == 'month_by_month':
        _, ind_time, _ = np.unique(dates_sim[:, :2], axis=0, return_index=True)

    ind_time = np.append(ind_time, len(dates_sim))

    for y in range(len(ind_time) - 1):
        Qobs = []
        Qsim = []
        # Get the specific streamflow (streamflow divided by drainage area)
        for w in range(n_watersheds):
            Qobs = np.append(Qobs, all_data[w]['Qobs'][ind_time[y]:ind_time[y + 1]] / drainage_area_obs[w])
            Qsim = np.append(Qsim, all_data[w]['Qsim'][ind_time[y]:ind_time[y + 1], :] / drainage_area_obs[w])

        # Apply random white noise around 1 (needed because some
        # simulations have the same values which can generate problems)
        a = np.random.randn(Qsim.shape[0], Qsim.shape[1]) / 100
        a = a + 1
        Qsim = Qsim * a

        # Remove values where no observed streamflows are available
        ind_nan = np.isnan(Qobs)
        Qobs = np.delete(Qobs, np.where(ind_nan))
        Qsim = np.delete(Qsim, np.where(ind_nan), axis=0)

        # Remove values where no simulated streamflows are available
        row, _ = np.where(np.isnan(Qsim))
        Qobs = np.delete(Qobs, row)
        Qsim = np.delete(Qsim, row, axis=0)

        # Granger-Ramanathan Average method A (GRA)
        X = Qsim
        Y = Qobs
        regional_weights[y, :] = np.linalg.inv(X.T @ X) @ X.T @ Y  # GRA weights

for i in range(n_watersheds):
    # Get observed streamflow for the current watershed
    Qobs = all_data[i]['Qobs']

    # Compute the annual maximum observed streamflow series
    Qx1day_obs = np.nanmax(np.reshape(Qobs, (365, -1)))

    # Remove values where no observed streamflows are available
    ind_nan = np.isnan(Qobs)
    Qobs = np.delete(Qobs, np.where(ind_nan))

    # Sort all Qobs values to get percentiles
    Qobs_sort = np.sort(Qobs)

    # Compute the metrics if there are observed streamflow for the period
    if len(Qobs) > 0:
        # Get the GRA averaged streamflow for each period
        Qsim = np.zeros(len(dates_sim))
        for y in range(len(ind_time) - 1):
            ind_start = (y - 1) * 365 + 1
            ind_end = (y - 1) * 365 + 365

            Qsim_tmp = all_data[i]['Qsim'][ind_time[y]:ind_time[y + 1], :]
            Qsim[ind_time[y]:ind_time[y + 1]] = np.nansum(Qsim_tmp * regional_weights[y, :], axis=1)
            all_Qsim.append({'Qsim': Qsim})

        # Compute the annual maximum GRA streamflow series
        Qx1day_sim = np.nanmax(np.reshape(Qsim, (365, -1)))

        # Remove values where no observed streamflows are available
        Qsim = np.delete(Qsim, np.where(ind_nan))

        # Sort all Qobs values to get percentiles
        Qsim_sort = np.sort(Qsim)

        regional_weights_results.append({
            'KGE': obj_fun_KGE(Qobs, Qsim),
            'NSE': obj_fun_NSE(Qobs, Qsim),
            'NRMSE_90_PCTL': obj_fun_NRMSE(Qobs_sort[int(round(len(Qobs_sort) * 0.90)):]),
            'NRMSE_95_PCTL': obj_fun_NRMSE(Qobs_sort[int(round(len(Qobs_sort) * 0.95)):]),
            'NRMSE_99_PCTL': obj_fun_NRMSE(Qobs_sort[int(round(len(Qobs_sort) * 0.99)):]),
            'NRMSE_Qx1day': obj_fun_NRMSE(Qx1day_obs, Qx1day_sim),
        })
    else:
        # Otherwise put NaNs for all metrics
        regional_weights_results.append({
            'KGE': np.nan,
            'NSE': np.nan,
            'NRMSE_90_PCTL': np.nan,
            'NRMSE_95_PCTL': np.nan,
            'NRMSE_99_PCTL': np.nan,
            'NRMSE_Qx1day': np.nan,
        })

# Save the GRA results
os.chdir(save_folder)
np.savez_compressed(f'GRA_{simulation_type}_{obs_stations_to_select}_{chosen_period}_weights.npz',
                    regional_weights=regional_weights,
                    regional_weights_results=regional_weights_results)

np.savez_compressed(f'GRA_{simulation_type}_{obs_stations_to_select}_{chosen_period}_weights_with_Qsim.npz',
                    regional_weights=regional_weights,
                    regional_weights_results=regional_weights_results,
                    all_Qsim=all_Qsim)
os.chdir(work_folder)
