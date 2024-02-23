"""Algorithms to calculate weights for annual maximum with observed and simulated data from HYDROTEL"""

import os

import numpy as np
import scipy.io

from xhydro.modelling.obj_funcs import get_objective_function


def apply_regional_weights():
    """
    Apply precalculated weights
    """

    data_folder = "C:\\Users\\AR21010\\HC3\\xhydro\\annual-maximum\\data"
    results_folder = "C:\\Users\\AR21010\\HC3\\xhydro\\annual-maximum\\results"
    save_folder = "C:\\Users\\AR21010\\HC3\\xhydro\\annual-maximum\\results"

    calculation_type = "AVG"
    # Type of simulations to use
    simulation_type = "HYDROTEL_6"  # 6 HYDROTEL simulation with ClimEx

    # Take all observed watersheds or only a selection of them (made by DEH)
    obs_stations_to_select = "selection"

    # Load data
    mat_obs_data = scipy.io.loadmat(
        os.path.join(
            data_folder,
            f"INFO_Crue_obs_stations_{simulation_type}_ClimEx_{obs_stations_to_select}.mat",
        )
    )
    obs_data = mat_obs_data["all_Qsim"]
    mat_avg_weights = scipy.io.loadmat(
        os.path.join(
            results_folder,
            f"{calculation_type}_{simulation_type}_{obs_stations_to_select}_weights.mat",
        )
    )
    regional_weights = mat_avg_weights["regional_weights"]
    mat_avg_data = scipy.io.loadmat(
        os.path.join(
            save_folder,
            f"{calculation_type}_{simulation_type}_{obs_stations_to_select}_ClimEx.mat",
        )
    )
    all_data = mat_avg_data["all_data"]

    # There should be 259 watersheds (all) or 96 watersheds (selection)
    n_watersheds = obs_data.shape[0]
    # Number of ClimEx members
    n_members = 50

    # All ClimEx 50-member simulation names
    ClimEx_sims = [
        "KDA",
        "KDB",
        "KDC",
        "KDD",
        "KDE",
        "KDF",
        "KDG",
        "KDH",
        "KDI",
        "KDJ",
        "KDK",
        "KDL",
        "KDM",
        "KDN",
        "KDO",
        "KDP",
        "KDQ",
        "KDR",
        "KDS",
        "KDT",
        "KDU",
        "KDV",
        "KDW",
        "KDX",
        "KDY",
        "KDZ",
        "KEA",
        "KEB",
        "KEC",
        "KED",
        "KEE",
        "KEF",
        "KEG",
        "KEH",
        "KEI",
        "KEJ",
        "KEK",
        "KEL",
        "KEM",
        "KEN",
        "KEO",
        "KEP",
        "KEQ",
        "KER",
        "KES",
        "KET",
        "KEU",
        "KEV",
        "KEW",
        "KEX",
    ]

    # Combine all Qobs and Qsim together
    for i in range(n_watersheds):
        for j in range(n_members):
            # Extract all simulated streamflows for the current member
            Qsim = obs_data[i, 0][ClimEx_sims[j]]

            # Remove nans and February 29
            Qsim = np.delete(
                Qsim, np.s_[54786:], axis=0
            )  # December 30, 2099 to December 31, 2100
            Qsim = np.delete(
                Qsim, np.s_[:1826], axis=0
            )  # January 01, 1950 to December 31, 1954
            ind29feb = np.where(
                (all_data[i, 0]["dates"][:, 1] == 2)
                & (all_data[i, 0]["dates"][:, 2] == 29)
            )[0]
            Qsim = np.delete(Qsim, ind29feb, axis=0)  # All February 29th

            # Copy the last day twice since there are missing 2 days at the end
            # of the dataset (it ends at December 29). This will not affect the
            # results since only the annual maxima will be used afterwards.
            Qsim = np.vstack([Qsim, [Qsim[-1]] * 2])

            # Compute the AVG average
            Qsim = np.sum(Qsim * regional_weights, axis=1)

            # Store the averaged data into the structure
            all_data[i, 0]["Qsim"][:, j] = Qsim

    # Save the AVG results
    save_path = os.path.join(
        save_folder,
        f"{calculation_type}_{simulation_type}_{obs_stations_to_select}_ClimEx.mat",
    )
    scipy.io.savemat(
        save_path,
        {"regional_weights": regional_weights, "all_data": all_data},
        format="5",
        long_field_names=True,
    )


def get_regional_weights():
    """
        Calculate weights following in period and a determined  algorithm type
    """
    # Set working directories
    data_folder = "C:\\Users\\AR21010\\HC3\\xhydro\\annual-maximum\\data"
    save_folder = "C:\\Users\\AR21010\\HC3\\xhydro\\annual-maximum\\results"

    # Determine the chosen period on which to compute the weights
    chosen_period = "year_by_year"
    # chosen_period = 'season_by_season'
    # chosen_period = 'month_by_month'

    # Determine the calculation type
    algorithm_type = "AVG"

    # Type of simulations to use
    simulation_type = "HYDROTEL_6"  # DEH 6 HYDROTEL simulations

    # Take all observed watersheds or only a selection of them (made by DEH)
    obs_stations_to_select = "selection"

    # Load data
    mat_data = scipy.io.loadmat(
        f"INFO_Crue_obs_stations_{simulation_type}_data_{obs_stations_to_select}.mat"
    )

    obs_stations_data = mat_data["obs_stations_data"][0]
    time_sim = mat_data["time_sim"][0]
    drainage_area_obs = mat_data["drainage_area_obs"][0]

    # There should be 259 watersheds (all) or 96 watersheds (selection)
    n_watersheds = obs_stations_data.shape[0]
    # Number of simulations in total used in the multi-model averaging approach
    n_models = obs_stations_data.shape[1]

    args = {
        "n_watersheds": n_watersheds,
        "time_sim": n_watersheds,
        "obs_stations_data": n_watersheds,
    }

    # Pre-process the data data
    all_data, dates_sim = pre_process_data(args)

    ind_time = None
    if chosen_period == "year_by_year":
        _, ind_time, _ = np.unique(dates_sim[:, 0], return_index=True)
    elif chosen_period == "season_by_season":
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
    elif chosen_period == "month_by_month":
        _, ind_time, _ = np.unique(dates_sim[:, :2], axis=0, return_index=True)

    ind_time = np.append(ind_time, len(dates_sim))

    # Pre-define the structures to store the results
    all_Qsim = []
    regional_weights = np.zeros((1, n_models))
    if len(ind_time) > 0:
        for y in range(len(ind_time) - 1):
            Qobs = []
            Qsim = []
            # Get the specific streamflow (streamflow divided by drainage area)
            for w in range(n_watersheds):
                Qobs = np.concatenate(
                    Qobs,
                    all_data[w]["Qobs"][ind_time[y] : ind_time[y + 1]]
                    / drainage_area_obs[w],
                )
                Qsim = np.concatenate(
                    Qsim,
                    all_data[w]["Qsim"][ind_time[y] : ind_time[y + 1], :]
                    / drainage_area_obs[w],
                )

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
    else:
        # Combine all Qobs and Qsim together
        Qobs = np.concatenate(
            [all_data[w]["Qobs"] / drainage_area_obs[w] for w in range(n_watersheds)]
        )
        Qsim = np.concatenate(
            [all_data[w]["Qsim"] / drainage_area_obs[w] for w in range(n_watersheds)]
        )

        # Apply random white noise around 1 (needed because some simulations have the same values which can generate problems)
        a = np.random.randn(Qsim.shape[0], Qsim.shape[1]) / 100
        a = a + 1
        Qsim = Qsim * a

        # Remove values where no observed streamflows are available
        ind_nan = np.isnan(Qobs)
        Qobs = Qobs[~ind_nan, :]
        Qsim = Qsim[~ind_nan, :]

        # # Remove values where no observed streamflows are available
        # ind_nan = np.where(np.isnan(Qobs))[0]
        # Qobs = np.delete(Qobs, ind_nan)
        # Qsim = np.delete(Qsim, ind_nan, axis=0)

        # Remove values where no simulated streamflows are available
        row, _ = np.where(np.isnan(Qsim))
        Qobs = np.delete(Qobs, row, axis=0)
        Qsim = np.delete(Qsim, row, axis=0)

        if algorithm_type == "AVG":
            # Simple Average (AVG)
            # A simple average where all weights are equal
            print("Currently calculating weights for the AVG...")
            regional_weights = np.ones((1, n_models)) / n_models  # AVG weights
        else:
            # Granger-Ramanathan Average method A (GRA)
            print("Currently calculating weights for GRA")

            X = Qsim
            Y = Qobs

            # Pre-define the structures to store the results
            all_Qsim = []
            regional_weights = np.zeros((1, n_models))
            # regional_weights = np.zeros((len(set(time_sim[:, 0])), n_models))
            regional_weights_results = []

            regional_weights[0, :] = np.linalg.inv(X.T @ X) @ X.T @ Y  # GRA weights

    # Compute the metrics for the multi-model averaged series
    Qobs = np.empty((0,))
    Qsim = np.empty((0,))

    # # Compute the metrics for the multi-model averaged series
    # Qobs = []
    # Qsim = []

    # Get the results for each watershed (no validation)
    for i in range(n_watersheds):
        # Get observed streamflow for the current watershed
        Qobs = all_data[i]["Qobs"]

        # Compute the annual maximum observed streamflow series
        Qx1day_obs = np.max(Qobs.reshape((365, -1)), axis=0)

        # Remove values where no observed streamflows are available
        ind_nan = np.isnan(Qobs)
        Qobs = Qobs[~ind_nan]

        # Sort all Qobs values to get percentiles
        Qobs_sort = np.sort(Qobs)

        # Compute the metrics if there are observed streamflow for the period
        if len(Qobs) > 0:

            # # Get the AVG averaged streamflow
            # Qsim = all_data[i]['Qsim']
            # Qsim = np.sum(Qsim * regional_weights, axis=1)

            if chosen_period != None:
                Qsim = np.zeros(len(dates_sim))
                for y in range(len(ind_time) - 1):
                    ind_start = (y - 1) * 365 + 1
                    ind_end = (y - 1) * 365 + 365

                    Qsim_tmp = all_data[i]["Qsim"][ind_time[y] : ind_time[y + 1], :]
                    Qsim[ind_time[y] : ind_time[y + 1]] = np.nansum(
                        Qsim_tmp * regional_weights[y, :], axis=1
                    )
                    all_Qsim.append({"Qsim": Qsim})
            else:
                # Get the GRA averaged streamflow
                Qsim = all_data[i]["Qsim"]
                Qsim = np.nansum(Qsim * regional_weights.T, axis=1)
                all_Qsim.append({"Qsim": Qsim})

            # Compute the annual maximum GRA streamflow series
            Qx1day_sim = np.nanmax(np.reshape(Qsim, (365, -1)))

            # Remove values where no observed streamflows are available
            Qsim = np.delete(Qsim, ind_nan)

            # Sort all Qobs values to get percentiles
            Qsim_sort = np.sort(Qsim)

            # Metrics
            KGE = get_objective_function(Qobs, Qsim, "kge")
            NSE = get_objective_function(Qobs, Qsim, "nse")

            NRMSE_90_PCTL = get_objective_function(
                Qobs_sort[round(len(Qobs_sort) * 0.90) :],
                Qsim_sort[round(len(Qsim_sort) * 0.90) :],
                "rrmse",
            )
            NRMSE_95_PCTL = get_objective_function(
                Qobs_sort[round(len(Qobs_sort) * 0.95) :],
                Qsim_sort[round(len(Qsim_sort) * 0.95) :],
                "rrmse",
            )
            NRMSE_99_PCTL = get_objective_function(
                Qobs_sort[round(len(Qobs_sort) * 0.99) :],
                Qsim_sort[round(len(Qsim_sort) * 0.99) :],
                "rrmse",
            )
            NRMSE_Qx1day = get_objective_function(Qx1day_obs, Qx1day_sim, "rrmse")

            # Save results

            regional_weights_results.append(
                {
                    "KGE": KGE,
                    "NSE": NSE,
                    "NRMSE_90_PCTL": NRMSE_90_PCTL,
                    "NRMSE_95_PCTL": NRMSE_95_PCTL,
                    "NRMSE_99_PCTL": NRMSE_99_PCTL,
                    "NRMSE_Qx1day": NRMSE_Qx1day,
                }
            )
        else:
            # Otherwise put NaNs for all metrics
            regional_weights_results.append(
                {
                    "KGE": np.nan,
                    "NSE": np.nan,
                    "NRMSE_90_PCTL": np.nan,
                    "NRMSE_95_PCTL": np.nan,
                    "NRMSE_99_PCTL": np.nan,
                    "NRMSE_Qx1day": np.nan,
                }
            )

    if algorithm_type == "AVG":
        # Save the AVG results
        save_path = os.path.join(
            save_folder, f"AVG_{simulation_type}_{obs_stations_to_select}_weights.mat"
        )
        scipy.io.savemat(
            save_path,
            {
                "regional_weights": regional_weights,
                "regional_weights_results": regional_weights_results,
            },
            format="5",
            long_field_names=True,
        )

    else:
        # Save the GRA results
        os.chdir(save_folder)
        np.savez_compressed(
            f"{algorithm_type}_{simulation_type}_{obs_stations_to_select}_weights.npz",
            regional_weights=regional_weights,
            regional_weights_results=regional_weights_results,
        )

        np.savez_compressed(
            f"{algorithm_type}_{simulation_type}_{obs_stations_to_select}_weights_with_Qsim.npz",
            regional_weights=regional_weights,
            regional_weights_results=regional_weights_results,
            all_Qsim=all_Qsim,
        )


def pre_process_data(args):
    n_watersheds = args["n_watersheds"]
    time_sim = args["time_sim"]
    obs_stations_data = args["obs_stations_data"]

    all_data = []
    dates_sim = []
    for i in range(n_watersheds):
        dates_sim = time_sim  # Start from the full vector date

        # Get all observed and simulated streamflow
        Qobs = obs_stations_data[i]["Qobs"][: len(dates_sim)]
        Qsim = obs_stations_data[i]["Qsim"][: len(dates_sim), :]

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
        all_data.append({"Qobs": Qobs, "Qsim": Qsim})

    return all_data, dates_sim
