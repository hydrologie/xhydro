"""Compare results between simulations and observations."""

import sys

import numpy as np

import xhydro.optimal_interpolation.functions.utilities as util
from xhydro.modelling.obj_funcs import get_objective_function


def compare(start_date, end_date, files, percentile_to_plot=50, show_comparaison=True):
    """Start the computation of the comparison method.

    Parameters
    ----------
    start_date : datetime
        Start date of the analysis.
    end_date : datetime
        End date of the analysis.
    files : list of str
        List of files path for getting observed, simulated, and leave-one-out cross-validation flows.
    percentile_to_plot : int, optional
        Percentile value to plot (default is 50).
    show_comparaison : bool, optional
        Whether to display the comparison plots (default is True).
    """
    time = (end_date - start_date).days

    station_validation, station_mapping, obs_data, sim_data, l1o_data = util.load_files(
        files
    )

    # Read station id
    stations_id = obs_data["station_id"]
    sections_id = sim_data["station_id"]
    l1o_stations_id = l1o_data["station_id"]

    # Read drainage area
    da_obs = obs_data.drainage_area
    da_sim = sim_data.drainage_area
    da_l1o = l1o_data.drainage_area

    # Read discharge percentiles
    dis_obs = obs_data.Dis
    dis_sim = sim_data.Dis
    dis_l1o = l1o_data.Dis

    # Read percentiles list (which percentile thresholds were used)
    percentile = l1o_data.percentile

    # Find position of the desired percentile
    idx_pct = np.where(percentile == percentile_to_plot)[0]
    if idx_pct is None:
        sys.exit(
            "The desired percentile is not computed in the results file \
             provided. Please make sure your percentile value is expressed \
             in percent (i.e. 50th percentile = 50)"
        )

    time_range = int(time)
    station_count = len(station_validation)
    debit_sim, debit_obs, debit_l1o = util.initialize_nan_arrays(
        (time_range, station_count), 3
    )

    for i in range(0, station_count):
        print("Lecture des données..." + str(i + 1) + "/" + str(station_count))

        station_id = station_validation[i][0]
        associate_section = util.find_station_section(station_mapping, station_id)

        idx_section = util.find_index(sections_id, "station_id", associate_section)
        idx_stat = util.find_index(stations_id, "station_id", station_id)
        idx_stat_l1o = util.find_index(l1o_stations_id, "station_id", station_id)

        sup_sim = da_sim[idx_section].item()
        sup_obs = da_obs[idx_stat].item()
        sup = da_l1o[idx_stat_l1o].item()

        debit_sim[:, i] = dis_sim[idx_section, 0:time_range].values[:] / sup_sim
        debit_obs[:, i] = dis_obs[idx_stat, 0:time_range].values[:] / sup_obs
        debit_l1o[:, i] = dis_l1o[idx_pct, idx_stat_l1o, 0:time_range].values[:] / sup

    kge, nse, kge_l1o, nse_l1o = util.initialize_nan_arrays(station_count, 4)

    for n in range(0, station_count):
        kge[n] = get_objective_function(debit_obs[:, n], debit_sim[:, n], "kge")
        kge[n] = get_objective_function(debit_obs[:, n], debit_sim[:, n], "nse")
        kge[n] = get_objective_function(debit_obs[:, n], debit_l1o[:, n], "kge")
        kge[n] = get_objective_function(debit_obs[:, n], debit_l1o[:, n], "nse")

    if show_comparaison:
        util.plot_results(kge, kge_l1o, nse, nse_l1o)
