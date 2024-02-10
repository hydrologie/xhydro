"""Perform the cross-validation for the optimal interpolation."""
import os
from multiprocessing import Pool

import numpy as np

from .functions import ECF_climate_correction as ecf_cc
from .functions import mathematical_algorithms as ma
from .functions import optimal_interpolation as opt
from .functions import utilities as util


def execute(
    start_date,
    end_date,
    files,
    ratio_var_bg=0.15,
    percentiles=[0.25, 0.50, 0.75, 1.00],
    iterations=10,
    parallelize=False,
):
    """
    Run the interpolation algorithm for cross-validation.

    Parameters
    ----------
    start_date : datetime
        Start date of the analysis.
    end_date : datetime
        End date of the analysis.
    files : list(str)
        List of files path for getting flows and watersheds info.
    ratio_var_bg : float, optional
        Ratio for background variance (default is 0.15).
    percentiles : list(float), optional
        List of percentiles to analyze (default is [0.25, 0.50, 0.75, 1.00]).
    iterations: int, optional
        Number of iterations for the interpolation (default is 10).
    parallelize : bool, optional
        Execute the profiler in parallel or in series (default is False).

    Returns
    -------
    list
        A list containing the results of the interpolated percentiles flow
    """
    # Run the code
    args = {
        'start_date' : start_date,
        'end_date' : end_date,
        'files': files,
        'ratio': ratio_var_bg,
        'percentiles': percentiles
    }

    time_range = (end_date - start_date).days

    results = execute_interpolation(
        start_date,
        end_date,
        time_range,
        files,
        ratio_var_bg,
        percentiles,
        iterations,
        parallelize=parallelize,
    )

    return results


def execute_interpolation(
    start_date, end_date, time_range, files, ratio_var_bg, percentiles, iterations, parallelize
):
    """
    Execute the main code, including setting constants to files, times, etc.
    Heavily modified to parallelize and to optimize.

    Parameters:
    - start_date (datetime.date): The start date of the interpolation period.
    - end_date (datetime.date): The end date of the interpolation period.
    - files (list): List of files containing Hydrotel runs and observations.
    - ratio_var_bg (float): Ratio for background variance.
    - percentiles (list): List of desired percentiles for flow quantiles.
    - iterations (int): The number of iterations for the interpolation.
    - parallelize (bool): Flag indicating whether to parallelize the interpolation.

    Returns:
    list: A list containing the flow quantiles for each desired percentile.
'
    """
    (
        stations_info,
        stations_mapping,
        stations_validation,
        flow_obs,
        flow_sim,
    ) = util.load_files(files)

    stations_id = [station[0] for station in stations_validation]

    args = {
        "flow_obs": flow_obs,
        "flow_sim": flow_sim,
        "start_date": start_date,
        "end_date": end_date,
        "time_range": time_range,
        "stations_info": stations_info,
        "stations_mapping": util.convert_list_to_dict(stations_mapping),
        "stations_id": stations_id
    }

    data = retreive_data(args)

    station_count = data["station_count"]
    drainage_area = data["drainage_area"]
    centroid_lat = data["centroid_lat"]
    centroid_lon = data["centroid_lon"]
    selected_flow_obs = data["selected_flow_obs"]
    selected_flow_sim = data["selected_flow_sim"]

    x, y = ma.latlon_to_xy(
        centroid_lat,
        centroid_lon,
        np.array([45] * len(centroid_lat)),
        np.array([-70] * len(centroid_lat)),
    )  # Projete dans un plan pour avoir des distances en km

    x_points, y_points = standardize_points_with_roots(
        x, y, station_count, drainage_area
    )

    # create the weighting function parameters
    ecf_fun, par_opt = ecf_cc.correction(
        selected_flow_obs,
        selected_flow_sim,
        x_points,
        y_points,
        savename="test",
        iteration_count=iterations,
    )

    # Create tuple with all info, required for the parallel processing.
    args = {
        "station_count": station_count,
        "selected_flow_obs": selected_flow_obs,
        "selected_flow_sim": selected_flow_sim,
        "ecf_fun": ecf_fun,
        "par_opt": par_opt,
        "x_points": x_points,
        "y_points": y_points,
        "time_range": time_range,
        "drainage_area": drainage_area,
        "ratio_var_bg": ratio_var_bg,
        "percentiles": percentiles,
        "iterations": iterations,
    }
    flow_quantiles = parallelize_operation(args, parallelize=parallelize)

    util.write_netcdf_debit("cross_validation_result",
                            stations_id,
                            centroid_lon,
                            centroid_lat,
                            drainage_area,
                            time_range,
                            percentiles,
                            flow_quantiles
                        )

    return flow_quantiles




def initialize_data_arrays(time_range, station_count):
    """
    Initialize empty data arrays for later use.

    Parameters:
    - time_range (int): The number of time steps in the data arrays.
    - station_count (int): The number of stations or data points.

    Returns:
    tuple: A tuple containing initialized empty arrays for selected flow observations,
           selected flow simulations, centroid latitude, centroid longitude, and drained area.
    """
    selected_flow_obs = np.empty((time_range, station_count))
    selected_flow_sim = np.empty((time_range, station_count))
    centroid_lat = np.empty(station_count)
    centroid_lon = np.empty(station_count)
    drained_area = np.empty(station_count)

    return (
        selected_flow_obs,
        selected_flow_sim,
        centroid_lat,
        centroid_lon,
        drained_area,
    )


def retreive_data(args):
    """
    Retrieve data from files to populate the Optimal Interpolation (OI) algorithm.

    Parameters:
    - args (dict): A dictionary containing the necessary information to retrieve and preprocess data.
                  Keys include 'flow_obs', 'flow_sim', 'start_date', 'end_date', 'time_range',
                  'stations_info', 'stations_mapping', and 'stations_id'.

    Returns:
    dict: A dictionary containing the retrieved and preprocessed data for OI algorithm.
    """
    flow_obs = args["flow_obs"]
    flow_sim = args["flow_sim"]
    start_date = args["start_date"]
    end_date = args["end_date"]
    time_range = args["time_range"]
    stations_info = args["stations_info"]
    stations_mapping = args["stations_mapping"]
    stations_id = args["stations_id"]

    station_count = len(stations_id)

    centroid_lat, centroid_lon, drainage_area = util.initialize_nan_arrays(
        station_count, 3
    )
    selected_flow_obs, selected_flow_sim = util.initialize_nan_arrays(
        (time_range, station_count), 2
    )
    for i in range(0, station_count):
        station_id = stations_id[i]
        associate_section = stations_mapping[station_id]

        index_section = util.find_index(flow_sim, "station_id", associate_section)
        index_station = util.find_index(flow_obs, "station_id", station_id)

        sup_sim = flow_sim.drainage_area[index_section].item()
        sup_obs = flow_obs.drainage_area[index_station].item()

        drainage_area[i] = sup_obs

        selected_flow_sim[:, i] = (
            flow_sim.Dis.sel(time=slice(start_date, end_date), station=index_section)[
                0:time_range
            ]
            / sup_sim
        )
        selected_flow_obs[:, i] = (
            flow_obs.Dis.sel(time=slice(start_date, end_date), station=index_station)[
                0:time_range
            ]
            / sup_obs
        )

        position_info = np.where(np.array(stations_info) == station_id)
        station_info = stations_info[position_info[0].item()]
        centroid_lon[i], centroid_lat[i] = station_info[4], station_info[5]

    # Transformation log-débit pour l'interpolation
    selected_flow_obs = np.log(selected_flow_obs)
    selected_flow_sim = np.log(selected_flow_sim)

    returned_dict = {
        "station_count": station_count,
        "stations_id": stations_id,
        "drainage_area": drainage_area,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "selected_flow_sim": selected_flow_sim,
        "selected_flow_obs": selected_flow_obs
    }

    return returned_dict


def standardize_points_with_roots(x, y, station_count, drained_area):
    """'
    Standardize points with roots based on drainage area.

    Parameters:
    - x (array-like): Array of x-coordinates of the original points.
    - y (array-like): Array of y-coordinates of the original points.
    - station_count (int): The number of stations or points.
    - drained_area (array-like): Array of drainage areas corresponding to each station.

    Returns:
    tuple: A tuple containing standardized x and y points with roots.
    """

    x_points = np.empty((4, station_count))
    y_points = np.empty((4, station_count))
    for i in range(station_count):
        root_area = np.sqrt(drained_area[i])
        xv = [x[i] - (root_area / 2), x[i] + root_area / 2]
        yv = [y[i] - (root_area / 2), y[i] + root_area / 2]
        [x_p, y_p] = np.meshgrid(xv, yv)

        x_points[:, i] = x_p.reshape(2 * len(np.transpose(x_p)))
        y_points[:, i] = y_p.reshape(2 * len(np.transpose(y_p)))

    return x_points, y_points


def parallelize_operation(args, parallelize=True):
    """Run the interpolator on the cross-validation and manage parallelization.

    New section for parallel computing. Several steps performed:
       1. Estimation of available computing power. Carried out by taking
          the number of threads available / 2 (to get the number of cores)
          then subtracting 2 to keep 2 available for other tasks.
       2. Starting a parallel computing pool
       3. Create an iterator (iterable) containing the data required
          by the new function, which loops on each station in leave-one-out
          validation. It's embarassingly parallelizable, so very useful.
       4. Run pool.map, which maps inputs (iterators) to the function.
       5. Collect the results and unzip the tuple returning from pool.map.
       6. Close the pool and return the parsed results.
    """
    station_count = args["station_count"]
    percentiles = args["percentiles"]
    time_range = args["time_range"]

    flow_quantiles = util.initialize_nan_arrays(
        (time_range, station_count), len(percentiles)
    )

    # Parallel
    if parallelize:
        processes_count = os.cpu_count() / 2 - 1
        p = Pool(int(processes_count))
        args_i = [(i, args) for i in range(0, station_count)]
        flow_quantiles_station = zip(
            *p.map(opt.loop_optimal_interpolation_stations, args_i)
        )
        p.close()
        p.join()
        flow_quantiles_station = tuple(flow_quantiles_station)
        for k in range(0, len(percentiles)):
            flow_quantiles[k][:] = np.array(flow_quantiles_station[k]).T

    # Serial
    else:
        for i in range(0, station_count):
            flow_quantiles_station = opt.loop_optimal_interpolation_stations(
                (i, args)
            )
            for k in range(0, len(percentiles)):
                flow_quantiles[k][:, i] = flow_quantiles_station[k][:]




    return flow_quantiles
