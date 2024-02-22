"""Package containing the optimal interpolation functions."""

import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

import xhydro.optimal_interpolation.functions.ECF_climate_correction as ecf_cc
import xhydro.optimal_interpolation.functions.mathematical_algorithms as ma
import xhydro.optimal_interpolation.functions.utilities as util

from .mathematical_algorithms import calculate_average_distance


def optimal_interpolation(oi_input: dict, args: dict):
    """Perform optimal interpolation to estimate values at specified locations.

    Parameters
    ----------
    oi_input : dict
        Input data dictionary containing necessary information for interpolation.
    args : dict
        Additional arguments and state information for the interpolation process.

    Returns
    -------
    tuple
        A tuple containing the updated oi_output dictionary and the modified args dictionary.
    """
    if len(args) == 0:
        args = {}

    estimated_count = (
        1 if len(oi_input["x_est"].shape) == 1 else oi_input["x_est"].shape[1]
    )
    observed_count = len(oi_input["x_obs"][0, :])
    oi_output = oi_input

    cond = 0

    if isinstance(args, dict):
        if "x_obs" in args:
            cond = (
                np.array_equal(args["x_est"], oi_input["x_est"])
                and np.array_equal(args["y_est"], oi_input["y_est"])
            ) and (
                np.array_equal(args["x_obs"], oi_input["x_obs"])
                and np.array_equal(args["y_obs"], oi_input["y_obs"])
            )
    if cond == 0:
        distance_obs_vs_obs = calculate_average_distance(
            oi_input["x_obs"], oi_input["y_obs"]
        )
    else:
        distance_obs_vs_obs = args["Doo"]

    args["x_obs"] = oi_input["x_obs"]
    args["y_obs"] = oi_input["y_obs"]
    args["Doo"] = distance_obs_vs_obs

    covariance_obs_vs_obs = oi_input["error_cov_fun"](distance_obs_vs_obs) / oi_input[
        "error_cov_fun"
    ](0)

    BEo_j = np.tile(oi_input["bg_var_obs"], (observed_count, 1))
    BEo_i = np.tile(
        np.resize(oi_input["bg_var_obs"], (1, observed_count)), (observed_count, 1)
    )

    Bij = covariance_obs_vs_obs * np.sqrt(BEo_j) / np.sqrt(BEo_i)

    OEo_j = np.tile(oi_input["var_obs"], (observed_count, 1))
    OEo_i = np.tile(oi_input["var_obs"], (1, observed_count))

    Oij = (np.sqrt(OEo_j) * np.sqrt(OEo_i)) * np.eye(len(OEo_j), len(OEo_j[0])) / BEo_i

    if cond == 0:
        distance_obs_vs_est = np.zeros((1, observed_count))
        x_est = oi_input["x_est"]
        y_est = oi_input["y_est"]

        for i in range(estimated_count):
            for j in range(observed_count // 2):
                distance_obs_vs_est[i, j] = np.mean(
                    np.sqrt(
                        np.power(oi_input["x_obs"][:, j] - x_est[:], 2)
                        + np.power(oi_input["y_obs"][:, j] - y_est[:], 2)
                    )
                )
                distance_obs_vs_est[i, -j - 1] = np.mean(
                    np.sqrt(
                        np.power(oi_input["x_obs"][:, -j - 1] - x_est[:], 2)
                        + np.power(oi_input["y_obs"][:, -j - 1] - y_est[:], 2)
                    )
                )
    else:
        distance_obs_vs_est = args["distance_obs_vs_est"]

    args["x_est"] = oi_input["x_est"]
    args["y_est"] = oi_input["y_est"]
    args["distance_obs_vs_est"] = distance_obs_vs_est

    BEe = np.tile(
        np.resize(oi_input["bg_var_est"], (1, observed_count)), (estimated_count, 1)
    )
    BEo = np.tile(oi_input["bg_var_obs"], (estimated_count, 1))

    Coe = oi_input["error_cov_fun"](distance_obs_vs_est) / oi_input["error_cov_fun"](0)

    Bei = np.resize(Coe * np.sqrt(BEe) / np.sqrt(BEo), (observed_count, 1))

    departures = oi_input["bg_departures"].reshape((1, len(oi_input["bg_departures"])))

    weights = np.linalg.solve(Bij + Oij, Bei)
    weights = weights.reshape((1, len(weights)))

    oi_output["v_est"] = oi_input["bg_est"] + np.sum(weights * departures)
    oi_output["var_est"] = oi_input["bg_var_est"] * (
        1 - np.sum(Bei[:, 0] * weights[0, :])
    )

    return oi_output, args


def loop_optimal_interpolation_stations(args):
    """Apply optimal interpolation to a single validation site (station) for the selected time range.

    Parameters
    ----------
    args : tuple
        A tuple containing the station index and a dictionary with various information. The dictionary should include
        keys such as 'station_count', 'time_range', 'percentiles', 'selected_flow_obs', 'selected_flow_sim',
        'ratio_var_bg', 'ecf_fun', 'par_opt', 'x_points', 'y_points', and 'drainage_area'.

    Returns
    -------
    list
        The quantiles of the flow values for each percentile over the specified time range.
    """
    (station_index, args) = args

    station_count = args["station_count"]
    time_range = args["time_range"]
    percentiles = args["percentiles"]
    selected_flow_obs = args["selected_flow_obs"]
    selected_flow_sim = args["selected_flow_sim"]
    ratio_var_bg = args["ratio_var_bg"]
    ecf_fun = args["ecf_fun"]
    par_opt = args["par_opt"]
    x_points = args["x_points"]
    y_points = args["y_points"]
    drainage_area = args["drainage_area"]

    # Get the number of stations from the dataset
    index = range(0, station_count)

    # Define the exit vectors for a single catchment at a time since we will
    # work on stations in parallel.
    flow_quantiles = util.initialize_nan_arrays(time_range, len(percentiles))

    # Start cross-validation, getting indexes of the validation set.
    index_validation = station_index
    index_calibration = np.setdiff1d(index, index_validation)

    # Compute difference between the obs and sim log-transformed flows for the
    # calibration basins
    difference = (
        selected_flow_obs[:, index_calibration]
        - selected_flow_sim[:, index_calibration]
    )
    vsim_at_est = selected_flow_sim[:, index_validation]

    # Create and update dictionary for the interpolation input data. This object
    # is updated later in the code and iterated, and updated. So keeps things
    # efficient this way.
    oi_input = {}
    oi_input.update(
        {
            "var_obs": ratio_var_bg,
            "error_cov_fun": partial(ecf_fun, par=par_opt),
            "x_est": x_points[:, index_validation],
            "y_est": y_points[:, index_validation],
        }
    )

    # Object with the arguments to the OI that is passed along at each time step
    # and updated. Named oi_args to not confuse with this functions' own args.
    oi_args = {}

    # For each timestep, build the interpolator and apply to the cross-
    # validation catchment.
    for j in range(time_range):
        # Need to skip days where no value exists for verification
        if not np.isnan(selected_flow_obs[j, index_validation]):
            val = difference[j, :]
            idx = ~np.isnan(val)

            # Update the optimal interpolation dictionary.
            oi_input.update(
                {
                    "x_obs": x_points[:, index_calibration[idx]],
                    "y_obs": y_points[:, index_calibration[idx]],
                    "bg_departures": difference[j, idx],
                    "bg_var_obs": np.ones(idx.sum()),
                    "bg_est": vsim_at_est[j],
                    "bg_var_est": 1,
                }
            )

            # Apply the interpolator and get outputs
            oi_output, oi_args = optimal_interpolation(oi_input, oi_args)

            # Get variance properties
            var_bg = np.var(difference[j, idx])
            var_est = oi_output["var_est"] * var_bg

            # Get the percentile values for each desired percentile.
            vals = norm.ppf(percentiles, loc=oi_output["v_est"], scale=np.sqrt(var_est))

            # Get the values in real units and scale according to drainage area
            vals = np.exp(vals) * drainage_area[station_index]
            for k in range(0, len(percentiles)):
                flow_quantiles[k][j] = vals[k]

    # return the flow quantiles as desired.
    return flow_quantiles


def execute_interpolation(
    start_date,
    end_date,
    time_range,
    files,
    ratio_var_bg,
    percentiles,
    iterations,
    parallelize,
    write_file,
):
    """Execute the main code, including setting constants to files, times, and other hyperparemeters.

    Parameters
    ----------
    start_date : datetime date
        The start date of the interpolation period.
    end_date : datetime date
        The end date of the interpolation period.
    time_range : int
        The number of time steps in the data arrays.
    files : list
        List of files containing Hydrotel runs and observations.
    ratio_var_bg : float
        Ratio for background variance.
    percentiles : list
        List of desired percentiles for flow quantiles.
    iterations : int
        The number of iterations for the interpolation.
    parallelize : bool
        Flag indicating whether to parallelize the interpolation.
    write_file : str
        Name of the NetCDF file to be created.

    Returns
    -------
    list
        The flow quantiles for each desired percentile.
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
        "stations_id": stations_id,
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

    time_vector = pd.date_range(start=start_date, end=end_date)

    util.write_netcdf_debit(
        write_file=write_file,
        station_id=stations_id,
        lon=centroid_lon,
        lat=centroid_lat,
        drain_area=drainage_area,
        time=time_vector,
        percentile=percentiles,
        discharge=flow_quantiles,
    )

    return flow_quantiles


def initialize_data_arrays(time_range, station_count):
    """Initialize empty data arrays for later use.

    Parameters
    ----------
    time_range : int
        The number of time steps in the data arrays.
    station_count : int
        The number of stations or data points.

    Returns
    -------
    tuple
        Initialized empty arrays for selected flow observations, selected flow simulations, centroid latitude, centroid
        longitude, and drained area.
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
    """Retrieve data from files to populate the Optimal Interpolation (OI) algorithm.

    Parameters
    ----------
    args : dict
        A dictionary containing the necessary information to retrieve and preprocess data. Keys include 'flow_obs',
        'flow_sim', 'start_date', 'end_date', 'time_range', 'stations_info', 'stations_mapping', and 'stations_id'.

    Returns
    -------
    dict
        The retrieved and preprocessed data for OI algorithm.
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
        "selected_flow_obs": selected_flow_obs,
    }

    return returned_dict


def standardize_points_with_roots(x, y, station_count, drained_area):
    """Standardize points with roots based on drainage area.

    Parameters
    ----------
    x : array-like
        X-coordinates of the original points.
    y : array-like
        Y-coordinates of the original points.
    station_count : int
        The number of stations or points.
    drained_area : array-like
        Drainage areas corresponding to each station.

    Returns
    -------
    tuple
        Standardized x and y points with roots.
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

    Parameters
    ----------
    args : dict
        A dictionary containing the necessary information to retrieve and preprocess data. Keys include 'flow_obs',
        'flow_sim', 'start_date', 'end_date', 'time_range', 'stations_info', 'stations_mapping', 'stations_id',
        'percentiles' and 'station_count'.
    parallelize : bool
        Flag to make the code run in parallel. True to use parallelization.

    Returns
    -------
    array-like
        Flow quantiles associated to the desired percentiles after optimal interpolation.
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
            *p.map(loop_optimal_interpolation_stations, args_i)
        )
        p.close()
        p.join()
        flow_quantiles_station = tuple(flow_quantiles_station)
        for k in range(0, len(percentiles)):
            flow_quantiles[k][:] = np.array(flow_quantiles_station[k]).T

    # Serial
    else:
        for i in range(0, station_count):
            flow_quantiles_station = loop_optimal_interpolation_stations((i, args))
            for k in range(0, len(percentiles)):
                flow_quantiles[k][:, i] = flow_quantiles_station[k][:]

    return flow_quantiles
