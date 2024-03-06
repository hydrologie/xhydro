"""Package containing the optimal interpolation functions."""

import os
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np
from numpy import ndarray, dtype, floating

from scipy.stats import norm
import xarray as xr

import xhydro.optimal_interpolation.ECF_climate_correction as ecf_cc  # noqa
import xhydro.optimal_interpolation.mathematical_algorithms as ma
import xhydro.optimal_interpolation.utilities as util

from .mathematical_algorithms import calculate_average_distance

__all__ = ["optimal_interpolation",
           "execute_interpolation",
           ]


def optimal_interpolation(oi_input: dict, args: dict) -> tuple[dict, dict]:
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

    beo_j = np.tile(oi_input["bg_var_obs"], (observed_count, 1))
    beo_i = np.tile(
        np.resize(oi_input["bg_var_obs"], (1, observed_count)), (observed_count, 1)
    )

    b_ij = covariance_obs_vs_obs * np.sqrt(beo_j) / np.sqrt(beo_i)

    o_eo_j = np.tile(oi_input["var_obs"], (observed_count, 1))
    o_eo_i = np.tile(oi_input["var_obs"], (1, observed_count))

    o_ij = (np.sqrt(o_eo_j) * np.sqrt(o_eo_i)) * np.eye(len(o_eo_j), len(o_eo_j[0])) / beo_i

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

    b_e_e = np.tile(
        np.resize(oi_input["bg_var_est"], (1, observed_count)), (estimated_count, 1)
    )
    b_e_o = np.tile(oi_input["bg_var_obs"], (estimated_count, 1))

    c_oe = oi_input["error_cov_fun"](distance_obs_vs_est) / oi_input["error_cov_fun"](0)

    b_ei = np.resize(c_oe * np.sqrt(b_e_e) / np.sqrt(b_e_o), (observed_count, 1))

    departures = oi_input["bg_departures"].reshape((1, len(oi_input["bg_departures"])))

    weights = np.linalg.solve(b_ij + o_ij, b_ei)
    weights = weights.reshape((1, len(weights)))

    oi_output["v_est"] = oi_input["bg_est"] + np.sum(weights * departures)
    oi_output["var_est"] = oi_input["bg_var_est"] * (
        1 - np.sum(b_ei[:, 0] * weights[0, :])
    )

    return oi_output, args


def loop_optimal_interpolation_stations(args: tuple[int, dict]) -> ndarray[Any, dtype[floating[Any]]]:
    """Apply optimal interpolation to a single validation site (station) for the selected time range.

    Parameters
    ----------
    args : tuple
        A tuple containing the station index and a dictionary with various information.
        The dictionary should include keys such as 'station_count', 'time_range', 'percentiles',
        'selected_flow_obs', 'selected_flow_sim', 'ratio_var_bg', 'ecf_fun', 'par_opt',
        'x_points', 'y_points', and 'drainage_area'.

    Returns
    -------
    ndarray
        A list containing the quantiles of the flow values for each percentile over the specified time range.
    """
    (station_index, args) = args

    filtered_dataset = args["filtered_dataset"]
    station_count = len(filtered_dataset["centroid_lat"].values)
    time_range = len(filtered_dataset["time"].values)
    selected_flow_obs = filtered_dataset["flow_obs"].values
    selected_flow_sim = filtered_dataset["flow_sim"].values
    drainage_area = filtered_dataset["drainage_area"].values

    percentiles = args["percentiles"]
    ratio_var_bg = args["ratio_var_bg"]
    ecf_fun = args["ecf_fun"]
    par_opt = args["par_opt"]
    x_points = args["x_points"]
    y_points = args["y_points"]

    # Get the number of stations from the dataset
    index = range(0, station_count)

    # Define the exit vectors for a single catchment at a time since we will
    # work on stations in parallel.
    flow_quantiles = np.array([np.empty(time_range) * np.nan] * len(percentiles))

    # Start cross-validation, getting indexes of the validation set.
    index_validation = station_index
    index_calibration = np.setdiff1d(index, index_validation)

    # Compute difference between the obs and sim log-transformed flows for the
    # calibration basins
    difference = (selected_flow_obs[:, index_calibration] - selected_flow_sim[:, index_calibration])
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

    # For each timestep, build the interpolator and apply to the cross-validation catchment.
    for j in range(time_range):
        # Need to skip days when no value exists for verification
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
    flow_obs: xr.Dataset,
    flow_sim: xr.Dataset,
    station_correspondence: xr.Dataset,
    crossvalidation_stations: list,
    ratio_var_bg,
    percentiles,
    iterations,
    parallelize,
    write_file,
):
    """Execute the main code, including setting constants to files, times, etc.

    Parameters
    ----------
    flow_obs : xr.Dataset
        Streamflow and catchment properties dataset for observed data.
    flow_sim : xr.Dataset
        Streamflow and catchment properties dataset for simulated data.
    station_correspondence: xr.Dataset
        Matching between the tag in the HYDROTEL simulated files and the observed station number for the obs dataset.
    crossvalidation_stations: list
        Observed hydrometric dataset stations to be used in the cross-validation step.
    ratio_var_bg : float
        Ratio for background variance.
    percentiles : list
        Desired percentiles for flow quantiles.
    iterations : int
        The number of iterations for the interpolation.
    parallelize : bool
        Flag indicating whether to parallelize the interpolation.
    write_file : str
        Name of the NetCDF file to be created.

    Returns
    -------
    list
        A list containing the flow quantiles for each desired percentile.
    """
    filtered_dataset = retrieve_data(flow_obs=flow_obs,
                                     flow_sim=flow_sim,
                                     station_correspondence=station_correspondence,
                                     crossvalidation_stations=crossvalidation_stations
                                     )

    # Project centroid lat/longs in a plane to get distances in km
    x, y = ma.latlon_to_xy(
        filtered_dataset["centroid_lat"].values,
        filtered_dataset["centroid_lon"].values,
        np.array([45] * len(filtered_dataset["centroid_lat"].values)),
        np.array([-70] * len(filtered_dataset["centroid_lat"].values)),
    )

    x_points, y_points = standardize_points_with_roots(
        x, y, len(filtered_dataset["station_id"].values), filtered_dataset["drainage_area"].values,
    )

    # create the weighting function parameters
    ecf_fun, par_opt = ecf_cc.correction(
        filtered_dataset["flow_obs"].values,
        filtered_dataset["flow_sim"].values,
        x_points,
        y_points,
        iteration_count=iterations,
    )

    # Create tuple with all info, required for the parallel processing.
    args = {
        "filtered_dataset": filtered_dataset,
        "ecf_fun": ecf_fun,
        "par_opt": par_opt,
        "x_points": x_points,
        "y_points": y_points,
        "ratio_var_bg": ratio_var_bg,
        "percentiles": percentiles,
        "iterations": iterations,
    }
    flow_quantiles = parallelize_operation(args, parallelize=parallelize)

    # Write results to netcdf file
    util.write_netcdf_flow_percentiles(
        write_file=write_file,
        station_id=crossvalidation_stations,
        lon=filtered_dataset["centroid_lon"].values,
        lat=filtered_dataset["centroid_lat"].values,
        drain_area=filtered_dataset["drainage_area"].values,
        time=filtered_dataset["time"].values,
        percentile=percentiles,
        discharge=flow_quantiles,
    )

    return flow_quantiles


def retrieve_data(flow_obs: xr.Dataset,
                  flow_sim: xr.Dataset,
                  station_correspondence: xr.Dataset,
                  crossvalidation_stations: list
                  ) -> xr.Dataset:
    """Retrieve data from files to populate the Optimal Interpolation (OI) algorithm.

    Parameters
    ----------
    flow_obs : xr.Dataset
        Streamflow and catchment properties dataset for observed data.
    flow_sim : xr.Dataset
        Streamflow and catchment properties dataset for simulated data.
    station_correspondence: xr.Dataset
        Matching between the tag in the HYDROTEL simulated files and the observed station number for the obs dataset.
    crossvalidation_stations: list
        Observed hydrometric dataset stations to be used in the cross-validation step.

    Returns
    -------
    xr.Dataset
        An xr.Dataset containing the retrieved and preprocessed data for the OI algorithm.
    """
    # Get some information from the input files
    time_range = len(flow_obs["time"].values)
    station_count = len(crossvalidation_stations)  # Number of validation stations

    # Preallocate some matrices
    centroid_lat = np.empty(station_count) * np.nan
    centroid_lon = np.empty(station_count) * np.nan
    drainage_area = np.empty(station_count) * np.nan
    selected_flow_obs = np.empty((time_range, station_count)) * np.nan
    selected_flow_sim = np.empty((time_range, station_count)) * np.nan

    # Loop over all validation stations
    for i in range(0, station_count):
        # Get the i^th station identification
        cv_station_id = crossvalidation_stations[i]

        # Get the station number from the obs database which has the same codification for station ids.
        index_correspondence = np.where(station_correspondence["station_id"] == cv_station_id)[0][0]
        station_code = station_correspondence["reach_id"][index_correspondence]

        # Search for data in the Qsim file
        index_in_sim = np.where(flow_sim["station_id"].values == station_code.data)[0]
        sup_sim = flow_sim["drainage_area"].values[index_in_sim]
        selected_flow_sim[:, i] = flow_sim["streamflow"].isel(station=index_in_sim) / sup_sim

        # Get the flows from the Qsim file
        index_in_obs = np.where(flow_obs["station_id"] == cv_station_id)[0]
        sup_obs = flow_obs["drainage_area"].values[index_in_obs]
        selected_flow_obs[:, i] = flow_obs["streamflow"].isel(station=index_in_obs) / sup_obs
        drainage_area[i] = sup_obs
        centroid_lon[i] = flow_obs["centroid_lon"][index_in_obs].values
        centroid_lat[i] = flow_obs["centroid_lat"][index_in_obs].values

    # Transformation log-débit pour l'interpolation
    selected_flow_obs = np.log(selected_flow_obs)
    selected_flow_sim = np.log(selected_flow_sim)

    filtered_dataset = xr.Dataset({'station_id': ('station', crossvalidation_stations),
                                   "centroid_lat": ("station", centroid_lat),
                                   "centroid_lon": ("station", centroid_lon),
                                   "flow_obs": (("time", "station"), selected_flow_obs),
                                   "flow_sim": (("time", "station"), selected_flow_sim),
                                   "time": ("time", flow_obs["time"].data),
                                   "drainage_area": ("station", drainage_area),
                                   })

    return filtered_dataset


def standardize_points_with_roots(
    x: np.ndarray, y: np.ndarray, station_count: int, drained_area: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize points with roots based on drainage area.

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates of the original points.
    y : np.ndarray
        Array of y-coordinates of the original points.
    station_count : int
        The number of stations or points.
    drained_area : np.ndarray
        Array of drainage areas corresponding to each station.

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


def parallelize_operation(args: dict, parallelize: bool = True) -> np.ndarray:
    """Run the interpolator on the cross-validation and manage parallelization.

    Parameters
    ----------
    args : dict
        A dictionary containing the necessary information for the interpolation.
    parallelize : bool
        Flag indicating whether to parallelize the interpolation.

    Returns
    -------
    np.ndarray
        An array containing the flow quantiles for each desired percentile.

    Notes
    -----
    New section for parallel computing. Several steps performed:
       1. Estimation of available computing power. Carried out by taking
          the number of threads available / 2 (to get the number of cores)
          then subtracting 2 to keep 2 available for other tasks.
       2. Starting a parallel computing pool
       3. Create an iterator (iterable) containing the data required
          by the new function, which loops on each station in leave-one-out
          validation. It's embarrassingly parallelized, so very useful.
       4. Run pool.map, which maps inputs (iterators) to the function.
       5. Collect the results and unzip the tuple returning from pool.map.
       6. Close the pool and return the parsed results.
    """
    filtered_dataset = args["filtered_dataset"]
    station_count = len(filtered_dataset["centroid_lat"].values)
    time_range = len(filtered_dataset["time"].values)
    percentiles = args["percentiles"]

    flow_quantiles = np.array([np.empty((time_range, station_count)) * np.nan] * len(percentiles))

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
