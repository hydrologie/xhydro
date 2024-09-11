"""Package containing the optimal interpolation functions."""

import os
from functools import partial
from multiprocessing import Pool
from typing import Any, Optional

import haversine
import numpy as np
import xarray as xr
from numpy import dtype, floating, ndarray
from scipy.stats import norm

import xhydro.optimal_interpolation.ECF_climate_correction as ecf_cc
import xhydro.optimal_interpolation.utilities as util

__all__ = [
    "execute_interpolation",
    "optimal_interpolation",
]


def optimal_interpolation(
    lat_obs: np.ndarray,
    lon_obs: np.ndarray,
    lat_est: np.ndarray,
    lon_est: np.ndarray,
    ecf: partial,
    bg_var_obs: np.ndarray,
    bg_var_est: np.ndarray,
    var_obs: np.ndarray,
    bg_departures: np.ndarray,
    bg_est: np.ndarray,
    precalcs: dict,
):
    """Perform optimal interpolation to estimate values at specified locations.

    Parameters
    ----------
    lat_obs : np.ndarray
        Vector of latitudes of the observation stations catchment centroids.
    lon_obs : np.ndarray
        Vector of longitudes of the observation stations catchment centroids.
    lat_est : np.ndarray
        Vector of latitudes of the estimation/simulation stations catchment centroids.
    lon_est : np.ndarray
        Vector of longitudes of the estimation/simulation stations catchment centroids.
    ecf : partial
        The function to use for the empirical distribution correction. It is a partial function from functools. The
        error covariance is a function of distance h, and this partial function represents this relationship.
    bg_var_obs : np.ndarray
        Background field variance at the observation stations (vector of size "observation stations").
    bg_var_est : np.ndarray
        Background field variance at estimation sites (vector of size "estimation stations").
    var_obs : np.ndarray
        Observation variance at observation sites (vector of size "observation stations").
    bg_departures : np.ndarray
        Difference between observation and background field at observation sites (vector of size "observation
        stations").
    bg_est : np.ndarray
        Background field values at estimation sites (vector of size "estimation stations").
    precalcs : dict
        Additional arguments and state information for the interpolation process, to accelerate calculations between
        timesteps.

    Returns
    -------
    v_est : float
        Estimated values at the estimation sites (vector of size "estimation stations").
    var_est : float
        Estimated variance at the estimation sites (vector of size "estimation stations").
    precalcs : dict
        Additional arguments and state information for the interpolation process, to accelerate calculations between
        timesteps. This variable returns the pre-calcualted distance matrices.
    """
    # Number of simulation stations
    estimated_count = len(bg_est)

    # Number of observed stations
    observed_count = len(bg_var_obs)

    # This block's entire purpose is to skip calculating distances when not necessary. If the input points and the
    # simulation points are the same as the previous time step (same NaN positions, so same stations contributing), then
    # simply load the distance matrix that was computed at the previous timestep. If it did change, we need to
    # recompute, and save it for the next iteration.
    # TODO: Check to see if xESMG reuse_weights could be useful here.
    cond = False
    if isinstance(precalcs, dict):
        if "lat_obs" in precalcs:
            cond = (
                np.array_equal(precalcs["lat_obs"], lat_obs)
                and np.array_equal(precalcs["lon_obs"], lon_obs)
                and np.array_equal(precalcs["lat_est"], lat_est)
                and np.array_equal(precalcs["lon_est"], lon_est)
            )

    # Depending on the status of the "cond" flag, either use the precalculated values or recalculate them.
    if cond:
        distance_obs_vs_obs = precalcs["distance_obs_obs"]
        distance_obs_vs_est = precalcs["distance_obs_vs_est"]
    else:
        # Not computed, so calculate and update the precalcs dict for later usage.
        observation_latlong = list(zip(lat_obs, lon_obs))
        distance_obs_vs_obs = haversine.haversine_vector(
            observation_latlong, observation_latlong, comb=True
        )

        # Also recompute the distance matrix between observation and estimation sites.
        simulation_latlong = list(zip(lat_est, lon_est))
        distance_obs_vs_est = haversine.haversine_vector(
            observation_latlong, simulation_latlong, comb=True
        )

        # And store for later
        precalcs["distance_obs_obs"] = distance_obs_vs_obs
        precalcs["lat_obs"] = lat_obs
        precalcs["lon_obs"] = lon_obs
        precalcs["distance_obs_vs_est"] = distance_obs_vs_est
        precalcs["lat_est"] = lat_est
        precalcs["lon_est"] = lon_est

    # Start doing the actual optimal interpolation math. "b" = background field variables; "o" = observations.
    covariance_obs_vs_obs = ecf(distance_obs_vs_obs) / ecf(0)

    # Background error at observation site
    beo_j = np.tile(bg_var_obs, (observed_count, 1))
    beo_i = beo_j.T

    # Background error covariance matrix at observation site
    b_ij = covariance_obs_vs_obs * np.sqrt(beo_j) / np.sqrt(beo_i)

    # Observation error at observation site
    o_eo_j = np.tile(var_obs, (observed_count, 1))
    o_eo_i = o_eo_j.T

    # Observation error covariance matrix
    o_ij = (np.sqrt(o_eo_j) * np.sqrt(o_eo_i)) * np.eye(len(o_eo_j)) / beo_i

    # Background error at estimation site
    b_e_e = np.tile(np.resize(bg_var_est, (1, observed_count)), (estimated_count, 1))
    b_e_o = np.tile(bg_var_obs, (estimated_count, 1))

    # Data to estimation site covariance
    c_oe = ecf(distance_obs_vs_est) / ecf(0)

    # Background error covariance matrix at estimation site
    b_ei = c_oe * (np.sqrt(b_e_e) / np.sqrt(b_e_o))

    # Matrix "I" for departures.
    departures = np.tile(bg_departures[:, np.newaxis], (1, estimated_count))

    # Get weights for each contributing station according to their covariance.
    weights = np.linalg.solve(b_ij + o_ij, b_ei.T)

    # Get value estimation by adding the background estimation and adding the weighted average of expected departures.
    v_est = bg_est + np.sum(weights * departures, axis=0)

    # Variance analysis for each estimated station based on weighted covariance.
    weighted_b_ei = np.diagonal(np.matmul(b_ei, weights))
    var_est = bg_var_est * (1 - weighted_b_ei)

    return v_est, var_est, precalcs


def loop_optimal_interpolation_stations_cross_validation(
    args: tuple[int, dict]
) -> ndarray[Any, dtype[floating[Any]]]:
    """Apply optimal interpolation to a single validation site (station) for the selected time range.

    Parameters
    ----------
    args : tuple
        A tuple containing the station index and a dictionary with various information.
        The dictionary should include the following keys and associated variables:
        - filtered_dataset: The dataset containing observed and simulated streamflow along with catchment sizes and
        centroid locations. Used to calculate deviations between observation stations and the simulated values at those
        locations.
        - full_background_dataset: The dataset containing the simulated stations where we want to apply the optimal
        interpolation results. This is the full background field, and is usually the complete simulated streamflow
        domain.
        - percentiles: The percentiles that we want to extract from the optimal interpolation at each station and time-
        step.
        - ratio_var_bg: The ratio of the observation variance to that of the background field (estimated).
        - ecf_fun: The partial function related to the Error Covariance Function model that estimates error covariance
        as a function of distance.
        - par_opt: The optimized parameters for the ecf_fun partial function.

    Returns
    -------
    ndarray
        A list containing the quantiles of the flow values for each percentile over the specified time range.
    """
    # TODO: Change variables that are input here, instead of dict pass more explicit. Need to find a way around the
    #  parallel comuputation
    (station_index, args) = args

    # Process data from the observations/simulations corresponding dataset
    filtered_dataset = args["filtered_dataset"]
    station_count = len(filtered_dataset["centroid_lat"].values)
    time_range = len(filtered_dataset["time"].values)
    selected_flow_obs = filtered_dataset["qobs"].values
    selected_flow_sim = filtered_dataset["qsim"].values
    drainage_area = filtered_dataset["drainage_area"].values
    centroid_lat_obs = filtered_dataset["centroid_lat"]
    centroid_lon_obs = filtered_dataset["centroid_lon"]

    # Process data from the background simulation field (complete simulation field)
    full_background_dataset = args["full_background_dataset"]
    centroid_lat_sim = full_background_dataset["centroid_lat"]
    centroid_lon_sim = full_background_dataset["centroid_lon"]

    # General hyperparameters for the optimal interpolation algorithm.
    percentiles = args["percentiles"]
    ratio_var_bg = args["ratio_var_bg"]
    ecf_fun = args["ecf_fun"]
    par_opt = args["par_opt"]

    # Get the number of stations from the dataset
    index = np.array([range(0, station_count)])

    # Define the vector of flow quantiles. We compute one value per time step and per percentile as requested by user.
    flow_quantiles = np.array([np.empty(time_range) * np.nan] * len(percentiles))

    # Start cross-validation, getting indexes of the validation set. If there is no cross-validation, all stations are
    # used and station_index is set to None.
    index_validation = np.array([station_index])

    # Necessary to handle the singleton dimensions of using a single station.
    index_validation = np.ravel(index_validation)

    index_calibration = np.setdiff1d(index, index_validation)

    # Compute difference between the obs and sim log-transformed flows for the
    # calibration basins
    difference = (
        selected_flow_obs[:, index_calibration]
        - selected_flow_sim[:, index_calibration]
    )

    vsim_at_est = selected_flow_sim[:, index_validation]

    # Object with the arguments to the OI that is passed along at each time step for calculations from previous steps
    # and updated to save computation time. Starts empty but will be updated for each time range later.
    precalcs = {}

    # For each timestep, build the interpolator and apply to the validation catchments.
    for j in range(time_range):

        val = difference[j, :]
        idx = ~np.isnan(val)

        # Apply the interpolator and get outputs
        v_est, var_est, precalcs = optimal_interpolation(
            lat_obs=centroid_lat_obs[index_calibration[idx]].values,
            lon_obs=centroid_lon_obs[index_calibration[idx]].values,
            lat_est=centroid_lat_sim[index_validation].values,
            lon_est=centroid_lon_sim[index_validation].values,
            ecf=partial(ecf_fun, par=par_opt),
            bg_var_obs=np.ones(idx.sum()),
            bg_var_est=np.ones(len(index_validation)),
            var_obs=np.ones(idx.sum()) * ratio_var_bg,
            bg_departures=difference[j, idx],
            bg_est=vsim_at_est[j, :],
            precalcs=precalcs,
        )

        # Get variance properties
        var_bg = np.var(difference[j, idx])
        var_est = var_est * var_bg

        # Get the percentile values for each desired percentile.
        vals = norm.ppf(
            np.array(percentiles) / 100.0, loc=v_est, scale=np.sqrt(var_est)
        )

        # Get the values in real units and scale according to drainage area
        vals = np.exp(vals) * drainage_area[station_index]
        for k in range(0, len(percentiles)):
            flow_quantiles[k][j] = vals[k]

    # return the flow quantiles as desired.
    return flow_quantiles


def optimal_interpolation_operational_control(
    filtered_dataset: xr.Dataset,
    full_background_dataset: xr.Dataset,
    percentiles: list,
    ratio_var_bg: float,
    ecf_fun: partial,
    par_opt: list,
) -> ndarray[Any, dtype[floating[Any]]]:
    """Apply optimal interpolation to a single validation site (station) for the selected time range.

    Parameters
    ----------
    filtered_dataset : xr.Dataset
        The dataset containing observed and simulated streamflow along with catchment sizes and centroid locations. Used
        to calculate deviations between observation stations and the simulated values at those locations.
    full_background_dataset : xr.Dataset
        The dataset containing the simulated stations where we want to apply the optimal interpolation results. This is
        the full background field, and is usually the complete simulated streamflow domain.
    percentiles : array-like
        The percentiles that we want to extract from the optimal interpolation at each station and timestep.
    ratio_var_bg : float
        The ratio of the observation variance to that of the background field (estimated).
    ecf_fun : partial
        The partial function related to the Error Covariance Function model that estimates error covariance as a
        function of distance.
    par_opt : array-like
        The optimized parameters for the ecf_fun partial function.

    Returns
    -------
    ndarray
        A list containing the quantiles of the flow values for each percentile over the specified time range.
    """
    # Process data from the observations/simulations corresponding dataset

    time_range = len(filtered_dataset["time"].values)
    selected_flow_obs = filtered_dataset["qobs"].values
    selected_flow_sim = filtered_dataset["qsim"].values
    centroid_lat_obs = filtered_dataset["centroid_lat"]
    centroid_lon_obs = filtered_dataset["centroid_lon"]

    # Process data from the background simulation field (complete simulation field)
    station_count = len(full_background_dataset["centroid_lat"].values)
    centroid_lat_sim = full_background_dataset["centroid_lat"]
    centroid_lon_sim = full_background_dataset["centroid_lon"]

    # Define the vector of flow quantiles. We compute one value per time step and per percentile as requested by user.
    flow_quantiles = np.array(
        [np.empty((time_range, station_count)) * np.nan] * len(percentiles)
    )

    # Compute difference between the obs and sim log-transformed flows for the calibration basins
    difference = selected_flow_obs - selected_flow_sim

    # Get the simulated values at the estimated points, which also includes the observation locations.
    vsim_at_est = full_background_dataset["qsim"].values

    # Object with the arguments to the OI that is passed along at each time step for calculations from previous steps
    # and updated to save computation time. Starts empty but will be updated for each time range later.
    precalcs = {}

    # For each timestep, build the interpolator and apply to the validation catchments.
    for j in range(time_range):

        val = difference[j, :]
        idx = ~np.isnan(val)

        # Apply the interpolator and get outputs
        v_est, var_est, precalcs = optimal_interpolation(
            lat_obs=centroid_lat_obs.values[idx],
            lon_obs=centroid_lon_obs.values[idx],
            lat_est=centroid_lat_sim.values,
            lon_est=centroid_lon_sim.values,
            ecf=partial(ecf_fun, par=par_opt),
            bg_var_obs=np.ones(idx.sum()),
            bg_var_est=np.ones(len(centroid_lon_sim)),
            var_obs=np.ones(idx.sum()) * ratio_var_bg,
            bg_departures=difference[j, idx],
            bg_est=vsim_at_est[j, :],
            precalcs=precalcs,
        )

        # Get variance properties
        var_bg = np.var(difference[j, idx])
        var_est = var_est * var_bg

        # For all stations, we need to compute the percentiles and un-log-transform the log transformation of flow.
        for stat in range(0, len(v_est)):
            # Get the percentile values for each desired percentile.
            vals = norm.ppf(
                np.array(percentiles) / 100.0,
                loc=v_est[stat],
                scale=np.sqrt(var_est[stat]),
            )
            # Get the values in real units and scale according to drainage area
            vals = np.exp(vals) * full_background_dataset["drainage_area"].values[stat]

            for k in range(0, len(percentiles)):
                flow_quantiles[:, j, stat] = vals[k]

    # return the flow quantiles as desired.
    return flow_quantiles


def execute_interpolation(
    qobs: xr.Dataset,
    qsim: xr.Dataset,
    station_correspondence: xr.Dataset,
    observation_stations: list,
    ratio_var_bg: float = 0.15,
    percentiles: list[float] | None = None,
    variogram_bins: int = 10,
    parallelize: bool = False,
    max_cores: int = 1,
    leave_one_out_cv: bool = False,
    form: int = 3,
    hmax_divider: float = 2.0,
    p1_bnds: list | None = None,
    hmax_mult_range_bnds: list | None = None,
):
    """Run the interpolation algorithm for leave-one-out cross-validation or operational use.

    Parameters
    ----------
    qobs : xr.Dataset
        Streamflow and catchment properties dataset for observed data.
    qsim : xr.Dataset
        Streamflow and catchment properties dataset for simulated data.
    station_correspondence : xr.Dataset
        Correspondence between the tag in the simulated files and the observed station number for the obs dataset.
    observation_stations : list
        Observed hydrometric dataset stations to be used in the ECF function building and optimal interpolation
        application step.
    ratio_var_bg : float
        Ratio for background variance (default is 0.15).
    percentiles : list(float), optional
        List of percentiles to analyze (default is [25.0, 50.0, 75.0, 100.0]).
    variogram_bins : int, optional
        Number of bins to split the data to fit the semi-variogram for the ECF. Defaults to 10.
    parallelize : bool
        Execute the profiler in parallel or in series (default is False).
    max_cores : int
        Maximum number of cores to use for parallel processing.
    leave_one_out_cv : bool
        Flag to determine if the code should be run in leave-one-out cross-validation (True) or should be applied
        operationally (False).
    form : int
        The form of the ECF equation to use (1, 2, 3 or 4. See documentation).
    hmax_divider : float
        Maximum distance for binning is set as hmax_divider times the maximum distance in the input data. Defaults to 2.
    p1_bnds : list, optional
        The lower and upper bounds of the parameters for the first parameter of the ECF equation for variogram fitting.
        Defaults to [0.95, 1].
    hmax_mult_range_bnds : list, optional
        The lower and upper bounds of the parameters for the second parameter of the ECF equation for variogram fitting.
        It is multiplied by "hmax", which is calculated to be the threshold limit for the variogram sill.
        Defaults to [0.05, 3].

    Returns
    -------
    flow_quantiles : list
        A list containing the flow quantiles for each desired percentile.
    ds : xr.Dataset
        An xarray dataset containing the flow quantiles and all the associated metadata.
    """
    # Set default flow percentiles to evaluate if none are provided.
    if percentiles is None:
        percentiles = [25.0, 50.0, 75.0, 99.0]

    # Get the filtered dataset (i.e. the one with the observations and corresponding background field simulations)
    filtered_dataset, full_background_dataset = retrieve_data(
        qobs=qobs,
        qsim=qsim,
        station_correspondence=station_correspondence,
        observation_stations=observation_stations,
    )

    if p1_bnds is None:
        p1_bnds = [0.95, 1]
    if hmax_mult_range_bnds is None:
        hmax_mult_range_bnds = [0.05, 3]

    # create the weighting function parameters using climatological errors (i.e. over many timesteps)
    ecf_fun, par_opt = ecf_cc.correction(
        da_qobs=filtered_dataset["qobs"],
        da_qsim=filtered_dataset["qsim"],
        centroid_lon_obs=filtered_dataset["centroid_lon"].values,
        centroid_lat_obs=filtered_dataset["centroid_lat"].values,
        variogram_bins=variogram_bins,
        form=form,
        hmax_divider=hmax_divider,
        p1_bnds=p1_bnds,
        hmax_mult_range_bnds=hmax_mult_range_bnds,
    )

    # If the user wants to do leave-one-out cross-validation, then the Qsims should be the same as Qobs file as we need
    # data at the same stations.
    if leave_one_out_cv:
        flow_quantiles = run_leave_one_out_cross_validation(
            filtered_dataset=filtered_dataset,
            full_background_dataset=full_background_dataset,
            ecf_fun=ecf_fun,
            par_opt=par_opt,
            ratio_var_bg=ratio_var_bg,
            percentiles=percentiles,
            parallelize=parallelize,
            max_cores=max_cores,
        )

        # TODO: Update this for dask parallelism, xscen and other optimizations regarding output dataset.
        # Write results to netcdf file
        ds = util.prepare_flow_percentiles_dataset(
            station_id=observation_stations,
            lon=filtered_dataset["centroid_lon"].values,
            lat=filtered_dataset["centroid_lat"].values,
            drain_area=filtered_dataset["drainage_area"].values,
            time=filtered_dataset["time"].values,
            percentile=percentiles,
            discharge=flow_quantiles,
        )

    # Run but not in leave-one-out cross-validation. Just run and apply on the qsim dataset to improve results
    else:
        flow_quantiles = optimal_interpolation_operational_control(
            filtered_dataset=filtered_dataset,
            full_background_dataset=full_background_dataset,
            percentiles=percentiles,
            ratio_var_bg=ratio_var_bg,
            ecf_fun=ecf_fun,
            par_opt=par_opt,
        )

        # Write results to netcdf file
        ds = util.prepare_flow_percentiles_dataset(
            station_id=full_background_dataset["station_id"].data,
            lon=full_background_dataset["centroid_lon"].values,
            lat=full_background_dataset["centroid_lat"].values,
            drain_area=full_background_dataset["drainage_area"].values,
            time=full_background_dataset["time"].values,
            percentile=percentiles,
            discharge=flow_quantiles,
        )

    return ds


def retrieve_data(
    qobs: xr.Dataset,
    qsim: xr.Dataset,
    station_correspondence: xr.Dataset,
    observation_stations: list,
) -> xr.Dataset:
    """Retrieve data from files to populate the Optimal Interpolation (OI) algorithm.

    Parameters
    ----------
    qobs : xr.Dataset
        Streamflow and catchment properties dataset for observed data.
    qsim : xr.Dataset
        Streamflow and catchment properties dataset for simulated data.
    station_correspondence : xr.Dataset
        Matching between the tag in the HYDROTEL simulated files and the observed station number for the obs dataset.
    observation_stations : list
        Observed hydrometric dataset stations to be used in the optimal interpolation step, for contributing to the
        generation of the error field.

    Returns
    -------
    filtered_dataset : xr.Dataset
        An xr.Dataset containing the retrieved and preprocessed data for the OI algorithm. Includes the corresponding
        datasets between the observation stations and the corresponding simulation stations, so includes only a
        reordered subset of the "full_background_dataset" simulation stations.
    full_background_dataset : xr.Dataset
        The dataset containing all the data (including positions and drainage areas of subcatchments) of the background
        field.
    """
    # Get some information from the input files
    if "time" in qobs.dims:
        time_range = len(qobs["time"].values)
    else:
        time_range = 1

    # These should probably be the stations only, not cross-validation, but all observed. For cross-validation, we
    # should then simply remove the ones we don't need.
    station_count = len(observation_stations)  # Number of validation stations

    # Preallocate some matrices. The "selected" variables are those that correspond between obs and sim stations.
    centroid_lat = np.empty(station_count) * np.nan
    centroid_lon = np.empty(station_count) * np.nan
    drainage_area = np.empty(station_count) * np.nan
    selected_flow_obs = np.empty((time_range, station_count)) * np.nan
    selected_flow_sim = np.empty((time_range, station_count)) * np.nan

    # Loop over all the observation stations and get the data from the correct simulation in the background field.
    for i in range(0, station_count):
        # Get the i^th observation station identification
        cv_station_id = observation_stations[i]

        # Get the station number from the obs database which has the same codification for station ids.
        index_correspondence = np.where(
            station_correspondence["station_id"] == cv_station_id
        )[0][0]
        station_code = station_correspondence["reach_id"][index_correspondence]

        # Search for data in the Qsim file
        index_in_sim = np.where(qsim["station_id"].values == station_code.data)[0]
        sup_sim = qsim["drainage_area"].values[index_in_sim]
        selected_flow_sim[:, i] = (
            qsim["streamflow"].isel(station=index_in_sim) / sup_sim
        )

        # Get the flows from the Qsim file
        index_in_obs = np.where(qobs["station_id"] == cv_station_id)[0]
        sup_obs = qobs["drainage_area"].values[index_in_obs]
        selected_flow_obs[:, i] = (
            qobs["streamflow"].isel(station=index_in_obs) / sup_obs
        )
        drainage_area[i] = sup_obs
        centroid_lon[i] = qobs["centroid_lon"][index_in_obs].values
        centroid_lat[i] = qobs["centroid_lat"][index_in_obs].values

    # Log-streamflow transformation for the interpolation
    selected_flow_obs = np.log(selected_flow_obs)
    selected_flow_sim = np.log(selected_flow_sim)

    if (time_range == 1) and (len(selected_flow_obs) == 1):
        if "time" in qobs.dims:
            time = qobs["time"].values[np.newaxis]
        else:
            time = np.array([np.nan])
    else:
        time = qobs["time"].values

    filtered_dataset = xr.Dataset(
        {
            "station_id": ("station", observation_stations),
            "centroid_lat": ("station", centroid_lat),
            "centroid_lon": ("station", centroid_lon),
            "qobs": (("time", "station"), selected_flow_obs),
            "qsim": (("time", "station"), selected_flow_sim),
            "time": ("time", time),
            "drainage_area": ("station", drainage_area),
        }
    )

    # Now we also need to make a new dataset that contains all the simulation stations.
    all_drainage_area = qsim["drainage_area"].values
    all_flow_sim = np.empty(qsim["streamflow"].shape)

    if len(all_flow_sim.shape) == 1:
        all_flow_sim = all_flow_sim[:, np.newaxis]

    for j in range(0, len(all_drainage_area)):
        all_flow_sim[j, :] = np.log(
            qsim["streamflow"].isel(station=j).values / all_drainage_area[j]
        )
    centroid_lat = qsim["lat"].values
    centroid_lon = qsim["lon"].values

    # Create the dataset for the complete background field of model simulations
    full_background_dataset = xr.Dataset(
        {
            "station_id": ("station", qsim["station_id"].data),
            "centroid_lat": ("station", centroid_lat),
            "centroid_lon": ("station", centroid_lon),
            "qsim": (("time", "station"), all_flow_sim.T),
            "time": ("time", time),
            "drainage_area": ("station", all_drainage_area),
        }
    )

    return filtered_dataset, full_background_dataset


def run_leave_one_out_cross_validation(
    filtered_dataset: xr.Dataset,
    full_background_dataset: xr.Dataset,
    ecf_fun: partial,
    par_opt: list,
    ratio_var_bg: float,
    percentiles: list,
    parallelize: bool = False,
    max_cores: int = 1,
) -> np.ndarray:
    """Run the interpolator on the cross-validation and manage parallelization.

    Parameters
    ----------
    filtered_dataset : xr.Dataset
        Flow data from qobs and qsim aligned according to the validation stations required.
    full_background_dataset : xr.Dataset
        Flow data from all the simulations used in the background field. Also has other metadata such as drainage area,
        and lat/lon of catchment centroids.
    ecf_fun : partial
        The function to use for the empirical distribution correction.
    par_opt : dict
        Parameters for the ecf_fun function, calibrated beforehand.
    ratio_var_bg : float
        Ratio for background variance (default is 0.15).
    percentiles : list(float), optional
        List of percentiles to analyze (default is [25.0, 50.0, 75.0, 100.0]).
    parallelize : bool
        Flag indicating whether to parallelize the interpolation.
    max_cores : int
        Maximum number of cores to use for parallel processing.

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
    station_count = len(filtered_dataset["centroid_lat"].values)
    time_range = len(filtered_dataset["time"].values)

    flow_quantiles = np.array(
        [np.empty((time_range, station_count)) * np.nan] * len(percentiles)
    )

    # TODO : Try and find a way to make this more explicit instead of passing a dictionary. Currently need to create
    #  this to pass to the looping function due to the *p.map() for parallelism.
    args = dict(
        {
            "filtered_dataset": filtered_dataset,
            "full_background_dataset": full_background_dataset,
            "percentiles": percentiles,
            "ratio_var_bg": ratio_var_bg,
            "ecf_fun": ecf_fun,
            "par_opt": par_opt,
        }
    )

    # Parallel
    if parallelize:
        processes_count = os.cpu_count() / 2 - 1
        processes_count = min(processes_count, max_cores)

        p = Pool(int(processes_count))
        args_i = [(i, args) for i in range(0, station_count)]
        flow_quantiles_station = zip(
            *p.map(loop_optimal_interpolation_stations_cross_validation, args_i)
        )
        p.close()
        p.join()
        flow_quantiles_station = tuple(flow_quantiles_station)
        for k in range(0, len(percentiles)):
            flow_quantiles[k][:] = np.array(flow_quantiles_station[k]).T

    # Serial
    else:
        for i in range(0, station_count):
            flow_quantiles_station = (
                loop_optimal_interpolation_stations_cross_validation((i, args))
            )
            for k in range(0, len(percentiles)):
                flow_quantiles[k][:, i] = flow_quantiles_station[k][:]

    return flow_quantiles
