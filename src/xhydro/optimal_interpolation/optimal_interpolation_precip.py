"""Apply optimal interpolation of precipitation data over a background field."""

import datetime as dt
from functools import partial
from pathlib import Path

import haversine
import numpy as np
import pandas as pd
import xarray as xr
from numpy import ndarray
from scipy.optimize import minimize
from scipy.stats import norm
from xarray import Dataset

import xhydro.optimal_interpolation.oi_cross_validation as cv


def check_netcdf_layout_gridded(
    nc_path: str | Path,
    var_name: str = "tp",
    required_dims: tuple[str, str, str] = ("longitude", "latitude", "time"),
    required_coords: tuple[str, str, str] = ("time", "latitude", "longitude"),
):
    """
    Validate that a gridded NetCDF file contains coordinates, dimensions and variables required for the process.

    Parameters
    ----------
    nc_path : Union[str, Path]
        The path and name of the file that contains the stations (observations) data. Must be a netcdf file.
    var_name : str
        The name of the variable to be used for interpolation.
    required_dims : Tuple[str, str, str]
        The names of the longitude, latitude and time variables in the Dataset.
    required_coords : Tuple[str, str, str]
        The names of the coordinates in the dataset, corresponding to the longitude, latitude and time dimensions.

    Returns
    -------
    bool :
        True if valid, otherwise raises ValueError with a readable message.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise ValueError(f"File not found: {nc_path}")

    with xr.open_dataset(nc_path) as ds:
        errors: list[str] = []

        # --- Required dimensions exist ---
        errors.extend(f"Missing dimension: {d!r}" for d in required_dims if d not in ds.dims)

        # --- Required coordinates exist and are 1D on their own dim ---
        errors.extend(
            [f"Missing coordinate: {c!r}" for c in required_coords if c not in ds.coords]
            + [
                f"Coordinate {c!r} must be 1D with dims ({c!r},), got {tuple(ds[c].dims)}"
                for c in required_coords
                if c in ds.coords and tuple(ds[c].dims) != (c,)
            ]
        )

        # --- Required variable exists and has the right dims (any order) ---
        if var_name not in ds.data_vars:
            errors.append(f"Missing data variable: {var_name!r}")
        else:
            v_dims = tuple(ds[var_name].dims)
            required = set(required_dims)
            got = set(v_dims)

            if required != got:
                errors.append(f"Variable {var_name!r} must use exactly dims {required_dims} (any order), got {v_dims}")

        if errors:
            raise ValueError("NetCDF layout check failed:\n- " + "\n- ".join(errors))

    return True


def check_netcdf_layout_stations(
    nc_path: str | Path,
    time_dim: str = "time",
    required_station_dim: str = "watershed",
    required_precip_var: str = "precip",
    required_station_vars: tuple[str, ...] = ("latitude", "longitude", "altitude"),
) -> bool:
    """
    Validate that a NetCDF file for the stations data contains required information.

    Dimensions:
      - time
      - station ID

    Coordinates:
      - time (time)  [must exist as a coordinate, 1D along 'time']

    Data variables:
      - precip with dims being a permutation of (station, time) (order can differ)
      - latitude (station)
      - longitude (station)
      - altitude (station)
      - StationID_SEQ (station)

    Parameters
    ----------
    nc_path : str | Path
        Path to the the netcdf file containing the dataset to explore.
    time_dim : str
        Name of the time dimension in the Dataset. Most likely "time".
    required_station_dim : str
        Name of the station dimension.
    required_precip_var : str
        Name of the variable containing the precipitation values.
    required_station_vars : Tuple[str, str, str]
        A tuple containing the names of the variables describing information about
        stations.

    Returns
    -------
    bool
        True if valid, otherwise raises ValueError with a readable message.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise ValueError(f"File not found: {nc_path}")

    with xr.open_dataset(nc_path) as ds:
        errors: list[str] = []

        # --- Required dimensions ---
        errors.extend([f"Missing dimension: {d!r}" for d in (time_dim, required_station_dim) if d not in ds.dims])

        # --- Required coordinate 'time' ---
        errors.extend(
            [f"Missing coordinate: {time_dim!r}"]
            if time_dim not in ds.coords
            else []
            + (
                [f"Coordinate {time_dim!r} must be 1D with dims ({time_dim!r},), got {tuple(ds[time_dim].dims)}"]
                if tuple(ds[time_dim].dims) != (time_dim,)
                else []
            )
        )

        # --- precip variable dims: any order of (watershed, time) ---
        if required_precip_var not in ds.data_vars:
            errors.append(f"Missing data variable: {required_precip_var!r}")
        else:
            v_dims = tuple(ds[required_precip_var].dims)
            required = {required_station_dim, time_dim}
            got = set(v_dims)
            if required != got:
                errors.append(
                    f"Variable {required_precip_var!r} must use exactly dims ({required_station_dim!r}, {time_dim!r}) (any order), got {v_dims}"
                )

        # --- watershed-metadata variables: must be 1D on watershed ---
        for v in required_station_vars:
            if v not in ds.data_vars:
                errors.append(f"Missing data variable: {v!r}")
            else:
                vdims = tuple(ds[v].dims)
                if vdims != (required_station_dim,):
                    errors.append(f"Variable {v!r} must be 1D with dims ({required_station_dim!r},), got {vdims}")

        if errors:
            raise ValueError("NetCDF layout check failed:\n- " + "\n- ".join(errors))

        return True


def general_ecf(h: np.ndarray, par: list | np.ndarray, form: int):
    """
    Define the form of the Error Covariance Function (ECF) equations.

    Parameters
    ----------
    h : float or array
        The distance or distances at which to evaluate the ECF.
    par : list or array-like
        Parameters for the ECF equation.
    form : int
        The form of the ECF equation to use (1, 2, 3 or 4). See :py:func:`correction` for details.

    Returns
    -------
    float or array :
        The calculated ECF values based on the specified form.
    """
    if form == 1:  # From Lachance-Cloutier et al. 2017.
        return par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
    elif form == 2:
        return par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
    elif form == 3:
        return par[0] * np.exp(-h / par[1])
    elif form == 4:
        return par[0] * np.exp(-(h ** par[1]) / par[0])
    else:
        raise (Exception("Unknown form of the Error Covariance Function (ECF). Please use values from 1 to 4."))


def find_position_of_observation_in_vectorized_grid(obslat: float, obslon: float, gridlat: list[float], gridlon: list[float]) -> int:
    """
    Find the gridpoint in a 2D grid that corresponds to a unique observation station location.

    Parameters
    ----------
    obslat : float
        Latitude of the observation station that needs to be found in the gridded product.
    obslon : float
        Longitude of the observation station that needs to be found in the gridded product.
    gridlat : array-like
        Latitude of all the gridpoints we need to search.
    gridlon : array-like
        Longitude of all the gridpoints we need to search.

    Returns
    -------
    np.array
        The position of the gridpoint that corresponds to the observation station location.
    """
    pos = np.argwhere((gridlon == obslon) & (gridlat == obslat))
    return pos[0][0]


def optimal_interpolation(
    lat_obs: np.ndarray,
    lon_obs: np.ndarray,
    lat_est: np.ndarray,
    lon_est: np.ndarray,
    ecf: partial,
    bg_var_obs: np.ndarray,
    bg_var_est: np.ndarray,
    var_obs: np.ndarray,
    bg_departures: xr.DataArray,
    bg_est: np.ndarray,
    precalcs: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Perform optimal interpolation to estimate values at specified locations of a background field.

    Parameters
    ----------
    lat_obs : np.ndarray
        Vector of latitudes of the observation stations.
    lon_obs : np.ndarray
        Vector of longitudes of the observation stations.
    lat_est : np.ndarray
        Vector of latitudes of the estimation/simulation stations to be estimated in the background field.
    lon_est : np.ndarray
        Vector of longitudes of the estimation/simulation stations to be estimated in the background field.
    ecf : partial
        The function to use for the empirical distribution correction. It is a partial function from functools.
        The error covariance is a function of distance h, and this partial function represents this relationship.
    bg_var_obs : np.ndarray
        Background field variance at the observation stations (vector of size "observation stations").
    bg_var_est : np.ndarray
        Background field variance at estimation sites (vector of size "estimation stations").
    var_obs : np.ndarray
        Observation variance at observation sites (vector of size "observation stations").
    bg_departures : DataArray
        Difference between observation and background field at observation sites (vector of size "observation stations").
    bg_est : np.ndarray
        Background field values at estimation sites (vector of size "estimation stations").
    precalcs : dict
        Additional arguments and state information for the interpolation process, to accelerate calculations between timesteps.

    Returns
    -------
    v_est : np.ndarray
        Estimated values at the estimation sites (vector of size "estimation stations").
    var_est : np.ndarray
        Estimated variance at the estimation sites (vector of size "estimation stations").
    precalcs : dict
        Additional arguments and state information for the interpolation process, to accelerate calculations between timesteps.
        This variable returns the pre-calculated distance matrices.
    """
    # Number of simulation stations
    estimated_count = len(bg_est)

    # Number of observed stations
    observed_count = len(bg_var_obs)

    # This block's entire purpose is to skip calculating distances when not necessary. If the input points and the
    # simulation points are the same as the previous time step (same NaN positions, so same stations contributing), then
    # simply load the distance matrix that was computed at the previous timestep. If it did change, we need to
    # recompute, and save it for the next iteration.
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
        observation_latlong = list(zip(lat_obs, lon_obs, strict=True))
        distance_obs_vs_obs = haversine.haversine_vector(observation_latlong, observation_latlong, comb=True)

        # Also recompute the distance matrix between observation and estimation sites.
        simulation_latlong = list(zip(lat_est, lon_est, strict=True))
        distance_obs_vs_est = haversine.haversine_vector(observation_latlong, simulation_latlong, comb=True)

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


def optimal_interpolation_operational_control(
    estimated_precip_reshaped,
    difference,
    lat_obs,
    lon_obs,
    lat_est,
    lon_est,
    percentiles: list[float] | None,
    ratio_var_bg: float,
    ecf_fun: partial,
    par_opt: list,
):
    """
    Apply optimal interpolation to the entire domain for the selected time range.

    Parameters
    ----------
    estimated_precip_reshaped : xr.Dataset
        The dataset containing observed and simulated precipitation along with locations. Used
        to calculate deviations between observation stations and the simulated (background field) values at those
        locations.
    difference : xr.Dataset
        The dataset containing the simulated stations where we want to apply the optimal interpolation results. This is
        the full background field, and is usually the complete simulated precipitation domain.
    lat_obs : np.ndarray
        Vector of latitudes of the observation stations.
    lon_obs : np.ndarray
        Vector of longitudes of the observation stations.
    lat_est : np.ndarray
        Vector of latitudes of the estimation/simulation stations to be estimated in the background field.
    lon_est : np.ndarray
        Vector of longitudes of the estimation/simulation stations to be estimated in the background field.
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
        An array containing the quantiles of the precipitation values for each percentile over the specified time range.
    """
    # Set a default flow percentile to evaluate if none are provided. Median is used as the most likely to be useful.
    if percentiles is None:
        percentiles = [50.0]

    # Process data from the observations/simulations corresponding dataset
    time_range = difference.shape[0]

    # Process data from the background simulation field (complete background field)
    station_count = len(lat_est)

    # Get the simulated values at the estimated points, which also includes the observation locations.
    vsim_at_est = estimated_precip_reshaped

    # Define the vector of precipitation quantiles. We compute one value per time step and per percentile as requested
    # by user.
    precipitation_quantiles = np.array([np.empty((time_range, station_count)) * np.nan] * len(percentiles))

    # Object with the arguments to the OI that is passed along at each time step for calculations from previous steps
    # and updated to save computation time. Starts empty but will be updated for each time range later.
    precalcs = {}

    # For each timestep, build the interpolator and apply to the estimated locations (i.e. all background field points).
    for j in range(time_range):
        print("Interpolating for timestep: " + str(j))

        # Get the values for the current day and identify which are not NaN.
        val = difference[j, :]
        idx = ~np.isnan(val)

        # Apply the interpolator and get outputs
        v_est, var_est, precalcs = optimal_interpolation(
            lat_obs=lat_obs[idx],
            lon_obs=lon_obs[idx],
            lat_est=lat_est,
            lon_est=lon_est,
            ecf=partial(ecf_fun, par=par_opt),
            bg_var_obs=np.ones(idx.sum()),
            bg_var_est=np.ones(len(lon_est)),
            var_obs=np.ones(idx.sum()) * ratio_var_bg,
            bg_departures=difference[j, idx],
            bg_est=vsim_at_est[j, :],
            precalcs=precalcs,
        )

        # Get variance properties
        var_bg = np.var(difference[j, idx])
        var_est = var_est * var_bg

        # Correct values below zero
        v_est[v_est < 0] = 0

        # For all stations, we need to compute the percentiles and un-log-transform the log transformation of
        # precipitation.
        for stat in range(0, len(v_est)):
            # Get the percentile values for each desired percentile.
            vals = norm.ppf(
                np.array(percentiles) / 100.0,
                loc=v_est[stat],
                scale=np.sqrt(var_est[stat]),
            )

            for k in range(0, len(percentiles)):
                precipitation_quantiles[:, j, stat] = vals[k]

    # return the precipitation quantiles as desired.
    return precipitation_quantiles


def prepare_initial_data(
    filename_stations: str,
    filename_gridded: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    frac_validation: float,
    do_cross_validation: bool,
    grid_resolution: float,
    var_name_gridded: str,
    var_name_stations: str,
    dims_gridded: tuple[str, str, str],
    dims_stations: str,
    coords_gridded: tuple[str, str, str],
    station_req_vars: tuple[str, str, str],
) -> tuple[ndarray, ndarray, list[float], list[float], list[float], list[float], list[float], list[float], ndarray, list[int], ndarray, Dataset]:
    """
    Perform the initial data processing steps to get data ready for the rest of the process according to user needs.

    Parameters
    ----------
    filename_stations : str
        The path and name of the file that contains the stations (observations) data. Must be a netcdf file.
    filename_gridded : str
        The path and name of the file that contains the estimated (background) data. Must be a netcdf file.
    start_time : dt.datetime
        The start date of the optimal interpolation period.
    end_time : dt.datetime
        The end date of the optimal interpolation period.
    frac_validation : float | None
        The fraction of stations to use for cross-validation, independent of the training data. Between 0 and 1.
    do_cross_validation : bool
        A flag to indicate that the user wants to perform a cross-validation step to assess the method's performance.
    grid_resolution : float
        Resolution of the gridded product, for rounding purposes.
    var_name_gridded : str
        Name of the meteo variable (i.e. precipitation) in the gridded dataset.
    var_name_stations : str
        Name of the meteo variable (i.e. precipitation) in the stations dataset.
    dims_gridded : Tuple[str, str, str]
        The three variable names that should be in the dataset, corresponding to latitude, longitude and time. For this
        package we are expecting "latitude", "longitude", and "time".
    dims_stations : str
        The name of the dimensions for the station data file. Should be "station" or "number" or some other integer-
        based ID.
    coords_gridded : Tuple[str, str, str]
        The coordinates of the gridded xr.Dataset that must be present, corresponding to latitude, longitude and time.
        For this package we are expecting "latitude", "longitude", and "time".
    station_req_vars : Tuple[str, str, str]
        The required variables in the station Dataset along with the precipitation values. These correspond to the
        latitude, longitude and altitude of stations.

    Returns
    -------
    departures_train: np.array
        Differences between observation stations and simulations at the observation station locations, for training
        stations.
    departures_valid: np.array
        Differences between observation stations and simulations at the observation station locations, for validation
        stations.
    lat_obs_train: array-like
        Latitude of the observation stations for the training dataset.
    lon_obs_train: array-like
        Longitude of the observation stations for the training dataset.
    lat_obs_valid: array-like
        Latitude of the observation stations for the validation dataset.
    lon_obs_valid: array-like
        Longitude of the observation stations for the validation dataset.
    lat_est: array-like
        Latitude of the estimated gridpoints (complete grid).
    lon_est: array-like
        Longitude of the estimated gridpoints (complete grid).
    estimated_precip_reshaped: np.array
        Estimated precipitation in a reshaped matrix such that it is compatible with the interpolator requirements.
    original_shape: list[int]
        Original shape of the estimated precipitation grid, to be able to plot it as a grid later.
    observed_precip_valid: np.array
        Observed precipitation for the validation set, used to compute evaluation metrics for cross-validation.
    ds_gridded: Dataset
        The original gridded dataset, loaded and subset for the correct time. Also used later for plotting in
        cross-validation.
    """
    # This first section processes the observation stations
    # Read the station data.
    check_netcdf_layout_stations(
        nc_path=filename_stations,
        time_dim="time",
        required_station_dim=dims_stations,
        required_precip_var=var_name_stations,
        required_station_vars=station_req_vars,
    )
    ds_stations_full = xr.open_dataset(filename_stations)
    ds_stations_full = ds_stations_full.sel(time=slice(start_time, end_time))

    # For cross validation, select a fraction of weather stations that we will keep for later.
    number_stations = ds_stations_full.latitude.shape[0]
    list_stations = np.arange(0, number_stations)

    # If we want to do cross-validation to evaluate the OI performance, we need to split into a training and a
    # validation set.
    # Shuffle the indices of the stations to select the training and validation stations at random
    np.random.shuffle(list_stations)
    valid_idx = list_stations[0 : round(number_stations * frac_validation)]

    # Make sure the user selected consistent options. We do not default to one or the other. The options must not
    # conflict.
    if (len(valid_idx) == 0) and do_cross_validation:
        raise (
            Exception(
                "Fraction of cross-validation stations is too low, no stations would be selected. Please "
                "increase the cross-validation fraction or set cross-validation to False."
            )
        )
    if (len(valid_idx) > 0) and not do_cross_validation:
        raise (
            Exception(
                "The fraction of cross-validation stations to use is not 0, but the user selected not to do "
                "cross-validation. Please set do_cross_validation to True or set fraction of cross validation "
                "stations to 0."
            )
        )

    # At this point, either we have:
    # a. Empty valid_idx and do_cross_validation = False
    # b. Some stations in valid_idx and do_cross_validation = True.
    # Therefore, validation variables will be empty if there is no cross-validation but the code will still run without
    # error.

    # Continue the process. Extract the training station indices.
    train_idx = list_stations[round(number_stations * frac_validation) :]

    # Save the IDs of the validation and training stations for future reference
    np.savetxt("./validation_stations.txt", valid_idx)
    np.savetxt("./training_stations.txt", train_idx)

    # Extract the training stations and valid stations in the 2 different datasets
    ds_stations_train = ds_stations_full.isel(watershed=train_idx)
    ds_stations_valid = ds_stations_full.isel(watershed=valid_idx)

    # prepare lat and long values for observations
    lat_obs_train = ds_stations_train.latitude.values
    lon_obs_train = ds_stations_train.longitude.values
    lat_obs_valid = ds_stations_valid.latitude.values
    lon_obs_valid = ds_stations_valid.longitude.values

    # Get the precipitation values from the station data
    observed_precip_train = ds_stations_train.precip.values
    observed_precip_valid = ds_stations_valid.precip.values

    # Process the gridded data in this section.
    # Read the background field data and ensure the format is correct and contains the desired variables.
    check_netcdf_layout_gridded(
        nc_path=filename_gridded,
        var_name=var_name_gridded,
        required_dims=dims_gridded,
        required_coords=coords_gridded,
    )

    ds_gridded = xr.open_dataset(filename_gridded)
    ds_gridded = ds_gridded.transpose("time", "longitude", "latitude")
    ds_gridded = ds_gridded.sel(time=slice(start_time, end_time))

    # Prepare lat and long values for background field/estimator
    lat_est = ds_gridded.latitude.values
    lon_est = ds_gridded.longitude.values

    # Prepare the latitudes and longitudes rounded to the nearest resolution unit to
    # eliminate any machine-precision errors. Will depend on the product!
    if grid_resolution == 0.1:
        lat_est = np.round(lat_est, 1)
        lon_est = np.round(lon_est, 1)
    elif grid_resolution == 0.25:
        lat_est = np.round(np.round(lat_est * 4, 0) / 4, 2)
        lon_est = np.round(np.round(lon_est * 4, 0) / 4, 2)
    else:
        ValueError("Grid resolutions are currently restricted to 0.1° or 0.25°.")

    # Make grid from latlong vectors
    lon_est, lat_est = np.meshgrid(lon_est, lat_est)

    # Transpose and extract data
    estimated_precip = ds_gridded.tp.transpose("latitude", "longitude", "time").values

    # Flatten the dataset to vector form
    original_shape = lat_est.shape
    lat_est = lat_est.flatten("C")
    lon_est = lon_est.flatten("C")

    # Still has 2D: all points in a line, all timesteps in columns.
    estimated_precip_reshaped = estimated_precip.reshape(original_shape[0] * original_shape[1], -1)

    # Now calculate departures (error) between the stations and the background field for each day and station.
    # Preallocate the departures vector, for corresponding observations and grid points
    departures_train = np.empty((len(lat_obs_train), observed_precip_train.shape[1]))
    departures_valid = np.empty((len(lat_obs_valid), observed_precip_valid.shape[1]))

    # Calculate departures for the training set
    for station in range(0, len(lat_obs_train)):
        pos = find_position_of_observation_in_vectorized_grid(lat_obs_train[station], lon_obs_train[station], lat_est, lon_est)
        departures_train[station, :] = observed_precip_train[station, :] - estimated_precip_reshaped[pos, :]

    # Calculate departures for the validation set
    for station in range(0, len(lat_obs_valid)):
        pos = find_position_of_observation_in_vectorized_grid(lat_obs_valid[station], lon_obs_valid[station], lat_est, lon_est)
        departures_valid[station, :] = observed_precip_valid[station, :] - estimated_precip_reshaped[pos, :]

    return (
        departures_train,
        departures_valid,
        lat_obs_train,
        lon_obs_train,
        lat_obs_valid,
        lon_obs_valid,
        lat_est,
        lon_est,
        estimated_precip_reshaped,
        original_shape,
        observed_precip_valid,
        ds_gridded,
    )


def correction(
    difference: np.ndarray,
    centroid_lon_obs: np.ndarray,
    centroid_lat_obs: np.ndarray,
    variogram_bins: int = 10,
    form: int = 1,
    hmax_divider: float = 2.0,
    p1_bnds: list | None = None,
    hmax_mult_range_bnds: list | None = None,
) -> tuple:
    """
    Perform correction on flow observations using optimal interpolation.

    Parameters
    ----------
    difference : np.ndarray
        A 2D array of differences for all stations over all time periods [nstat x time].
    centroid_lon_obs : np.ndarray
        Longitude vector of the catchment centroids for the observed stations.
    centroid_lat_obs : np.ndarray
        Latitude vector of the catchment centroids for the observed stations.
    variogram_bins : int, optional
        Number of bins to split the data to fit the semi-variogram for the ECF. Defaults to 10.
    form : int
        The form of the ECF equation to use (1, 2, 3 or 4. See Notes below).
    hmax_divider : float
        Maximum distance for binning is set as hmax_divider times the maximum distance in the input data. Defaults to 2.
    p1_bnds : list, optional
        The lower and upper bounds of the parameters for the first parameter of the ECF equation for variogram fitting.
        Defaults to [0.95, 1.0].
    hmax_mult_range_bnds : list, optional
        The lower and upper bounds of the parameters for the second parameter of the ECF equation for variogram fitting.
        It is multiplied by "hmax", which is calculated to be the threshold limit for the variogram sill.
        Defaults to [0.05, 3.0].

    Returns
    -------
    tuple
        A tuple containing the following:
        - ecf_fun: Partial function for the error covariance function.
        - par_opt: Optimized parameters for the interpolation.

    Notes
    -----
    The possible forms for the ecf function fitting are as follows:
        Form 1 (From Lachance-Cloutier et al. 2017; and Garand & Grassotti 1995) :
            ecf_fun = par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
        Form 2 (Gaussian form) :
            ecf_fun = par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
        Form 3 :
            ecf_fun = par[0] * np.exp(-h / par[1])
        Form 4 :
            ecf_fun = par[0] * np.exp(-(h ** par[1]) / par[0])
    """
    # Number of timesteps. If we have more than 1 timestep, we can use the mean climatological variogram later.
    difference = difference.T
    time_range = difference.shape[0]

    # Preallocate matrices for all timesteps for the heights of the histograms, the covariances and standard deviations
    # of the values within each bin of the histogram.
    heights = np.empty((time_range, variogram_bins)) * np.nan
    covariances = np.empty((time_range, variogram_bins)) * np.nan
    standard_deviations = np.empty((time_range, variogram_bins)) * np.nan

    # Ensure we have some bounds by default in case it is missed.
    if p1_bnds is None:
        p1_bnds = [0.95, 1]
    if hmax_mult_range_bnds is None:
        hmax_mult_range_bnds = [0.05, 3]

    # Pairwise distance between all observation stations.
    observation_latlong = list(zip(centroid_lat_obs, centroid_lon_obs, strict=True))
    distance = haversine.haversine_vector(observation_latlong, observation_latlong, comb=True)

    # If there is more than 1 time step, we need to do a climatological mean ECF.
    if time_range > 1:
        # For each timestep, we will compute the histogram bins and other data required to provide the data for the ECF
        # function optimizer.
        for i in range(time_range):
            # Check for NaN observed streamflow data points (if difference is NaN, necessarily obs is NaN. But in case a
            # sim also provides a NaN, then take the difference instead).
            is_nan = np.isnan(difference[i, :])

            # Calculate the error for this particular day and remove NaN days.
            day_diff = difference[i, ~is_nan]
            day_diff = day_diff + np.random.rand(day_diff.shape[0]) / np.power(10, 5)

            # If there are at least as many stations worth of data as there are required bins, we can compute the
            # histogram.
            if len(day_diff) >= variogram_bins:
                # Get the stations that did not have NaN observations. Since the matrix is 2D due to pairwise distances,
                # need to remove rows and columns of NaN-stations from the distance matrix distance_pc.
                distance_pc = np.delete(distance, is_nan, axis=0)
                distance_pc = np.delete(distance_pc, is_nan, axis=1)

                # Sanity check: length of distance_pc should be equal to day_diff.
                if len(day_diff) != distance_pc.shape[0]:
                    raise AssertionError("day_diff not equal to the size of distance_pc in histogram bin definition.")

                # Sort the data into bins and get their stats.
                h_b, cov_b, std_b, num_p = eval_covariance_bin(
                    distances=distance_pc,
                    values=day_diff,
                    hmax_divider=hmax_divider,
                    variogram_bins=variogram_bins,
                )

                # If there are at least "variogram_bins" number of bins, then add it to the results matrix
                if len(num_p[0]) >= variogram_bins:
                    heights[i, :] = h_b[0, 0 : variogram_bins + 1]
                    covariances[i, :] = cov_b[0, 0 : variogram_bins + 1]
                    standard_deviations[i, :] = std_b[0, 0 : variogram_bins + 1]

        # The histogram bins for each day have been calculated. Now is time to prepare the statistics overall for the
        # ECF function fitting for the semi-variogram for the number of days (i.e. weighted average of climatology).
        # This first function reformats the data according to the timestep and computes the weighted average histograms.
        distance, covariance, covariance_weights, valid_heights = initialize_stats_variables(
            heights, covariances, standard_deviations, variogram_bins
        )

        # And this second part does the binning of the histogram as was done before for all days.
        h_b, cov_b, std_b = calculate_ecf_stats(distance, covariance, covariance_weights, valid_heights)

    else:
        # Just compute the covariance bin as a one-shot deal
        h_b, cov_b, std_b, num_p = eval_covariance_bin(
            distances=distance,
            values=np.squeeze(difference),
            hmax_divider=hmax_divider,
            variogram_bins=variogram_bins,
        )

    # Calculate the real maximum distance for the covariance estimation.
    hmax = max(np.reshape(distance, (-1, 1))) / hmax_divider

    # This determines the shape of the fit that we want the optimizer to fit to the correlation variogram.
    ecf_fun = partial(general_ecf, form=form)

    # Weight according to the inverse of the variance of each bin and then normalize them
    weights = 1 / np.power(std_b, 2)
    weights = weights / np.sum(weights)

    # Define the objective function used for the ECF function training.
    def _rmse_func(par):
        # Compute the RMSE of the fit between the observations and variogram fit according to the ecf_fun chosen.
        return np.sqrt(np.mean(weights * np.power(ecf_fun(h=h_b, par=par) - cov_b, 2)))

    # Perform the training using the bounds for the parameters as passed by the users before.
    par_opt = minimize(
        _rmse_func,
        x0=[np.mean(cov_b), np.mean(h_b) / 3],
        bounds=(
            [p1_bnds[0], p1_bnds[1]],
            [hmax_mult_range_bnds[0] * hmax, hmax_mult_range_bnds[1] * hmax],
        ),
    )["x"]

    # Return the fitting function as determined by the user and the optimal calibrated parameter set.
    return ecf_fun, par_opt


def calculate_ecf_stats(  # noqa: N802
    distance: np.ndarray,
    covariance: np.ndarray,
    covariance_weights: np.ndarray,
    valid_heights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate statistics for Empirical Covariance Function (ECF), climatological version.

    Uses the histogram data from all previous days and reapplies the same steps, but inputs are of size (timesteps x
    variogram_bins). So if we use many days to compute the histogram bins, we get a histogram per day. This function
    generates a single output from a new histogram.

    Parameters
    ----------
    distance : np.ndarray
        Array of distances.
    covariance : np.ndarray
        Array of covariances.
    covariance_weights : np.ndarray
        Array of weights for covariances.
    valid_heights : np.ndarray
        Array of valid heights.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the following:
        - h_b: Array of mean distances for each height bin.
        - cov_b: Array of weighted average covariances for each height bin.
        - std_b: Array of standard deviations for each height bin.
    """
    valid_heights_count = len(valid_heights)

    # Create the empty arrays for the covariance, height and standard_deviation matrices for the correct number of bins
    # (This is valid_heights_count -1)
    cov_b = np.zeros(valid_heights_count - 1)
    h_b = np.zeros(valid_heights_count - 1)
    std_b = np.zeros(valid_heights_count - 1)

    # For each bin, get and aggregate the data that fits into that bin.
    for i in range(valid_heights_count - 1):
        # Find the indices of the data that need to go into that bin
        ind = np.where((distance >= valid_heights[i]) & (distance < valid_heights[i + 1]))

        # Compute the mean distance of points in that bin
        h_b[i] = np.mean(distance[ind])

        # Get the weights for that bin
        weight = covariance_weights[ind] / np.sum(covariance_weights[ind])

        # Get the covariance, weighted average of the covariance
        cov_b[i] = np.sum(weight * covariance[ind])
        average = np.average(covariance[ind], weights=weight)

        # Get the weighted average of the covariance error field
        variance = np.average((covariance[ind] - average) ** 2, weights=weight)

        # Get the standard deviation of the error field
        std_b[i] = np.sqrt(variance)

    return h_b, cov_b, std_b


def eval_covariance_bin(
    distances: np.ndarray,
    values: np.ndarray,
    hmax_divider: float = 2.0,
    variogram_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the covariance of a binomial distribution.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances for each data point.
    values : np.ndarray
        Array of values corresponding to each data point.
    hmax_divider : float
        Maximum distance for binning is set as hmax_divider times the maximum distance in the input data. Defaults to 2.
    variogram_bins : int, optional
        Number of bins to split the data to fit the semi-variogram for the ECF. Defaults to 10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays for heights, covariance, standard deviation, row length.
    """
    # Step 1: Calculate weights based on errors
    weights = np.power(1 / np.ones(len(values)), 2)
    weights = weights / np.sum(weights)

    # Step 2: Calculate weighted average and variances
    weighted_average = np.sum(values * weights) / np.sum(weights)
    variances = np.var(values, ddof=1)

    # Step 3: Calculate covariance matrix
    weighted_values = values - weighted_average
    covariance = weighted_values * weighted_values[:, np.newaxis]
    covariance_weight = weights * weights[:, np.newaxis]

    # Flatten matrices for further processing and binning
    covariance = covariance.reshape(len(values) * len(values))
    covariance_weight = covariance_weight.reshape(len(values) * len(values))
    distances = distances.reshape(len(distances) * len(distances))

    # Step 4: Apply distance threshold (hmax) for binning. Keep only those catchments that are less than
    # hmax/hmax_divider.
    hmax = max(distances) / hmax_divider
    covariance = covariance[distances < hmax]
    distances = distances[distances < hmax]

    # Step 5: Define quantiles for binning
    quantiles = np.linspace(0.0, 1.0, num=variogram_bins + 1)

    # Step 6: Get the edge values of each bin / class, based on the unraveled distance vector (timesteps x stations)
    cl = np.unique(np.quantile(distances, quantiles))

    # Initialize arrays for results, for all data that will go into each bin. Using the final bins after the unique
    # function in case of many values in the same bin (ex. zeros), so len(cl)-1 = number of actual bins (because
    # cl are edges).
    returned_covariance = np.empty((1, len(cl) - 1)) * np.nan
    returned_heights = np.empty((1, len(cl) - 1)) * np.nan
    returned_standard = np.empty((1, len(cl) - 1)) * np.nan
    returned_row_length = np.empty((1, len(cl) - 1)) * np.nan

    # Step 6: Iterate over distance bins.
    for i in range(0, len(cl) - 1):
        # For each bin (between edges), find all distance values falling in this bin.
        ind = np.where((distances >= cl[i]) & (distances < cl[i + 1]))[0]

        # Take the mean value of all distances values within the bin.
        returned_heights[:, i] = np.mean(distances[ind])
        # Take the covariance weights and covariance values of stations in the bin
        selected_covariance_weight = covariance_weight[ind]
        selected_covariance = covariance[ind]

        # Step 7: Calculate covariance, standard deviation, and row length
        weight = selected_covariance_weight / np.sum(selected_covariance_weight)

        # Compute the weighted covariance within the bin
        returned_covariance[:, i] = (
            np.sum(weight) / (np.power(np.sum(weight), 2) - np.sum(np.power(weight, 2))) * np.sum(weight * selected_covariance)
        ) / variances

        # Get the standard variation of the covariances of stations in the bin
        returned_standard[:, i] = np.sqrt(np.var(selected_covariance))

        # Also get the number of stations within that bin.
        returned_row_length[:, i] = len(ind)

    # Step 8: Return the final results as a tuple
    return returned_heights, returned_covariance, returned_standard, returned_row_length


def initialize_stats_variables(
    heights: np.ndarray,
    covariances: np.ndarray,
    standard_deviations: np.ndarray,
    variogram_bins: int = 10,
) -> tuple:
    """
    Initialize variables for statistical calculations in an Empirical Covariance Function (ECF).

    Parameters
    ----------
    heights : np.ndarray
        Array of heights.
    covariances : np.ndarray
        Array of covariances.
    standard_deviations : np.ndarray
        Array of standard deviations.
    variogram_bins : int
        Number of bins to split the data to fit the semi-variogram for the ECF. Defaults to 10.

    Returns
    -------
    tuple
        A tuple containing the following:
        - distance: Array of distances.
        - covariance: Array of covariances.
        - covariance_weights: Array of weights for covariances.
        - valid_heights: Array of valid heights.
    """
    quantiles = np.linspace(0.0, 1.0, variogram_bins + 1)
    valid_heights = np.unique(np.quantile(heights[~np.isnan(heights)], quantiles))

    distance = heights.T.reshape(len(heights) * len(heights[0]))
    covariance = covariances.T.reshape(len(covariances) * len(covariances[0]))
    covariance_weights = 1 / np.power(
        standard_deviations.T.reshape(len(standard_deviations) * len(standard_deviations[0])),
        2,
    )

    return distance, covariance, covariance_weights, valid_heights


def main(
    start_time: dt.datetime,
    end_time: dt.datetime,
    filename_stations: str,
    filename_gridded: str,
    filename_output: str,
    grid_resolution: float,
    var_name_gridded: str,
    var_name_stations: str,
    dims_gridded: tuple[str, str, str],
    dims_stations: str,
    coords_gridded: tuple[str, str, str],
    station_req_vars: tuple[str, str, str],
    percentiles: list[float] | None,
    p1_bnds: list[float] | None,
    hmax_mult_range_bnds: list[float] | None,
    var_bg_ratio: float | None = 0.15,
    variogram_bins: int | None = 10,
    hmax_divider: float | None = 2.0,
    ecf_form: int = 1,
    do_cross_validation: bool = True,
    frac_validation: float | None = 0.20,
):
    """
    Main optimal interpolation access point and controller.

    Parameters
    ----------
    start_time : dt.datetime
        The start date of the optimal interpolation period.
    end_time : dt.datetime
        The end date of the optimal interpolation period.
    filename_stations : str
        The path and name of the file that contains the stations (observations) data. Must be a netcdf file.
    filename_gridded : str
        The path and name of the file that contains the estimated (background) data. Must be a netcdf file.
    filename_output : str
        The path and name of the file that will be output, containing the processed optimal interpolation. Must be a
        netcdf file.
    grid_resolution : float
        Resolution of the gridded product, for rounding purposes.
    var_name_gridded : str
        Name of the meteo variable (i.e. precipitation) in the gridded dataset.
    var_name_stations : str
        Name of the meteo variable (i.e. precipitation) in the stations dataset.
    dims_gridded : Tuple[str, str, str]
        The three variable names that should be in the dataset, corresponding to latitude, longitude and time. For this
        package we are expecting "latitude", "longitude", and "time".
    dims_stations : str
        The name of the dimensions for the station data file. Should be "station" or "number" or some other integer-
        based ID.
    coords_gridded : Tuple[str, str, str]
        The coordinates of the gridded xr.Dataset that must be present, corresponding to latitude, longitude and time.
        For this package we are expecting "latitude", "longitude", and "time".
    station_req_vars : Tuple[str, str, str]
        The required variables in the station Dataset along with the precipitation values. These correspond to the
        latitude, longitude and altitude of stations.
    percentiles : list[float] | None
        The percentiles of the distribution to sample after generating the optimal interpolation.
    p1_bnds : list[float] | None
        Bounds of the parameter p1 in the ECF function.
    hmax_mult_range_bnds : list[float] | None
        Bounds of the hmax parameter in the ECF function.
    var_bg_ratio : float | None
        Variance ratio of the background field (estimated gridpoints).
    variogram_bins : int | None
        Number of bins to categorize data to build the ECF. Needs to be scaled such that each bin has a sufficient number
        of data points.
    hmax_divider : float | None
        Parameter that divides the hmax distance to know how far to look for station covariance. Higher means a smaller
        range.
    ecf_form : int
        Form of the ECF fitting equation. Can be between 1 and 4. Use trial and error to determine the best empirical
        fit. See notes below for more details.
    do_cross_validation : bool
        A flag to indicate that the user wants to perform a cross-validation step to assess the method's performance.
    frac_validation : float | None
        The fraction of stations to use for cross-validation, independent of the training data. Between 0 and 1.

    Notes
    -----
    The possible forms for the ecf function fitting are as follows:
        Form 1 (From Lachance-Cloutier et al. 2017; and Garand & Grassotti 1995) :
            ecf_fun = par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
        Form 2 (Gaussian form) :
            ecf_fun = par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
        Form 3 :
            ecf_fun = par[0] * np.exp(-h / par[1])
        Form 4 :
            ecf_fun = par[0] * np.exp(-(h ** par[1]) / par[0])
    """
    # Initialize variables if not provided
    if percentiles is None:
        percentiles = [50]

    if p1_bnds is None:
        p1_bnds = [0.95, 1.0]

    if hmax_mult_range_bnds is None:
        hmax_mult_range_bnds = [0.05, 3]

    # Prepare all the initial data for cross-validation
    (
        departures_train,
        departures_valid,
        lat_obs_train,
        lon_obs_train,
        lat_obs_valid,
        lon_obs_valid,
        lat_est,
        lon_est,
        estimated_precip_reshaped,
        original_shape,
        observed_precip_valid,
        ds_gridded,
    ) = prepare_initial_data(
        filename_stations,
        filename_gridded,
        start_time,
        end_time,
        frac_validation,
        do_cross_validation,
        grid_resolution,
        var_name_gridded,
        var_name_stations,
        dims_gridded,
        dims_stations,
        coords_gridded,
        station_req_vars,
    )

    # create the weighting function parameters using climatological errors (i.e. over many timesteps)
    ecf_fun, par_opt = correction(
        difference=departures_train,
        centroid_lon_obs=lon_obs_train,
        centroid_lat_obs=lat_obs_train,
        variogram_bins=variogram_bins,
        form=ecf_form,
        hmax_divider=hmax_divider,
        p1_bnds=p1_bnds,
        hmax_mult_range_bnds=hmax_mult_range_bnds,
    )

    # Do the actual implementation of optimal interpolation and return the estimated precipitation percentiles at all
    # grid cells of the background field.
    precip_percentiles = optimal_interpolation_operational_control(
        estimated_precip_reshaped.T,
        difference=departures_train.T,
        lat_obs=lat_obs_train,
        lon_obs=lon_obs_train,
        lat_est=lat_est,
        lon_est=lon_est,
        percentiles=percentiles,
        ratio_var_bg=var_bg_ratio,
        ecf_fun=ecf_fun,
        par_opt=par_opt,
    )

    # Save results before going any further. This is an intensive computation so saving results before processing is
    # recommended.
    # Generate time vector from start and end dates.
    time_list = pd.date_range(start_time, end_time)

    # Create the final dataset that cab be used in other processes at a later time.
    ds_final = xr.Dataset(
        {
            "tp": (
                ("percentile", "time", "latitude", "longitude"),
                precip_percentiles.reshape(len(percentiles), len(time_list), original_shape[0], original_shape[1]),
            )
        },
        coords={
            "percentile": percentiles,
            "time": pd.date_range(start_time, end_time),
            "latitude": ds_gridded.latitude,
            "longitude": ds_gridded.longitude,
        },
    )
    ds_final.to_netcdf(filename_output)

    # Now let's evaluate how good the optimal-interpolation gridded product is on the validation set.
    if do_cross_validation:
        # For testing purposes, select a percentile value. We only have 1 here (50th) and
        # it occupies the 1st (and only) element in the 1st dimension. Extract all values
        # for this percentile.
        precip_medians = precip_percentiles[0, :, :].T

        # Calculate some indices and RMSE values
        cv.evaluate_cross_validation(observed_precip_valid, estimated_precip_reshaped, precip_medians, lat_obs_valid, lon_obs_valid, lat_est, lon_est)

        # Plot them for quick analysis
        cv.plot_cross_validation_rmse_results(filename_stations)
