# noqa: N999
"""Empirical Covariance Function variogram calibration package."""

from functools import partial
from typing import Optional

import haversine
import numpy as np
import scipy.optimize
import xarray as xr


def correction(
    da_qobs: xr.DataArray,
    da_qsim: xr.DataArray,
    centroid_lon_obs: np.ndarray,
    centroid_lat_obs: np.ndarray,
    variogram_bins: int = 10,
    form: int = 3,
    hmax_divider: float = 2.0,
    p1_bnds: list | None = None,
    hmax_mult_range_bnds: list | None = None,
) -> tuple:
    """Perform correction on flow observations using optimal interpolation.

    Parameters
    ----------
    da_qobs : xr.DataArray
        An xarray DataArray of observed flow data.
    da_qsim : xr.DataArray
        An xarray DataArray of simulated flow data.
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
    # Calculate the difference between the background field (qsim) and the point observations (qobs) at the station
    # locations.
    difference = da_qsim.values - da_qobs.values

    # Number of timesteps. If we have more than 1 timestep, we can use the mean climatological variogram later.
    if "time" in da_qobs.dims:
        time_range = len(da_qobs.time)
    else:
        time_range = 1

    # Preallocate matrices for all timesteps for the heights of the histograms, the covariances and standard deviations
    # of the values within each bin of the histogram.
    heights = np.empty((time_range, variogram_bins)) * np.nan
    covariances = np.empty((time_range, variogram_bins)) * np.nan
    standard_deviations = np.empty((time_range, variogram_bins)) * np.nan

    if p1_bnds is None:
        p1_bnds = [0.95, 1]
    if hmax_mult_range_bnds is None:
        hmax_mult_range_bnds = [0.05, 3]

    # Pairwise distance between all observation stations.
    observation_latlong = list(zip(centroid_lat_obs, centroid_lon_obs))
    distance = haversine.haversine_vector(
        observation_latlong, observation_latlong, comb=True
    )

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

            # If there are at least as many stations worth of data as there are required bins, we can compute the
            # histogram.
            if len(day_diff) >= variogram_bins:

                # Get the stations that did not have NaN observations. Since the matrix is 2D due to pairwise distances,
                # need to remove rows and columns of NaN-stations from the distance matrix distance_pc.
                distance_pc = np.delete(distance, is_nan, axis=0)
                distance_pc = np.delete(distance_pc, is_nan, axis=1)

                # Sanity check: length of distance_pc should be equal to day_diff.
                if len(day_diff) != distance_pc.shape[0]:
                    raise AssertionError(
                        "day_diff not equal to the size of distance_pc in histogram bin definition."
                    )

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
        distance, covariance, covariance_weights, valid_heights = (
            initialize_stats_variables(
                heights, covariances, standard_deviations, variogram_bins
            )
        )

        # And this second part does the binning of the histogram as was done before for all days.
        h_b, cov_b, std_b = calculate_ECF_stats(
            distance, covariance, covariance_weights, valid_heights
        )

    else:
        # Just compute the covariance bin as a one-shot deal
        h_b, cov_b, std_b, num_p = eval_covariance_bin(
            distances=distance,
            values=np.squeeze(difference),
            hmax_divider=hmax_divider,
            variogram_bins=variogram_bins,
        )

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
    par_opt = scipy.optimize.minimize(
        _rmse_func,
        [
            np.mean(cov_b),
            np.mean(h_b) / 3,
        ],  # TODO: Find out why the "3" is here. Seems out of place.
        bounds=(
            [p1_bnds[0], p1_bnds[1]],
            [hmax_mult_range_bnds[0] * hmax, hmax_mult_range_bnds[1] * hmax],
        ),
    )["x"]

    # Return the fitting function as determined by the user and the optimal calibrated parameter set.
    return ecf_fun, par_opt


def calculate_ECF_stats(  # noqa: N802
    distance: np.ndarray,
    covariance: np.ndarray,
    covariance_weights: np.ndarray,
    valid_heights: np.ndarray,
) -> tuple:
    """Calculate statistics for Empirical Covariance Function (ECF), climatological version.

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
    tuple
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
        ind = np.where(
            (distance >= valid_heights[i]) & (distance < valid_heights[i + 1])
        )

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
):
    """Evaluate the covariance of a binomial distribution.

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
    tuple
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
            np.sum(weight)
            / (np.power(np.sum(weight), 2) - np.sum(np.power(weight, 2)))
            * np.sum(weight * selected_covariance)
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
        standard_deviations.T.reshape(
            len(standard_deviations) * len(standard_deviations[0])
        ),
        2,
    )

    return distance, covariance, covariance_weights, valid_heights


def general_ecf(h: np.ndarray, par: list, form: int):
    """Define the form of the Error Covariance Function (ECF) equations.

    Parameters
    ----------
    h : float or array
        The distance or distances at which to evaluate the ECF.
    par : list
        Parameters for the ECF equation.
    form : int
        The form of the ECF equation to use (1, 2, 3 or 4). See :py:func:`correction` for details.

    Returns
    -------
    float or array:
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
