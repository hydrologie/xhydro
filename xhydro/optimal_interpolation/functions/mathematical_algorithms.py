"""Set of mathematical algorithms required for the optimal interpolation."""

import numpy as np


def eval_covariance_bin(
    distances: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    hmax_divider: int = 2,
    iteration_count: int = 10,
):
    """Evaluate the covariance of a binomial distribution.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances for each data point.
    values : np.ndarray
        Array of values corresponding to each data point.
    errors : np.ndarray
        Array of errors (uncertainties) associated with each value.
    hmax_divider : int
        Maximum distance for binning is set as hmax_divider times the maximum distance in the input data. Defaults to 2.
    iteration_count : int, optional
        Number of iterations for refining the covariance estimate. Defaults to 10.

    Returns
    -------
    tuple
        Arrays for heights, covariance, standard deviation, and row length.
    """
    # Step 1: Calculate weights based on errors
    weights = np.power(1 / errors, 2)
    weights = weights / np.sum(weights)

    # Step 2: Calculate weighted average and variances
    weighted_average = np.sum(values * weights) / np.sum(weights)
    variances = np.var(values, ddof=1)

    # Step 3: Calculate covariance matrix
    weighted_values = values - weighted_average
    covariance = weighted_values * weighted_values[:, np.newaxis]
    covariance_weight = weights * weights[:, np.newaxis]

    # Flatten matrices for further processing
    covariance = covariance.reshape(len(values) * len(values))
    covariance_weight = covariance_weight.reshape(len(values) * len(values))
    distances = distances.reshape(len(distances) * len(distances))

    # Step 4: Apply distance threshold (hmax) for binning
    hmax = max(distances) / hmax_divider
    covariance = covariance[distances < hmax]
    distances = distances[distances < hmax]

    # Step 5: Define quantiles for binning
    quantiles = np.round(
        [(1 / iteration_count) * i for i in range(0, iteration_count + 1)], 2
    )
    cl = np.unique(np.quantile(distances, quantiles))

    # Initialize arrays for results
    returned_covariance = np.empty((1, iteration_count))
    returned_heights = np.empty((1, iteration_count))
    returned_standard = np.empty((1, iteration_count))
    returned_row_length = np.empty((1, iteration_count))

    # Set initial values to NaN
    returned_covariance[:, :] = np.nan
    returned_heights[:, :] = np.nan
    returned_standard[:, :] = np.nan
    returned_row_length[:, :] = np.nan

    # Step 6: Iterate over distance bins
    for i in range(0, len(cl) - 1):
        ind = np.where((distances >= cl[i]) & (cl[i + 1] > distances))[0]
        returned_heights[:, i] = np.mean(distances[ind])

        selected_covariance_weight = covariance_weight[ind]
        selected_covariance = covariance[ind]

        # Step 7: Calculate covariance, standard deviation, and row length
        weight = selected_covariance_weight / np.sum(selected_covariance_weight)
        returned_covariance[:, i] = (
            np.sum(weight)
            / (np.power(np.sum(weight), 2) - np.sum(np.power(weight, 2)))
            * np.sum(weight * selected_covariance)
        ) / variances
        returned_standard[:, i] = np.sqrt(np.var(selected_covariance))
        returned_row_length[:, i] = len(ind)

    # Step 8: Return the final results as a tuple
    return returned_heights, returned_covariance, returned_standard, returned_row_length


def calculate_average_distance(x_points: np.ndarray, y_points: np.ndarray) -> np.array:
    """Calculate the average Euclidean distance between points in 2D space.

    Parameters
    ----------
    x_points : np.ndarray
        List of x-coordinates of points.
    y_points : np.ndarray
        List of y-coordinates of corresponding points.

    Returns
    -------
    np.ndarray
        The average Euclidean distance between the points.
    """
    count = x_points.shape[1]
    average_distances = np.zeros((count, count))

    # Compute all pairwise distances in a vectorized manner
    for i in range(count):
        distances = np.sqrt(
            (x_points[:, i, np.newaxis] - x_points) ** 2
            + (y_points[:, i, np.newaxis] - y_points) ** 2
        )
        average_distances[i, :] = np.mean(distances, axis=0)

    # Fill in the symmetric part of the matrix
    average_distances = (average_distances + average_distances.T) / 2

    return average_distances


def latlon_to_xy(
    lat: np.ndarray, lon: np.ndarray, lat0: float = 0.0, lon0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Transform the geographic coordinate into the cartesian coordinate.

    Will shift the position of the origin at a specific latitude and longitude if required.

    Parameters
    ----------
    lat : np.ndarray
        List of latitude points.
    lon : np.ndarray
        List of longitude points.
    lat0 : float
        Latitude at origin. Defaults to 0.0.
    lon0 : float
        Longitude at origin. Defaults to 0.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Abscissas points and Ordinates points.
    """
    ray = 6371  # km

    lon = lon - lon0

    cos_lat = np.cos(np.deg2rad(lat))
    cos_lon = np.cos(np.deg2rad(lon))
    cos_lat0 = np.cos(np.deg2rad(lat0))

    sin_lat = np.sin(np.deg2rad(lat))
    sin_lon = np.sin(np.deg2rad(lon))
    sin_lat0 = np.sin(np.deg2rad(lat0))

    x = ray * cos_lat * sin_lon
    y = -ray * cos_lat * sin_lat0 * cos_lon + ray * cos_lat0 * sin_lat

    return x, y
