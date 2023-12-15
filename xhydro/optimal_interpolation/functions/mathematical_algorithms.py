import numpy as np

"""
Evalute the covariance of a binomial distribution
"""
def eval_covariance_bin(distances, values, errors, hmax_divider=2, iteration_count=20):
    n_data = len(values)
    weights = np.power(1 / errors, 2)
    weights = weights / np.sum(weights)

    weighted_average = np.sum(values * weights) / np.sum(weights)
    variances = np.var(values, ddof=1)

    weighted_values = values - weighted_average

    covariance = weighted_values * weighted_values[:, np.newaxis]
    covariance_weight = weights * weights[:, np.newaxis]

    covariance = covariance.reshape(n_data * n_data)
    covariance_weight = covariance_weight.reshape(n_data * n_data)

    distances = distances.reshape(len(distances) * len(distances))

    hmax = max(distances) / hmax_divider
    covariance = covariance[distances < hmax]
    distances = distances[distances < hmax]

    quantiles = np.round([(1 / iteration_count) * i for i in range(0, iteration_count + 1)], 2)
    cl = np.unique(np.quantile(distances, quantiles))

    cov_b = np.empty((1, iteration_count))
    h_b = np.empty((1, iteration_count))
    std_b = np.empty((1, iteration_count))
    n_ind = np.empty((1, iteration_count))

    cov_b[:, :] = np.nan
    h_b[:, :] = np.nan
    std_b[:, :] = np.nan
    n_ind[:, :] = np.nan

    for i in range(0, len(cl) - 1):
        ind = np.where((distances >= cl[i]) & (cl[i + 1] > distances))[0]
        h_b[:, i] = np.mean(distances[ind])

        selected_covariance_weight = covariance_weight[ind]
        selected_covariance = covariance[ind]

        weight = selected_covariance_weight / np.sum(selected_covariance_weight)

        cov_b[:, i] = (np.sum(weight) / (np.power(np.sum(weight), 2) - np.sum(np.power(weight, 2))) * np.sum(
            weight * selected_covariance)) / variances

        std_b[:, i] = np.sqrt(np.var(selected_covariance))
        n_ind[:, i] = len(ind)

    return h_b, cov_b, std_b, n_ind

"""
Function that computes the pairwise distance between sets of points.
"""
def calculate_average_distance(x_points, y_points):
    count = x_points.shape[1]
    average_distances = np.zeros((count, count))

    # Compute all pairwise distances in a vectorized manner
    for i in range(count):
        distances = np.sqrt((x_points[:, i, np.newaxis] - x_points) ** 2 +
                            (y_points[:, i, np.newaxis] - y_points) ** 2)
        average_distances[i, :] = np.mean(distances, axis=0)

    # Fill in the symmetric part of the matrix
    average_distances = (average_distances + average_distances.T) / 2

    return average_distances


"""
Function that computes the distance in length units instead of lat/long units.

Arguments:


"""


def latlon_to_xy(lat, lon, lat0=0, lon0=0):
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
