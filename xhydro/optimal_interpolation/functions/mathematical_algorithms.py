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

    returned_covariance = np.empty((1, iteration_count))
    returned_heights = np.empty((1, iteration_count))
    returned_standard = np.empty((1, iteration_count))
    returned_row_length = np.empty((1, iteration_count))

    returned_covariance[:, :] = np.nan
    returned_heights[:, :] = np.nan
    returned_standard[:, :] = np.nan
    returned_row_length[:, :] = np.nan

    for i in range(0, len(cl) - 1):
        ind = np.where((distances >= cl[i]) & (cl[i + 1] > distances))[0]
        returned_heights[:, i] = np.mean(distances[ind])

        selected_covariance_weight = covariance_weight[ind]
        selected_covariance = covariance[ind]

        weight = selected_covariance_weight / np.sum(selected_covariance_weight)

        returned_covariance[:, i] = (np.sum(weight) / (np.power(np.sum(weight), 2) - np.sum(np.power(weight, 2))) * np.sum(
            weight * selected_covariance)) / variances

        returned_standard[:, i] = np.sqrt(np.var(selected_covariance))
        returned_row_length[:, i] = len(ind)

    return returned_heights, returned_covariance, returned_standard, returned_row_length

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


"""
Calcule le coefficient d'efficacité KGE
Arguments :
obs (list): Liste qui contient les débits observés
sim (list) : Liste qui contient les débits simulés
Retourne :
(float): Le coefficient d'efficacité KGE.
"""
def kge_prime(obs, sim):
    is_nan = np.isnan(obs) | np.isnan(sim)

    obs = obs[~is_nan]
    sim = sim[~is_nan]

    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)

    obs_std = np.std(obs)
    sim_std = np.std(sim)

    r = np.corrcoef(obs, sim)[0, 1]

    beta = sim_mean / obs_mean
    gamma = (sim_std / sim_mean) / (obs_std / obs_mean)

    return 1 - np.sqrt(np.power((r - 1), 2) + np.power((beta - 1), 2) + np.power((gamma - 1), 2))


"""
Calcule le coefficient d'efficacité Nash–Sutcliffe
Arguments :
obs (list): Liste qui contient les débits observés
sim (list) : Liste qui contient les débits simulés
Retourne :
(float): Le coefficient d'efficacité Nash–Sutcliffe.
"""
def nash(obs, sim):
    sim = np.ma.array(sim, mask=np.isnan(obs))
    obs = np.ma.array(obs, mask=np.isnan(obs))

    sse = np.sum(np.power(obs - sim, 2))
    ssu = np.sum(np.power(obs - np.mean(obs), 2))
    return 1 - sse / ssu
