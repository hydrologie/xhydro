import numpy as np
import scipy.optimize
from functools import partial
from .mathematical_algorithms import calculate_average_distance, eval_covariance_bin
from .utilities import initialize_nan_arrays, general_ecf

def correction(flow_obs, flow_sim, x_points, y_points, savename, iteration_count=10):
    difference = flow_sim - flow_obs
    time_range = np.shape(difference)[0]

    heights, covariances, standard_deviations = initialize_nan_arrays((time_range, iteration_count), 3)

    input_opt = {'hmax_divider': 2, 'p1_bnds': [0.95, 1], 'hmax_mult_range_bnds': [0.05, 3]}
    form = 3

    distance = calculate_average_distance(x_points, y_points)

    for i in range(time_range):
        is_nan = np.isnan(difference[i, :])
        ecart_jour = difference[i, ~is_nan]
        errors = np.ones((len(ecart_jour)))

        if len(ecart_jour) >= 10:

            is_nan_horizontal = is_nan[:len(distance)]
            is_nan_vertical = is_nan_horizontal.reshape(1, len(distance))

            distancePC = distance[~is_nan_horizontal]
            distancePC = distancePC[:, ~is_nan_vertical[0, :]]

            h_b, cov_b, std_b, NP = eval_covariance_bin(distancePC, ecart_jour, errors,
                                                        input_opt['hmax_divider'], iteration_count)

            if len(NP[0]) >= 10:
                heights[i, :] = h_b[0, 0:11]
                covariances[i, :] = cov_b[0, 0:11]
                standard_deviations[i, :] = std_b[0, 0:11]

    distance, covariance, covariance_weights, valid_heights, valid_heights_count = \
        initialize_stats_variables(heights, covariances, standard_deviations, iteration_count)

    h_b, cov_b, std_b = \
        calculate_ECF_stats(distance, covariance, covariance_weights, valid_heights, valid_heights_count)

    # Nouvelle approche pour le ecf_fun, au lieu d'avoir des lambdas on y va avec des partial.
    ecf_fun = partial(general_ecf, form=form)

    weights = 1 / np.power(std_b, 2)
    weights = weights / np.sum(weights)
    rmse_fun = lambda par: np.sqrt(np.mean(weights * np.power(ecf_fun(h=h_b, par=par) - cov_b, 2)))

    par_opt = scipy.optimize.minimize(rmse_fun, [np.mean(cov_b), np.mean(h_b) / 3],
                                      bounds=([input_opt['p1_bnds'][0], input_opt['p1_bnds'][1]],
                                              [0, input_opt['hmax_mult_range_bnds'][1] * 500]))['x']

    # Faire graphique et sauvegarde

    return ecf_fun, par_opt

def initialize_ajusted_ECF_climate_variables(flow_obs, flow_sim, x_points, y_points, iteration_count):
    difference = flow_sim - flow_obs

    station_count = np.shape(x_points)[1]
    time_range = np.shape(difference)[0]

    heights = np.empty((time_range, iteration_count))
    covariances = np.empty((time_range, iteration_count))
    standard_deviations = np.empty((time_range, iteration_count))

    heights[:, :] = np.nan
    covariances[:, :] = np.nan
    standard_deviations[:, :] = np.nan

    return difference, station_count, time_range, heights, covariances, standard_deviations

def calculate_ECF_stats(distance, covariance, covariance_weights, valid_heights, valid_heights_count):
    cov_b = np.zeros(valid_heights_count - 1)
    h_b = np.zeros(valid_heights_count - 1)
    std_b = np.zeros(valid_heights_count - 1)
    for i in range(valid_heights_count - 1):
        ind = np.where((distance >= valid_heights[i]) & (distance < valid_heights[i + 1]))
        h_b[i] = np.mean(distance[ind])

        weight = covariance_weights[ind] / np.sum(covariance_weights[ind])

        cov_b[i] = np.sum(weight * covariance[ind])
        average = np.average(covariance[ind], weights=weight)
        variance = np.average((covariance[ind] - average) ** 2, weights=weight)
        std_b[i] = np.sqrt(variance)

    return h_b, cov_b, std_b

def initialize_stats_variables(heights, covariances, standard_deviations, iteration_count=10):
    quantities = np.linspace(0, 1, iteration_count + 1)
    valid_heights = np.unique(np.quantile(heights[~np.isnan(heights)], quantities))
    valid_heights_count = len(valid_heights)

    distance = heights.T.reshape(len(heights) * len(heights[0]))
    covariance = covariances.T.reshape(len(covariances) * len(covariances[0]))
    covariance_weights = 1 / np.power(
        standard_deviations.T.reshape(len(standard_deviations) * len(standard_deviations[0])), 2)

    return distance, covariance, covariance_weights, valid_heights, valid_heights_count
