import numpy as np
import csv
import scipy.optimize
import xarray as xr
from cProfile import Profile
from pstats import SortKey, Stats
import os
from multiprocessing import Pool
from functools import partial
import constants

"""
We will use functools.partial to define functions instead of lambdas as it is
more efficient and will allow parallelization. Therefore, define the ECF
function shape here. New function.
"""


def general_ecf(h, par, form):
    if form == 1:
        return par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
    elif form == 2:
        return par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
    else:
        return par[0] * np.exp(-h / par[1])


"""
Retourne une liste qui contient les valeurs d'un fichier CSV
Arguments :
csv_filename (string): Le nom du fichier CSV
header (bool) : Le fichier CSV contient une entête
Retourne :
(list): Liste qui contient les valeurs du fichier
"""


def read_csv_file(csv_filename, header, delimiter):
    items = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        if header == 1:
            skip = 1

        for row in reader:
            if skip != 1:
                items.append(row)
            else:
                skip = 0

    return items


"""
Trouve l'association d'une section à une station.
Arguments :
stations (list): Liste qui contient les stations
section_id (string) : L'indentificateur de la section
Retourne :
(string): Vide si la section est introuvable ou la clé d'association entre la station et une section.
"""


def find_section(stations, section_id):
    value = ""
    section_position = 0
    section_value = 1
    for i in range(0, len(stations)):
        if section_id == stations[i][section_position]:
            value = stations[i]
    return value[section_value]


"""
Trouve l'indince d'un élément donné dans une liste.
Arguments :
array (list): Liste qui contient les données
key (string) : Élément à trouver dans la liste
Retourne :
(float): -1 si l'élément est introuvable ou l'indice de l'élément
Modifié pas mal pour vectoriser.
"""


def find_index(array, key):
    logical = array.station_id.data == key.encode('UTF-8')
    return np.where(logical)[0][0]


"""
Does something!  Did not modify.
"""


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


"""
Does something!  Did not modify.
"""


def initialize_stats_variables(heights, covariances, standard_deviations, iteration_count=10):
    quantities = np.linspace(0, 1, iteration_count + 1)
    valid_heights = np.unique(np.quantile(heights[~np.isnan(heights)], quantities))
    valid_heights_count = len(valid_heights)

    distance = heights.T.reshape(len(heights) * len(heights[0]))
    covariance = covariances.T.reshape(len(covariances) * len(covariances[0]))
    covariance_weights = 1 / np.power(
        standard_deviations.T.reshape(len(standard_deviations) * len(standard_deviations[0])), 2)

    return distance, covariance, covariance_weights, valid_heights, valid_heights_count


"""
Does Something!  Did not modify.
"""


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


"""
Fait quelque chose!
Celle-ci a été significativement modifiée.
"""


def ajustement_ECF_climatologique(flow_obs, flow_sim, x_points, y_points, savename, iteration_count=10):
    difference, station_count, time_range, heights, covariances, standard_deviations = \
        initialize_ajusted_ECF_climate_variables(flow_obs, flow_sim, x_points, y_points, iteration_count)

    input_opt = {'hmax_divider': 2, 'p1_bnds': [0.95, 1], 'hmax_mult_range_bnds': [0.05, 3]}
    form = 3

    distance = calculate_average_distance(x_points, y_points)

    for i in range(time_range):
        print(i)
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


"""
Does something!  Did not modify.
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
Function that computes the distance in length units instead of lat/long units.
Did not modify.
"""


def latlon_to_xy(lat, lon, lat0, lon0):
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
Function that computes the pairwise distance between sets of points.
Modified to make use of vectorization, large speedup
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
Perform the actual optimal interpolation step. Did not modify.
"""


def optimal_interpolation(oi_input, args):
    if len(args) == 0:
        args = {}

    estimated_count = 1 if len(oi_input['x_est'].shape) == 1 else oi_input['x_est'].shape[1]
    observed_count = len(oi_input['x_obs'][0, :])
    oi_output = oi_input

    cond = 0

    if isinstance(args, dict):
        if 'x_obs' in args:
            cond = (np.array_equal(args['x_est'], oi_input['x_est']) \
                    and np.array_equal(args['y_est'], oi_input['y_est'])) \
                   and (np.array_equal(args['x_obs'], oi_input['x_obs']) \
                        and np.array_equal(args['y_obs'], oi_input['y_obs']))
    if cond == 0:
        distance_obs_vs_obs = calculate_average_distance(oi_input['x_obs'], oi_input['y_obs'])
    else:
        distance_obs_vs_obs = args['Doo']

    args['x_obs'] = oi_input['x_obs']
    args['y_obs'] = oi_input['y_obs']
    args['Doo'] = distance_obs_vs_obs

    covariance_obs_vs_obs = oi_input['error_cov_fun'](distance_obs_vs_obs) #/ oi_input['error_cov_fun'](0)

    BEo_j = np.tile(oi_input['bg_var_obs'], (observed_count, 1))
    BEo_i = np.tile(np.resize(oi_input['bg_var_obs'], (1, observed_count)), (observed_count, 1))

    Bij = covariance_obs_vs_obs * np.sqrt(BEo_j) / np.sqrt(BEo_i)

    OEo_j = np.tile(oi_input['var_obs'], (observed_count, 1))
    OEo_i = np.tile(oi_input['var_obs'], (1, observed_count))

    Oij = (np.sqrt(OEo_j) * np.sqrt(OEo_i)) * np.eye(len(OEo_j), len(OEo_j[0])) / BEo_i

    if cond == 0:
        distance_obs_vs_est = np.zeros((1, observed_count))
        x_est = oi_input['x_est']
        y_est = oi_input['y_est']

        for i in range(estimated_count):
            for j in range(observed_count // 2):
                distance_obs_vs_est[i, j] = np.mean(np.sqrt(
                    np.power(oi_input['x_obs'][:, j] - x_est[:], 2) + np.power(oi_input['y_obs'][:, j] - y_est[:], 2)))
                distance_obs_vs_est[i, -j - 1] = np.mean(np.sqrt(
                    np.power(oi_input['x_obs'][:, -j - 1] - x_est[:], 2) + np.power(oi_input['y_obs'][:, -j - 1] - y_est[:], 2)))
    else:
        distance_obs_vs_est = args['distance_obs_vs_est']

    args['x_est'] = oi_input['x_est']
    args['y_est'] = oi_input['y_est']
    args['distance_obs_vs_est'] = distance_obs_vs_est

    BEe = np.tile(np.resize(oi_input['bg_var_est'], (1, observed_count)), (estimated_count, 1))
    BEo = np.tile(oi_input['bg_var_obs'], (estimated_count, 1))

    Coe = oi_input['error_cov_fun'](distance_obs_vs_est) / oi_input['error_cov_fun'](0)

    Bei = np.resize(Coe * np.sqrt(BEe) / np.sqrt(BEo), (observed_count, 1))

    departures = oi_input['bg_departures'].reshape((1,len(oi_input['bg_departures'])))

    weights = np.linalg.solve(Bij + Oij, Bei)
    weights = weights.reshape((1, len(weights)))

    oi_output['v_est'] = oi_input['bg_est'] + np.sum(weights * departures)
    oi_output['var_est'] = oi_input['bg_var_est'] * (1 - np.sum(Bei[:, 0] * weights[0,:]))

    return oi_output, args


"""
Execute the main code, including setting constants to files, times, etc. Should
be converted to a function that takes these values as parameters.
Heavily modified to parallelize and to optimize.
"""


def execute():
    start_date = np.datetime64('1961-01-01')
    end_date = np.datetime64('2019-01-01')
    time_series = (end_date - start_date) / np.timedelta64(1, 'D')

    obs_data_filename = constants.DATA_PATH + 'A20_HYDOBS_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_XXX_XXXXXX_XXX_HC_13102020.nc'
    sim_data_filename = constants.DATA_PATH + 'A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_MG24HS_GCQ_SC_18092020.nc'

    station_validation_filename = constants.DATA_PATH + "stations_retenues_validation_croisee.csv"
    station_mapping_filename = constants.DATA_PATH + "Table_Correspondance_Station_Troncon.csv"
    station_info_filename = constants.DATA_PATH + "Table_Info_Station_Hydro_2020.csv"

    print("Lecture des CSV")
    stations_validation = read_csv_file(station_validation_filename, 1, ',')
    stations_mapping = read_csv_file(station_mapping_filename, 1, ',')
    stations_info = read_csv_file(station_info_filename, 1, ";")

    print("Lecture des NC")
    ds_obs = xr.open_dataset(obs_data_filename)
    ds_sim = xr.open_dataset(sim_data_filename)

    stations_id_obs = ds_obs.station_id
    stations_id_sim = ds_sim.station_id

    da_obs = ds_obs.drainage_area
    da_sim = ds_sim.drainage_area

    dis_obs = ds_obs.Dis
    dis_sim = ds_sim.Dis

    time_range = int(time_series)
    station_count = len(stations_validation)
    debit_sim = np.empty((time_range, station_count))
    debit_obs = np.empty((time_range, station_count))
    centroide_lat = np.empty(station_count)
    centroide_lon = np.empty(station_count)
    superficie_drainee = np.empty(station_count)
    longitude_station = np.empty(station_count)
    latitude_station = np.empty(station_count)

    for i in range(0, station_count):
        print("Lecture des données..." + str(i + 1) + "/" + str(station_count))

        station_id = stations_validation[i][0]
        associate_section = find_section(stations_mapping, station_id)

        index_section = find_index(stations_id_sim, associate_section)
        index_station = find_index(stations_id_obs, station_id)

        sup_sim = da_sim[index_section].item()
        sup_obs = da_obs[index_station].item()

        superficie_drainee[i] = sup_obs

        debit_sim[:, i] = dis_sim.isel(station=index_section)[0:time_range] / sup_sim
        debit_obs[:, i] = dis_obs.isel(station=index_station)[0:time_range] / sup_obs

        position_info = np.where(np.array(stations_info) == station_id)
        station_info = stations_info[position_info[0].item()]
        centroide_lat[i] = station_info[5]
        centroide_lon[i] = station_info[4]
        longitude_station[i] = station_info[3]
        latitude_station[i] = station_info[2]

    lat0 = np.array([45] * len(centroide_lat))
    lon0 = np.array([-70] * len(centroide_lat))

    x, y = latlon_to_xy(centroide_lat, centroide_lon, lat0, lon0)  # Projete dans un plan pour avoir des distances en km

    PX = np.empty((4, station_count))
    PY = np.empty((4, station_count))
    for i in range(station_count):
        root_superficie = np.sqrt(superficie_drainee[i])
        xv = [x[i] - (root_superficie / 2), x[i] + root_superficie / 2]
        yv = [y[i] - (root_superficie / 2), y[i] + root_superficie / 2]
        [x_p, y_p] = np.meshgrid(xv, yv)

        x_p = np.transpose(x_p)
        y_p = np.transpose(y_p)

        PX[:, i] = x_p.reshape(2 * len(x_p))
        PY[:, i] = y_p.reshape(2 * len(y_p))

    # Transformation log-débit pour l'interpolation
    qsim_log = np.log(debit_sim)
    qobs_log = np.log(debit_obs)

    # Fonction modifiée en profondeur ici par rapport au traitement du ecf_fun
    ecf_fun, par_opt = ajustement_ECF_climatologique(qobs_log, qsim_log, PX, PY, "test")

    """
    Nouvelle section pour le calcul parallèle. Plusieurs étapes effectuées:
        1. Estimation de la puissance de calcul disponible. Réalisé en prenant
           le nombre de threads dispo / 2 (pour avoir le nombre de cores) puis
           soustrayant 2 pour en garder 2 dispo pour d'autres tâches.
        2. Démarrage d'un pool de calcul parallèle
        3. Création de l'itérateur (iterable) contenant les données requises
           par la nouvelle fonction qui boucle sur chacune des stations en
           validation leave-one-out. C'est grossièrement parallélisable alors
           on y va pour ça.
        4. Lance le pool.map qui map les inputs (iterateur) sur la fonction.
        5. Ramasser les résultats etdézipper le tuple qui retourne de pool.map.
        6. Fermer le pool et retourner les résultats "parsés".
    """
    parallel_calcs = False

    # Faire en parallèle
    if parallel_calcs == True:
        processes_count = os.cpu_count() / 2 - 1
        p = Pool(int(processes_count))
        args = [
            (i, station_count, qobs_log, qsim_log, ecf_fun, par_opt, PX, PY, time_range, debit_obs, superficie_drainee)
            for i in range(0, station_count)]
        qest_l1o, qest_l1o_q25, qest_l1o_q75 = zip(*p.map(loop_interpolation_optimale_stations, args))
        p.close()
        p.join()

    # Faire en série
    else:
        qest_l1o = np.empty((time_range, station_count))
        qest_l1o_q25 = np.empty((time_range, station_count))
        qest_l1o_q75 = np.empty((time_range, station_count))
        qest_l1o[:, :] = np.nan
        qest_l1o_q25[:, :] = np.nan
        qest_l1o_q75[:, :] = np.nan

        for i in range(0, station_count):
            args = [i, station_count, qobs_log, qsim_log, ecf_fun, par_opt, PX, PY, time_range, debit_obs,
                    superficie_drainee]
            qest_l1o[:, i], qest_l1o_q25[:, i], qest_l1o_q75[:, i] = loop_interpolation_optimale_stations(args)

    return qest_l1o, qest_l1o_q25, qest_l1o_q75


"""
Nouvelle fonction qui prend en input les valeurs de l'itérateur "args" préparés
juste avant, les dézip dans ses composantes, puis roule le code pour une seule
station de validation leave-one-out. L'ordre est contrôlé par le pool, donc
cette fonction est roulée en parallèle. Retourne les 3 quantiles souhaités vers
le pool.map qui l'appelle.
"""


def loop_interpolation_optimale_stations(args):
    # Dézipper les inputs requis de l'itérateur args
    i, station_count, qobs_log, qsim_log, ecf_fun, par_opt, PX, PY, time_range, debit_obs, superficie_drainee = args

    # J'ai importé des constantes ici pour éviter de les traîner pour rien
    index = range(0, station_count)
    ratio_var_bg = 0.15

    # Définition des 3 vecteurs de sortie. Ici ils sont des vecteurs et pas des
    # matrices car on travaille sur un seul bassin à la fois (en parallèle).
    qest_l1o = np.empty(time_range)
    qest_l1o_q25 = np.empty(time_range)
    qest_l1o_q75 = np.empty(time_range)
    qest_l1o[:] = np.nan
    qest_l1o_q25[:] = np.nan
    qest_l1o_q75[:] = np.nan

    # Code initial commence en quelque part ici.
    index_validation = i
    index_calibration = np.setdiff1d(index, i)

    ecart = qobs_log[:, index_calibration] - qsim_log[:, index_calibration]
    vsim_at_est = qsim_log[:, index_validation]

    # Restructuré l'objet oi_input pour éviter de le réécrire à chaque fois
    # dans la boucle "j" ici-bas. On fait juste updater le dictionnaire.
    oi_input = {}
    oi_input.update({
        'var_obs': ratio_var_bg,
        'error_cov_fun': partial(ecf_fun, par=par_opt),
        'x_est': PX[:, index_validation],
        'y_est': PY[:, index_validation],
    })

    preCalcul = {}

    for j in range(time_range):
        if not np.isnan(debit_obs[j, index_validation]):
            val = ecart[j, :]
            idx = ~np.isnan(val)

            # Même chose ici, j'ai modifié le format de dictionnaire pour l'updater seulement.
            oi_input.update({
                'x_obs': PX[:, index_calibration[idx]],
                'y_obs': PY[:, index_calibration[idx]],
                'bg_departures': ecart[j, idx],
                'bg_var_obs': np.ones(idx.sum()),
                'bg_est': vsim_at_est[j],
                'bg_var_est': 1
            })

            oi_output, preCalcul = optimal_interpolation(oi_input, preCalcul)

            qest_l1o[j] = np.exp(oi_output['v_est']) * superficie_drainee[i]

            var_bg = np.var(ecart[j, idx])
            var_est = oi_output['var_est'] * var_bg

            # Ici j'ai changé le nombre de samples à 500, c'est suffisant.
            # De plus, on sample 1 fois et on calcule les quantiles 25 et 75 à
            # partir de la même distribution échantillonnée, ça ne change rien
            # et sauve 50% des calculs.
            random_samples = np.random.normal(oi_output['v_est'], np.sqrt(var_est), 500)  # Shorten to 100?

            qest_l1o_q25[j] = np.exp(np.percentile(random_samples, 25)) * superficie_drainee[i]
            qest_l1o_q75[j] = np.exp(np.percentile(random_samples, 75)) * superficie_drainee[i]

    # Retourne les 3 vecteurs à notre pool parallèle.
    return qest_l1o, qest_l1o_q25, qest_l1o_q75
    return qest_l1o, qest_l1o_q25, qest_l1o_q75


"""
Launch main code here by pressing run/debug
"""
if __name__ == '__main__':
    """
    Start the profiler, run the entire code, and print stats on profile time
    """

    with Profile() as profile:
        # Run the code
        results_1, results_2, results_3 = execute()

        # Print stats sorted by cumulative runtime.
        (Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

