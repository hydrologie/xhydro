import numpy as np
from functools import partial
from .mathematical_algorithms import calculate_average_distance
from .utilities import initialize_nan_arrays

"""
Perform the actual optimal interpolation step.
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

    covariance_obs_vs_obs = oi_input['error_cov_fun'](distance_obs_vs_obs) / oi_input['error_cov_fun'](0)

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
Nouvelle fonction qui prend en input les valeurs de l'itérateur "args" préparés
juste avant, les dézip dans ses composantes, puis roule le code pour une seule
station de validation leave-one-out. L'ordre est contrôlé par le pool, donc
cette fonction est roulée en parallèle. Retourne les 3 quantiles souhaités vers
le pool.map qui l'appelle.
"""
def loop_interpolation_optimale_stations(i, args):
    # Dézipper les inputs requis de l'itérateur args
    station_count, qobs_log, qsim_log, ecf_fun, par_opt, PX, PY, time_range, debit_obs, superficie_drainee = args

    # J'ai importé des constantes ici pour éviter de les traîner pour rien
    index = range(0, station_count)
    ratio_var_bg = 0.15

    # Définition des 3 vecteurs de sortie. Ici ils sont des vecteurs et pas des
    # matrices car on travaille sur un seul bassin à la fois (en parallèle).
    qest_l1o, qest_l1o_q25, qest_l1o_q75 = initialize_nan_arrays(time_range, 3)

    # Code initial commence en quelque part ici.
    index_validation = i
    index_calibration = np.setdiff1d(index, index_validation)

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
