import numpy as np
import xarray as xr
from cProfile import Profile
from pstats import SortKey, Stats
import os
from multiprocessing import Pool
from . import constants
from .functions import utilities as util
from .functions import mathematical_algorithms as ma
from .functions import ECF_climate_correction as ecf_cc
from .functions import optimal_interpolation as opt

def execute():
    """
        Start the profiler, run the entire code, and print stats on profile time
        """

    with Profile() as profile:
        # Run the code
        results_1, results_2, results_3 = execute_interpolation()

        # Print stats sorted by cumulative runtime.
        (Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

    return results_1, results_2, results_3

"""
Execute the main code, including setting constants to files, times, etc. Should
be converted to a function that takes these values as parameters.
Heavily modified to parallelize and to optimize.
"""
def execute_interpolation():
    stations_info, stations_mapping, stations_validation, flow_obs, flow_sim = load_files()

    time_range = time_selection('1961-01-01', '2019-01-01')

    station_count, drained_area, centroid_lat, centroid_lon, selected_flow_obs, selected_flow_sim = \
        retreive_data(flow_obs, flow_sim, time_range, stations_info, stations_mapping, stations_validation)

    x, y = ma.latlon_to_xy(centroid_lat, centroid_lon, np.array([45] * len(centroid_lat)), np.array([-70] * len(centroid_lat)))  # Projete dans un plan pour avoir des distances en km

    x_points, y_points = standardize_points_with_roots(x, y , station_count, drained_area)

    # Fonction modifiée en profondeur ici par rapport au traitement du ecf_fun
    ecf_fun, par_opt = ecf_cc.correction(selected_flow_obs, selected_flow_sim, x_points, y_points, "test")
    args = (station_count, selected_flow_obs, selected_flow_sim, ecf_fun, par_opt, x_points, y_points, time_range,
         selected_flow_obs, drained_area)
    return parallelizing_operation(args, False)

def load_files():
    print("Lecture des CSV")
    stations_info = util.read_csv_file(constants.DATA_PATH + "Table_Info_Station_Hydro_2020.csv", 1, ";")
    stations_mapping = util.read_csv_file(constants.DATA_PATH + "Table_Correspondance_Station_Troncon.csv", 1, ',')
    stations_validation = util.read_csv_file(constants.DATA_PATH + "stations_retenues_validation_croisee.csv", 1, ',')

    print("Lecture des NC")
    flow_obs = xr.open_dataset(
        constants.DATA_PATH + 'A20_HYDOBS_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_XXX_XXXXXX_XXX_HC_13102020.nc')
    flow_sim = xr.open_dataset(
        constants.DATA_PATH + 'A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_MG24HS_GCQ_SC_18092020.nc')

    return stations_info, stations_mapping, stations_validation, flow_obs, flow_sim

def time_selection(start_date, end_date):
    return int((np.datetime64(end_date) - np.datetime64(start_date)) / np.timedelta64(1, 'D'))

def retreive_data(flow_obs, flow_sim, time_range, stations_info, stations_mapping, stations_validation):
    station_count = len(stations_validation)
    selected_flow_sim = np.empty((time_range, station_count))
    selected_flow_obs = np.empty((time_range, station_count))
    centroid_lat = np.empty(station_count)
    centroid_lon = np.empty(station_count)
    drained_area = np.empty(station_count)
    longitude_station = np.empty(station_count)
    latitude_station = np.empty(station_count)

    for i in range(0, station_count):
        print("Lecture des données..." + str(i + 1) + "/" + str(station_count))

        station_id = stations_validation[i][0]
        associate_section = util.find_section(stations_mapping, station_id)

        index_section = util.find_index(flow_sim.station_id, associate_section)
        index_station = util.find_index(flow_obs.station_id, station_id)

        sup_sim = flow_sim.drainage_area[index_section].item()
        sup_obs = flow_obs.drainage_area[index_station].item()

        drained_area[i] = sup_obs

        selected_flow_sim[:, i] = flow_sim.Dis.isel(station=index_section)[0:time_range] / sup_sim
        selected_flow_obs[:, i] = flow_obs.Dis.isel(station=index_station)[0:time_range] / sup_obs

        position_info = np.where(np.array(stations_info) == station_id)
        station_info = stations_info[position_info[0].item()]
        centroid_lat[i] = station_info[5]
        centroid_lon[i] = station_info[4]
        longitude_station[i] = station_info[3]
        latitude_station[i] = station_info[2]

    # Transformation log-débit pour l'interpolation
    selected_flow_obs = np.log(selected_flow_obs)
    selected_flow_sim = np.log(selected_flow_sim)

    return station_count, drained_area, centroid_lat, centroid_lon, selected_flow_obs, selected_flow_sim

def standardize_points_with_roots(x, y, station_count, drained_area):
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

def parallelizing_operation(args, parallezation=True):
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

    # Faire en parallèle
    if parallezation == True:
        processes_count = os.cpu_count() / 2 - 1
        p = Pool(int(processes_count))
        qest_l1o, qest_l1o_q25, qest_l1o_q75 = zip(*p.map(opt.loop_interpolation_optimale_stations, args))
        p.close()
        p.join()

    # Faire en série
    else:
        station_count = args[0]
        time_range = args[7]
        qest_l1o = np.empty((time_range, station_count))
        qest_l1o_q25 = np.empty((time_range, station_count))
        qest_l1o_q75 = np.empty((time_range, station_count))
        qest_l1o[:, :] = np.nan
        qest_l1o_q25[:, :] = np.nan
        qest_l1o_q75[:, :] = np.nan

        for i in range(0, station_count):
            qest_l1o[:, i], qest_l1o_q25[:, i], qest_l1o_q75[:, i] = opt.loop_interpolation_optimale_stations(i, args)
    return qest_l1o, qest_l1o_q25, qest_l1o_q75
