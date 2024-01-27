import numpy as np
import xarray as xr
from cProfile import Profile
from pstats import SortKey, Stats
import os
from multiprocessing import Pool
from .functions import mathematical_algorithms as ma
from .functions import ECF_climate_correction as ecf_cc
from .functions import optimal_interpolation as opt
from .functions import utilities as util


def execute(start_date, end_date, files, cpu_parrallel=False):
    r"""Start the profiler, run the entire code, and print stats on profile time


        Parameters
        ----------
        start_date : datetime
            Start date of the analysis.
        end_date : datetime
            End date of the analysis.
        files : list(str), optional
            List of files path for getting flows and wathersheds infos
        cpu_parrallel : bool, optional
            Execute the profiler in parallel or in series

        Returns
        -------
        flow_l1o (Leave one out cross validation flow)
        flow_l1o_percentile_25
        flow_l1o_percentile_75
        See Also
        --------
        xarray.open_dataset
        """
    with Profile() as profile:
        # Run the code
        results_1, results_2, results_3 = execute_interpolation(start_date, end_date, files, cpu_parrallel)

        # Print stats sorted by cumulative runtime.
        (Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

    return results_1, results_2, results_3

"""
Execute the main code, including setting constants to files, times, etc. Should
be converted to a function that takes these values as parameters.
Heavily modified to parallelize and to optimize.
"""
def execute_interpolation(start_date, end_date, files, cpu_parralel):
    stations_info, stations_mapping, stations_validation, flow_obs, flow_sim = load_files(files)


    args = {
        'flow_obs' : flow_obs,
        'flow_sim' : flow_sim,
        'start_date' : start_date,
        'end_date' : end_date,
        'stations_info' : stations_info,
        'stations_mapping' : util.convert_list_to_dict(stations_mapping),
        'stations_id' : [station[0] for station in stations_validation]
    }

    station_count, drained_area, centroid_lat, centroid_lon, selected_flow_obs, selected_flow_sim = \
        retreive_data(args)

    x, y = ma.latlon_to_xy(centroid_lat, centroid_lon, np.array([45] * len(centroid_lat)), np.array([-70] * len(centroid_lat)))  # Projete dans un plan pour avoir des distances en km

    x_points, y_points = standardize_points_with_roots(x, y , station_count, drained_area)

    # Fonction modifiée en profondeur ici par rapport au traitement du ecf_fun
    ecf_fun, par_opt = ecf_cc.correction(selected_flow_obs, selected_flow_sim, x_points, y_points, "test")
    args = (station_count, selected_flow_obs, selected_flow_sim, ecf_fun, par_opt, x_points, y_points, start_date,
         end_date, selected_flow_obs, drained_area)
    return parallelizing_operation(args, cpu_parralel)

def load_files(files):
    extract_files = [0] * len(files)
    count = 0
    for filepath in files:
        file_name, file_extension  = os.path.splitext(filepath)
        if file_extension == '.csv':
            extract_files[count] = util.read_csv_file(filepath)
        elif file_extension == '.nc':
            extract_files[count] = xr.open_dataset(filepath)
        count += 1
    return extract_files

def initialize_data_arrays(time_range, station_count):
    selected_flow_obs = np.empty((time_range, station_count))
    selected_flow_sim = np.empty((time_range, station_count))
    centroid_lat = np.empty(station_count)
    centroid_lon = np.empty(station_count)
    drained_area = np.empty(station_count)

    return selected_flow_obs, selected_flow_sim, centroid_lat, centroid_lon, drained_area
def retreive_data(args):
    flow_obs = args['flow_obs']
    flow_sim = args['flow_sim']
    start_date = args['start_date']
    end_date = args['end_date']
    stations_info = args['stations_info']
    stations_mapping = args['stations_mapping']
    stations_id = args['stations_id']

    station_count = len(stations_id)

    time_range = (end_date - start_date).days
    centroid_lat, centroid_lon, drained_area = util.initialize_nan_arrays(station_count, 3)
    selected_flow_obs, selected_flow_sim = util.initialize_nan_arrays((time_range, station_count), 2)

    for i in range(0, station_count):

        station_id = stations_id[i]
        associate_section = stations_mapping[station_id]

        index_section = util.find_index(flow_sim, 'station_id', associate_section)
        index_station = util.find_index(flow_obs, 'station_id', station_id)

        sup_sim = flow_sim.drainage_area[index_section].item()
        sup_obs = flow_obs.drainage_area[index_station].item()

        drained_area[i] = sup_obs

        selected_flow_sim[:, i] = flow_sim.Dis.sel(time=slice(start_date, end_date),station=index_section)[0:time_range] / sup_sim
        selected_flow_obs[:, i] = flow_obs.Dis.sel(time=slice(start_date, end_date), station=index_station)[0:time_range] / sup_obs

        position_info = np.where(np.array(stations_info) == station_id)
        station_info = stations_info[position_info[0].item()]
        centroid_lon[i], centroid_lat[i] = station_info[4], station_info[5]

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
        start_date = args[7]
        end_date = args[8]

        time_range = (end_date - start_date).days
        qest_l1o, qest_l1o_q25, qest_l1o_q75 = util.initialize_nan_arrays((time_range, station_count), 3)

        for i in range(0, station_count):
            qest_l1o[:, i], qest_l1o_q25[:, i], qest_l1o_q75[:, i] = opt.loop_interpolation_optimale_stations(i, args)
    return qest_l1o, qest_l1o_q25, qest_l1o_q75