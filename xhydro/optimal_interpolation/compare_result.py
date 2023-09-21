import netCDF4
import numpy as np
import csv
import matplotlib.pyplot as plt

"""
Retourne une liste qui contient avec les valeurs d'un fichier netCDF
Arguments :
filename (string): Le nom du fichier netCDF
key (string) : La clé d'information recherché dans le fichier
Retourne :
(list): Liste qui contient les valeurs du fichier
"""


def nc_read(filename, key):
    return netCDF4.Dataset(filename).variables[key]


"""
Retourne une liste qui contient avec les valeurs d'un fichier netCDF dont celles-ci furent
séparés en caractères
Arguments :
filename (string): Le nom du fichier netCDF
key (string) : La clé d'information recherché dans le fichier
nchar_dimid (int) : La dimension des chaines de caractères séparés
Retourne :
(list): Liste qui contient les valeurs du fichier
"""


def nc_read_char2string(filename, key, nchar_dimid):
    dataset = netCDF4.Dataset(filename)
    data_values = dataset.variables[key]
    result = []
    index = -1

    if data_values.dimensions.count(nchar_dimid) > 0:
        index = data_values.dimensions.index(nchar_dimid)

    if index > 0:
        for i in range(0, data_values.shape[0]):
            result.append(''.join(str(data_values[i, :], encoding='utf-8')))
    return result


"""
Retourne une liste qui contient les valeurs d'un fichier CSV
Arguments :
csv_filename (string): Le nom du fichier CSV
header (bool) : Le fichier CSV contient une entête
Retourne :
(list): Liste qui contient les valeurs du fichier
"""


def read_csv_file(csv_filename, header):
    items = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_count = 0
        if header == 1:
            skip = 1

        for row in reader:
            if skip != 1:
                split = row[0].split(',')
                items.append(split)
                row_count += 1
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
"""


def find_index(array, key):
    value = -1
    for i in range(0, len(array)):
        if key == array[i]:
            value = i
    return value


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


def compare():
    start_date = np.datetime64('1961-01-01')
    end_date = np.datetime64('2018-12-31')
    time = (end_date - start_date) / np.timedelta64(1, 'D')

    obs_data_filename = 'C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\A20_HYDOBS.nc'
    sim_data_file = 'C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\A20_HYDREP.nc'
    l10_data_file = 'C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\A20_ANALYS_DEBITJ_RESULTAT_VALIDATION_CROISEE_L1O.nc'
    station_validation_filename = "C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\stations_retenues_validation_croisee.csv"
    station_mapping_filename = "C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\Table_Correspondance_Station_Troncon.csv"

    print("Lecture des CSV")
    station_validation = read_csv_file(station_validation_filename, 1)
    station_mapping = read_csv_file(station_mapping_filename, 1)

    time_range = int(time)
    station_count = len(station_validation)
    debit_sim = np.empty((time_range, station_count))
    debit_obs = np.empty((time_range, station_count))
    debit_l1o = np.empty((time_range, station_count))

    print("Lecture des NC ids")
    stations_id = nc_read_char2string(obs_data_filename, 'station_id', 'nchar_station_id')
    sections_id = nc_read_char2string(sim_data_file, 'station_id', 'nchar_station_id')
    l1o_stations_id = nc_read_char2string(l10_data_file, 'station_id', 'nchar_station_id')

    print("Lecture des NC drainage")
    da_obs = nc_read(obs_data_filename, 'drainage_area')
    da_sim = nc_read(sim_data_file, 'drainage_area')
    da_l10 = nc_read(l10_data_file, 'drainage_area')

    print("Lecture des NC discharge")
    dis_obs = nc_read(obs_data_filename, 'Dis')
    dis_sim = nc_read(sim_data_file, 'Dis')
    dis_l10 = nc_read(l10_data_file, 'Dis')

    percentile = nc_read(l10_data_file, 'percentile')

    index_percentile = 0
    for i in range(len(percentile)):
        if percentile[i] == 50:
            index_percentile = i

    for i in range(0, station_count):
        print("Lecture des données..." + str(i + 1) + "/" + str(station_count))

        station_id = station_validation[i][0]
        associate_section = find_section(station_mapping, station_id)

        index_section = find_index(sections_id, associate_section)
        index_station = find_index(stations_id, station_id)
        index_station_l10 = find_index(l1o_stations_id, station_id)

        sup_sim = da_sim[index_section].item()
        sup_obs = da_obs[index_station].item()
        sup = da_l10[index_station_l10].item()

        data_values = dis_sim[index_section][0:time_range] / sup_sim
        debit_sim[:, i] = data_values.filled(np.nan)[:]

        data_values = dis_obs[index_station][0:time_range] / sup_obs
        debit_obs[:, i] = data_values.filled(np.nan)[:]

        data_values = dis_l10[index_percentile][index_station_l10][0:time_range] / sup
        debit_l1o[:, i] = data_values.filled(np.nan)[:]

    kge = np.empty(station_count)
    nse = np.empty(station_count)
    kge_l1o = np.empty(station_count)
    nse_l1o = np.empty(station_count)

    for n in range(0, station_count):
        kge[n] = kge_prime(debit_obs[:, n], debit_sim[:, n])
        nse[n] = nash(debit_obs[:, n], debit_sim[:, n])
        kge_l1o[n] = kge_prime(debit_obs[:, n], debit_l1o[:, n])
        nse_l1o[n] = nash(debit_obs[:, n], debit_l1o[:, n])

    fig, ax = plt.subplots()
    ax.scatter(kge, kge_l1o)
    ax.set_xlabel("KGE")
    ax.set_ylabel("KGE L10")
    ax.axline((0, 0), (1, 1), linewidth=2)
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0.3, 1)
    plt.show()
