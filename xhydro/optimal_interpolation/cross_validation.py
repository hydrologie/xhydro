import netCDF4
import numpy as np
import csv
import utm

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
"""


def find_index(array, key):
    value = -1
    for i in range(0, len(array)):
        if key == array[i]:
            value = i
    return value


def ajustement_ECF_climatologique(debit_obs, debit_sim, PX, PY, savename):
    ecart = debit_sim - debit_obs

    station_count = np.shape(PX)[1]
    time_range = np.shape(ecart)[0]

    distance = np.empty((station_count, station_count))

    for i in range(0, station_count):
        for j in range(0, station_count):
            distance[i, j] = np.mean(np.sqrt(np.power(PX[:, j] - PX[:, i], 2) + np.power(PY[:, j] - PY[:, i], 2)))

    input_opt = {}
    input_opt['hmax_divider'] = 2
    input_opt['p1_bnds'] = [0.95, 1]
    input_opt['hmax_mult_range_bnds'] = [0.05, 3]

    nbin = 10
    hB = np.empty((time_range, nbin))
    covB = np.empty((time_range, nbin))
    stdB = np.empty((time_range, nbin))

    for i in range(0, time_range):
        inan = np.isnan(ecart[i, :])
        ecart_jour = ecart[i, ~inan]

        tableau_de_1 = np.array([1 for i in range(0,len(ecart_jour))])

        if len(ecart_jour) >= 10:


            inanh = inan[0:len(distance)]
            inanv = inanh.reshape(1,len(distance))


            distancePC = distance[~inanh]
            distancePC = distancePC[:, ~inanv[0,:]]
            [h_b,cov_b,std_b,NP] = eval_covariance_bin(distancePC, ecart_jour,tableau_de_1,1,input_opt,nbin)


def eval_covariance_bin(distance, val, err, form, input_opt=None, nbin=None):

    if nbin is not None:
        hmax_divider = input_opt['hmax_divider']
        p1_bnds = input_opt['p1_bnds']
        hmax_mult_range_bnds = input_opt['hmax_mult_range_bnds']
    elif input_opt is not None:
        hmax_divider = input_opt['hmax_divider']
        p1_bnds = input_opt['p1_bnds']
        hmax_mult_range_bnds = input_opt['hmax_mult_range_bnds']
        nbin=20
    else:
        hmax_divider = 2
        p1_bnds = [0, 1]
        hmax_mult_range_bnds = [0, 10]
        nbin = 20

    ndata = len(val)
    nh = np.sum([k for k in range(1, ndata)])
    poids = np.power(1 / err, 2)
    poids = poids / np.sum(poids)

    moyennePonderee = np.sum(val * poids) / np.sum(poids)
    variance = np.var(val, ddof=1)

    d = val - moyennePonderee

    Covariance = d * d[:, np.newaxis]
    poidsCovariance = poids * poids[:, np.newaxis]

    Covariance = Covariance.reshape(ndata*ndata)
    poidsCovariance = poidsCovariance.reshape(ndata*ndata)

    #Remplacer par le code PX
    distance = distance.reshape(len(distance)*len(distance))

    hmx = max(distance) / hmax_divider
    Covariance = Covariance[distance > hmx]
    distance = distance[distance > hmx]

    nc = nbin
    qt = [(1/nc)*i for i in range(0, nc + 1)]
    cl = np.unique(np.quantile(distance, qt))
    nc = len(cl) - 1

    cov_b = np.empty((1, nc))
    h_b = np.empty((1, nc))
    std_b = np.empty((1, nc))
    NP = np.empty((1, nc))

    cov_b[:, :] = np.nan
    h_b[:, :] = np.nan
    std_b[:, :] = np.nan
    NP[:, :] = np.nan


    for i in range(0, nc):

        ind = np.where((distance >= cl[i].item()) & (cl[i+1].item() > distance))
        h_b[:, i] = np.mean(distance[ind])

        wt = poidsCovariance[ind] / np.sum(poidsCovariance[ind])

        cov_b[:, i] = (np.sum(wt) / (np.power(np.sum(wt), 2) - np.sum(np.power(wt, 2)) * np.sum(wt * Covariance[ind]))) / variance
        std_b[:, i] = np.sqrt(np.var(Covariance[ind] * wt, ddof=1))
        NP[:, i] = np.sum(ind)

    return h_b, cov_b, std_b, NP
def latlon_to_xy(lat, lon):
    R = 6371  # km
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)

    return x, y


def execute():
    start_date = np.datetime64('1961-01-01')
    end_date = np.datetime64('2018-12-31')
    time = (end_date - start_date) / np.timedelta64(1, 'D')

    obs_data_filename = 'C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\A20_HYDOBS.nc'
    sim_data_file = 'C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\A20_HYDREP.nc'

    station_validation_filename = "C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\stations_retenues_validation_croisee.csv"
    station_mapping_filename = "C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\Table_Correspondance_Station_Troncon.csv"
    station_info_filename = "C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\Table_Info_Station_Hydro_2020.csv"

    print("Lecture des CSV")
    stations_validation = read_csv_file(station_validation_filename, 1, ',')
    stations_mapping = read_csv_file(station_mapping_filename, 1, ',')
    stations_info = read_csv_file(station_info_filename, 1, ";")

    print("Lecture des NC ids")
    stations_id = nc_read_char2string(obs_data_filename, 'station_id', 'nchar_station_id')
    sections_id = nc_read_char2string(sim_data_file, 'station_id', 'nchar_station_id')

    print("Lecture des NC drainage")
    da_obs = nc_read(obs_data_filename, 'drainage_area')
    da_sim = nc_read(sim_data_file, 'drainage_area')

    print("Lecture des NC discharge")
    dis_obs = nc_read(obs_data_filename, 'Dis')
    dis_sim = nc_read(sim_data_file, 'Dis')

    time_range = int(time)
    station_count = len(stations_validation)
    debit_sim = np.empty((time_range, station_count))
    debit_obs = np.empty((time_range, station_count))
    centroide_lat = np.empty(station_count)
    centroide_lon = np.empty(station_count)
    superficie_drainee = np.empty(station_count)
    longitude_station = np.empty(station_count)
    latitude_station = np.empty(station_count)
    PX = np.empty((time_range, station_count))
    PY = np.empty((time_range, station_count))

    for i in range(0, station_count):
        print("Lecture des données..." + str(i + 1) + "/" + str(station_count))

        station_id = stations_validation[i][0]
        associate_section = find_section(stations_mapping, station_id)

        index_section = find_index(sections_id, associate_section)
        index_station = find_index(stations_id, station_id)

        sup_sim = da_sim[index_section].item()
        sup_obs = da_obs[index_station].item()

        superficie_drainee[i] = sup_obs

        data_values = dis_sim[index_section][0:time_range] / sup_sim
        debit_sim[:, i] = data_values.filled(np.nan)[:]

        data_values = dis_obs[index_station][0:time_range] / sup_obs
        debit_obs[:, i] = data_values.filled(np.nan)[:]

        position_info = np.where(np.array(stations_info) == station_id)
        station_info = stations_info[position_info[0].item()]
        centroide_lat[i] = station_info[5]
        centroide_lon[i] = station_info[4]
        longitude_station[i] = station_info[3]
        latitude_station[i] = station_info[2]


    x, y = latlon_to_xy(centroide_lat, centroide_lon)  # Projete dans un plan pour avoir des distances en km

    for i in range(station_count):
        L = np.sqrt(superficie_drainee[i])
        xv = [x[i] - (L / 2) * x[i] + L / 2]
        yv = [y[i] - (L / 2) * y[i] + L / 2]
        [Xp, Yp] = np.meshgrid(xv, yv)

        PX[:, i] = Xp.reshape(len(Xp))
        PY[:, i] = Yp.reshape(len(Yp))


    # Transformation log-débit pour l'interpolation
    qsim_log = np.log(debit_sim)
    qobs_log = np.log(debit_obs)

    # PX = np.array([[165.5620, 169.3326, 358.0216, 306.8524, 187.99811],
    #                [165.5620, 169.3326, 358.0216, 306.8524, 187.99811],
    #                [189.1205, 221.8016, 383.0416, 341.4934, 228.67980],
    #                [189.1205, 221.8016, 383.0416, 341.4934, 228.67980]])
    #
    # PY = np.array([[385.3479, 354.8049, 447.9909, 438.2225, 394.4463],
    #                [408.9064, 407.2739, 473.0109, 472.8635, 435.1280],
    #                [385.3479, 354.8049, 447.9909, 438.2225, 394.4463],
    #                [408.9064, 407.2739, 473.0109, 472.8635, 435.1280]])

    ajustement_ECF_climatologique(qobs_log, qsim_log, PX, PY, "test")

    for i in range(0, station_count):
        ew = 1
