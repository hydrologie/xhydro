import netCDF4
import numpy as np
import csv
import scipy.optimize
from scipy.stats import invgauss

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


    hB[:, :] = np.nan
    covB[:, :] = np.nan
    stdB[:, :] = np.nan

    for i in range(0, time_range):
        print(i)
        is_nan = np.isnan(ecart[i, :])
        ecart_jour = ecart[i, ~is_nan]

        tableau_de_1 = np.array([1 for i in range(0,len(ecart_jour))])

        if len(ecart_jour) >= 10:


            inanh = is_nan[0:len(distance)]
            inanv = inanh.reshape(1,len(distance))


            distancePC = distance[~inanh]
            distancePC = distancePC[:, ~inanv[0,:]]

            [h_b,cov_b,std_b,NP] = eval_covariance_bin(distancePC, ecart_jour, tableau_de_1, 1, input_opt, nbin)

            if len(NP[0]) >= 10:

                hB[i, :] = h_b[0, 0:11]
                covB[i, :] = cov_b[0, 0:11]
                stdB[i, :] = std_b[0, 0:11]

    distance = hB.T.reshape(len(hB) * len(hB[0]))
    Covariance = covB.T.reshape(len(covB) * len(covB[0]))
    poidsCovariance = 1 / np.power(stdB.T.reshape(len(stdB) * len(stdB[0])), 2)

    nc = nbin
    qt = np.linspace(0, 1, nc + 1)
    cl = np.unique(np.quantile(distance[~np.isnan(distance)], qt))
    nc = len(cl) - 1
    cov_b = np.zeros((nc))
    h_b = np.zeros((nc))
    std_b = np.zeros((nc))

    for i in range(0, nc):

        ind = np.where((distance >= cl[i]) & (distance < cl[i+1]))
        h_b[i] = np.mean(distance[ind])

        wt = poidsCovariance[ind] / np.sum(poidsCovariance[ind])

        cov_b[i] = np.sum(wt * Covariance[ind])
        average = np.average(Covariance[ind], weights=wt)
        variance = np.average((Covariance[ind] - average) ** 2, weights=wt)
        std_b[i] = np.sqrt(variance)

    hmax_divider = input_opt['hmax_divider']
    p1_bnds = input_opt['p1_bnds']
    hmax_mult_range_bnds = input_opt['hmax_mult_range_bnds']
    form = 3
    if form == 1:
        ecf_fun = lambda h, par: par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
    elif form == 2:
        ecf_fun = lambda h, par: par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
    else:
        ecf_fun = lambda h, par: par[0] * np.exp(-h / par[1])

    weight = 1 / np.power(std_b, 2)
    weight = weight / np.sum(weight)
    rmse_fun = lambda par: np.sqrt(np.mean(weight * np.power(ecf_fun(h_b, par) - cov_b, 2)))

    par_opt = scipy.optimize.minimize(rmse_fun, [np.mean(cov_b), np.mean(h_b)/3], \
                                      bounds=([p1_bnds[0], p1_bnds[1]], [0, hmax_mult_range_bnds[1]*500]))['x']
    error_cov_fun = lambda x: ecf_fun(x, par_opt)

    #
    # Faire graphique et sauvegarde
    #
    #
    #
    #
    #
    #

    return ecf_fun, par_opt

def ecf_fun(h, par, form=1):
     if form == 2:
         return par[0] * np.exp(-0.5 * np.power(h/par[1], 2))
     elif form == 3:
         return par[0] * np.exp(-h/par[1])
     else:
         return par[0]*(1+h/par[1]) * np.exp(-h/par[1])

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
    Covariance = Covariance[distance < hmx]
    distance = distance[distance < hmx]

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

        cov_b[:, i] = (np.sum(wt) / (np.power(np.sum(wt), 2) - np.sum(np.power(wt, 2))) * np.sum(wt * Covariance[ind])) / variance


        std_b[:, i] = np.sqrt(np.var(Covariance[ind]))
        NP[:, i] = len(ind[0])

    return h_b, cov_b, std_b, NP
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6371  # km

    lon = lon - lon0

    coslat = np.cos(np.deg2rad(lat))
    coslon = np.cos(np.deg2rad(lon))
    coslat0 = np.cos(np.deg2rad(lat0))

    sinlat = np.sin(np.deg2rad(lat))
    sinlon = np.sin(np.deg2rad(lon))
    sinlat0 = np.sin(np.deg2rad(lat0))

    x = R * coslat * sinlon
    y = -R * coslat*sinlat0*coslon+R*coslat0*sinlat

    return x, y

def optimal_interpolation_vBox(oi_input,preCalcul):
    if len(preCalcul) == 0:
        preCalcul = {}

    S = 1 #len(oi_input['x_est'])
    N = len(oi_input['x_obs'][0, :])
    oi_output = oi_input

    cond = 0

    if isinstance(preCalcul, dict):
        if 'x_obs' in preCalcul:
            cond = (np.array_equal(preCalcul['x_est'],oi_input['x_est']) \
                    and np.array_equal(preCalcul['y_est'],oi_input['y_est'])) \
                    and (np.array_equal(preCalcul['x_obs'],oi_input['x_obs']) \
                    and np.array_equal(preCalcul['y_obs'],oi_input['y_obs']))

    if cond == 0:
        Doo = np.zeros((N, N))
        for s1 in range(0, N):
            for s2 in range(0, N):
                eq1 = np.power(oi_input['x_obs'][:, s2] - oi_input['x_obs'][:, s1], 2)
                eq2 = np.power(oi_input['y_obs'][:, s2] - oi_input['y_obs'][:, s1], 2)
                Doo[s1, s2] = np.mean(np.sqrt(eq1 + eq2))

    else:
        Doo = preCalcul['Doo']

    preCalcul['x_obs'] = oi_input['x_obs']
    preCalcul['y_obs'] = oi_input['y_obs']
    preCalcul['Doo'] = Doo

    Coo = oi_input['error_cov_fun'](Doo) / oi_input['error_cov_fun'](0)

    BEo_j = np.tile(oi_input['bg_var_obs'], (N, 1))
    BEo_i = np.tile(np.resize(oi_input['bg_var_obs'], (1, N)), (N, 1))

    Bij = Coo * np.sqrt(BEo_j) / np.sqrt(BEo_i)

    OEo_j = np.tile(oi_input['var_obs'], (N, 1))
    OEo_i = np.tile(oi_input['var_obs'], (1, N))

    Oij = (np.sqrt(OEo_j) * np.sqrt(OEo_i)) * np.eye(len(OEo_j), len(OEo_j[0])) / BEo_i

    if cond == 0:
        Doe = np.zeros((S,N))

        for s1 in range(0, S):
            for s2 in range(0, N):
                eq1 = np.power(oi_input['x_obs'][:, s2] - oi_input['x_est'], 2)
                eq2 = np.power(oi_input['y_obs'][:, s2] - oi_input['y_est'], 2)

                Doe[s1, s2] = np.mean(np.sqrt(eq1 + eq2))
    else:
        Doe = preCalcul['Doe']

    preCalcul['x_est'] = oi_input['x_est']
    preCalcul['y_est'] = oi_input['y_est']
    preCalcul['Doe'] = Doe

    BEe = np.tile(np.resize(oi_input['bg_var_est'], (1, N)), (S, 1))
    BEo = np.tile(oi_input['bg_var_obs'], (S, 1))

    Coe = oi_input['error_cov_fun'](Doe) / oi_input['error_cov_fun'](0)

    Bei = np.resize(Coe * np.sqrt(BEe) / np.sqrt(BEo), (N, 1))

    weights = np.linalg.solve(Bij + Oij, Bei)

    I = np.resize(np.tile(oi_input['bg_departures'], (1, S)), (S, 1))

    t1 = weights * I
    t2 = np.sum(np.resize(weights * I, (S, 1)))
    oi_output['v_est'] = oi_input['bg_est'] + np.sum(weights * I)

    oi_output['var_est'] = oi_input['bg_var_est'] * np.diag(1 - Bei * weights)

    return oi_output, preCalcul


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

    lat0 = np.array([45]*len(centroide_lat))
    lon0 = np.array([-70]*len(centroide_lat))

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

        PX[:, i] = x_p.reshape(2*len(x_p))
        PY[:, i] = y_p.reshape(2*len(y_p))


    # Transformation log-débit pour l'interpolation
    qsim_log = np.log(debit_sim)
    qobs_log = np.log(debit_obs)

    ecf_fun,par_opt = ajustement_ECF_climatologique(qobs_log, qsim_log, PX, PY, "test")

    index = range(0, station_count)
    ratio_var_bg = 0.15
    qest_l1o = np.empty((time_range, station_count))
    qest_l1o_q25 = np.empty((time_range, station_count))
    qest_l1o_q75 = np.empty((time_range, station_count))

    qest_l1o[:, :] = np.nan
    qest_l1o_q25[:, :] = np.nan
    qest_l1o_q75[:, :] = np.nan


    for i in range(0, station_count):
        print(f'validation station croisé {i} de {station_count}')
        index_validation = i
        index_calibration = np.setdiff1d(index, i)

        ecart = qobs_log[:, index_calibration] - qsim_log[:, index_calibration]
        vsim_at_est = qsim_log[:, index_validation]
        oi_input = {}

        oi_input['var_obs'] = ratio_var_bg
        oi_input['error_cov_fun'] = lambda h : ecf_fun(h, par_opt)
        oi_input['x_est'] = PX[:, index_validation]
        oi_input['y_est'] = PY[:, index_validation]

        preCalcul = {}
        for j in range(0, time_range):
            if not np.isnan(debit_obs[j, index_validation]):
                val = ecart[j, :]
                idx = ~np.isnan(val)

                oi_input['x_obs'] = PX[:, index_calibration[idx]]
                oi_input['y_obs'] = PY[:, index_calibration[idx]]
                oi_input['bg_departures'] = ecart[j, idx]
                oi_input['bg_var_obs'] = np.ones(len(oi_input['bg_departures']))
                oi_input['bg_est'] = vsim_at_est[j]
                oi_input['bg_var_est'] = 1

                oi_output, preCalcul = optimal_interpolation_vBox(oi_input, preCalcul)
                qest_l1o[j, i] = np.exp(oi_output['v_est']) * superficie_drainee[i]

                var_bg = np.var(oi_input['bg_departures'])
                var_est = oi_output['var_est'] * var_bg

                t1 = invgauss.cdf(0.25, oi_output['v_est']) #, np.sqrt(var_est))

                qest_l1o_q25[j, i] = np.exp(np.percentile(np.random.normal(oi_output['v_est'], np.sqrt(var_est), 10000), 25)) * superficie_drainee[i]
                qest_l1o_q75[j, i] = np.exp(np.percentile(np.random.normal(oi_output['v_est'], np.sqrt(var_est), 10000), 75)) * superficie_drainee[i]
    test =1
