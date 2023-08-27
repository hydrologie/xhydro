import netCDF4
import numpy as np
import csv
import matplotlib.pyplot as plt

def nc_read(ncfile, varid):
    data = netCDF4.Dataset(ncfile)
    return data.variables[varid]
def nc_read_char2string(ncfile, varid, nchar_dimid):
    data = netCDF4.Dataset(ncfile)
    vars = data.variables[varid]
    result = []
    index = -1

    if vars.dimensions.count(nchar_dimid) > 0:
        index = vars.dimensions.index(nchar_dimid)

    if index > 0:
        for i in range(0, vars.shape[0]):
            result.append(''.join(str(vars[i, :], encoding='utf-8')))
    return result


def nc_readtime(ncfile, timevar_id):

    result = []
    data = netCDF4.Dataset(ncfile)
    times = data.variables[timevar_id]
    time_units = times.units

    regex = time_units.split()

    for i in range(0, times.shape[0]):
        delta_time = np.timedelta64()

        match regex[0]:
            case "days":
                delta_time = np.timedelta64(int(np.ma.getdata(times[i]).item()), 'D')
            case "hours":
                delta_time = np.timedelta64(int(np.ma.getdata(times[i]).item()/24), 'D')
            case "minutes":
                delta_time = np.timedelta64(int(np.ma.getdata(times[i]).item()/(24*60)), 'D')

        date = np.datetime64(regex[2] + "T" + regex[3]) + delta_time

        result.append(date)

    return result

def read_csv_file(csv_file_name, header):
    items = []
    with open(csv_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_count = 0
        if header == 1:
            skip=1

        for row in reader:
            if skip != 1:
                split = row[0].split(',')
                items.append(split)
                row_count += 1
            else:
                skip=0

    return items

def find(array, match, position):
    value = -1
    for i in range (0,len(array)):
        test = array[i][position]
        if match == array[i][position]:
            value = array[i]
    return value

def find_index(array, match):
    value = -1
    for i in range (0,len(array)):
        if match == array[i]:
            value = i
    return value

def KGE_prime(obs, sim):

    isNan = np.isnan(obs) | np.isnan(sim)

    obs = np.ma.array(obs, mask=isNan)
    sim = np.ma.array(sim, mask=isNan)

    obs = obs[~isNan]
    sim = sim[~isNan]

    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)

    obs_std = np.std(obs)
    sim_std = np.std(sim)

    r = np.corrcoef(obs, sim)[0, 1]

    beta = sim_mean / obs_mean
    gamma = (sim_std / sim_mean) / (obs_std / obs_mean)

    return 1 - np.sqrt(np.power((r - 1), 2) + np.power((beta - 1), 2) + np.power((gamma - 1), 2))

def nash(obs, sim):
    sim = np.ma.array(sim, mask=np.isnan(obs))
    obs = np.ma.array(obs, mask=np.isnan(obs))

    SSE = np.sum(np.power(obs - sim, 2))
    SSU = np.sum(np.power(obs - np.mean(obs), 2))
    return 1 - SSE / SSU

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

    NT = int(time)
    NS = len(station_validation)
    debit_sim = np.empty((NT, NS))
    debit_sim[:] = np.nan
    debit_obs = np.empty((NT, NS))
    debit_obs[:] = np.nan
    debit_l1o = np.empty((NT, NS))
    debit_l1o[:] = np.nan
    # debit_l1o_q25 = np.empty((NT, NS))
    # debit_sim[:] = np.nan
    # debit_l1o_q75 = np.empty((NT, NS))
    # debit_sim[:] = np.nan

    print("Lecture des NC ids")
    nc_station_id = nc_read_char2string(obs_data_filename, 'station_id', 'nchar_station_id')
    nc_troncon_id = nc_read_char2string(sim_data_file, 'station_id', 'nchar_station_id')
    nc_l1o_station_id = nc_read_char2string(l10_data_file, 'station_id', 'nchar_station_id')
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

    for i in range(0, NS):
        print("Lecture des données..." + str(i+1) + "/" + str(NS))

        station_id = station_validation[i][0]
        associate_section = find(station_mapping, station_id, 0)

        index_section = find_index(nc_troncon_id, associate_section[1])
        sup_sim = da_sim[index_section].item()
        data_values = dis_sim[index_section][0:NT] / sup_sim
        debit_sim[:, i] = data_values.filled(np.nan)[:]

        index_station = find_index(nc_station_id, station_id)
        sup_obs = da_obs[index_station].item()
        data_values = dis_obs[index_station][0:NT] / sup_obs
        debit_obs[:, i] = data_values.filled(np.nan)[:]

        index_station_l10 = find_index(nc_l1o_station_id, station_id)
        sup = da_l10[index_station_l10].item()

        data_values = dis_l10[index_percentile][index_station_l10][0:NT] / sup
        debit_l1o[:, i] = data_values.filled(np.nan)[:]

    kge = np.empty(NS)
    nse = np.empty(NS)
    kge_l1o = np.empty(NS)
    nse_l1o = np.empty(NS)

    for n in range(0, NS):
        kge[n] = KGE_prime(debit_obs[:, n], debit_sim[:, n])
        nse[n] = nash(debit_obs[:, n], debit_sim[:, n])
        kge_l1o[n] = KGE_prime(debit_obs[:, n], debit_l1o[:, n])
        nse_l1o[n] = nash(debit_obs[:, n], debit_l1o[:, n])

    fig, ax = plt.subplots()
    ax.scatter(kge, kge_l1o)
    ax.set_xlabel("KGE")
    ax.set_ylabel("KGE L10")
    ax.axline((0, 0), (1, 1), linewidth=2)
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0.3, 1)
    plt.show()

def test():

    obs_data_file = 'C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\A20_HYDOBS.nc'
    # nc_station_id = nc_read_char2string(obs_data_file, 'station_id', 'nchar_station_id')
    test = nc_read(obs_data_file, 'Dis')[5][0:22]/55
    test2 = test[0].item()
    debit_obs = np.empty((22, 22))
    debit_obs[:] = np.nan
    debit_obs[:,0] = test
    compare()

    # l10_data_file = 'C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\A20_ANALYS_DEBITJ_RESULTAT_VALIDATION_CROISEE_L1O.nc'
    #
    # sup = nc_read(l10_data_file, 'drainage_area')[0].item()
    # test = nc_read(l10_data_file, 'Dis')[0][4][0:21183] / sup

    # sub_obs = nc_read(obs_data_file, 'Dis')[7]/55
    # table_station_validation = read_csv_file('C:\\Users\\AR21010\\Documents\\GitHub\\xhydro\\xhydro\\optimal_interpolation\\stations_retenues_validation_croisee.csv',1);

    x=1

