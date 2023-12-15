import numpy as np
import csv
import matplotlib.pyplot as plt
import xarray as xr
import sys

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
    logical = array.station_id.data == key.encode('UTF-8')
    return np.where(logical)[0][0]


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


"""
Code to plot results as per user request
"""
def plot_results(kge, kge_l1o, nse, nse_l1o):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(kge, kge_l1o)
    ax1.set_xlabel("KGE")
    ax1.set_ylabel("KGE Leave-one-out OI")
    ax1.axline((0, 0), (1, 1), linewidth=2)
    ax1.set_xlim(0.3, 1)
    ax1.set_ylim(0.3, 1)

    ax2.scatter(nse, nse_l1o)
    ax2.set_xlabel("NSE")
    ax2.set_ylabel("NSE Leave-one-out OI")
    ax2.axline((0, 0), (1, 1), linewidth=2)
    ax2.set_xlim(0.3, 1)
    ax2.set_ylim(0.3, 1)

    plt.show()


"""
Main code: Here is where we start the computation of the comparison method

TODO to make programmatic:
    1. Update compare function to take filepaths as inputs
    2. Add checks for data quality/error handling
    3. Add parameters to suit user needs (flexibility)
    4. Add parameter to ask for plot or not (user defined)
    5. Comment/document
    6. remove if __name__=="__main__" line and below to push to package.
    7. Eventually, refer to hydroeval package for the KGE and NSE metrics calculation.
    8. read_csv_files, find_index and find_section functions are duplicates between this and the "cross_validation" code, can be in a shared utils package
    9. Check to make sure files and indexes are in the correct order when reading
"""
def compare(percentileToPlot=50):
    start_date = np.datetime64('1961-01-01')
    end_date = np.datetime64('2018-12-31')
    time = ((end_date - start_date) / np.timedelta64(1, 'D')) + 1

    obs_data_filename = 'data\\A20_HYDOBS.nc'
    sim_data_file = 'data\\A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_MG24HS_GCQ_SC_18092020.nc'
    l1o_data_file = 'data\\A20_ANALYS_DEBITJ_RESULTAT_VALIDATION_CROISEE_L1O.nc'
    station_validation_filename = "data\\stations_retenues_validation_croisee.csv"
    station_mapping_filename = "data\\Table_Correspondance_Station_Troncon.csv"

    print("Lecture des CSV")
    station_validation = read_csv_file(station_validation_filename, 1)
    station_mapping = read_csv_file(station_mapping_filename, 1)

    time_range = int(time)
    station_count = len(station_validation)
    debit_sim = np.empty((time_range, station_count))
    debit_obs = np.empty((time_range, station_count))
    debit_l1o = np.empty((time_range, station_count))

    print("Lecture des NC")
    # Open the dataset for reading
    obs_data = xr.open_dataset(obs_data_filename)
    sim_data = xr.open_dataset(sim_data_file)
    l1o_data = xr.open_dataset(l1o_data_file)

    # Read station id
    stations_id = obs_data["station_id"]
    sections_id = sim_data["station_id"]
    l1o_stations_id = l1o_data["station_id"]

    # Read drainage area
    da_obs = obs_data.drainage_area
    da_sim = sim_data.drainage_area
    da_l1o = l1o_data.drainage_area

    # Read discharge percentiles
    dis_obs = obs_data.Dis
    dis_sim = sim_data.Dis
    dis_l1o = l1o_data.Dis

    # Read percentiles list (which percentile thresholds were used)
    percentile = l1o_data.percentile

    # Find position of the desired percentile
    idx_pct = np.where(percentile == percentileToPlot)[0]
    if idx_pct is None:
        sys.exit(
            "The desired percentile is not computed in the results file \
             provided. Please make sure your percentile value is expressed \
             in percent (i.e. 50th percentile = 50)"
                )

    for i in range(0, station_count):
        print("Lecture des données..." + str(i + 1) + "/" + str(station_count))

        station_id = station_validation[i][0]
        associate_section = find_section(station_mapping, station_id)

        idx_section = find_index(sections_id, associate_section)
        idx_stat = find_index(stations_id, station_id)
        idx_stat_l1o = find_index(l1o_stations_id, station_id)

        sup_sim = da_sim[idx_section].item()
        sup_obs = da_obs[idx_stat].item()
        sup = da_l1o[idx_stat_l1o].item()

        debit_sim[:, i] = dis_sim[idx_section, 0:time_range].values[:] / sup_sim
        debit_obs[:, i] = dis_obs[idx_stat, 0:time_range].values[:] / sup_obs
        debit_l1o[:, i] =  dis_l1o[idx_pct, idx_stat_l1o, 0:time_range].values[:] / sup

    kge = np.empty(station_count)
    nse = np.empty(station_count)
    kge_l1o = np.empty(station_count)
    nse_l1o = np.empty(station_count)

    for n in range(0, station_count):
        kge[n] = kge_prime(debit_obs[:, n], debit_sim[:, n])
        nse[n] = nash(debit_obs[:, n], debit_sim[:, n])
        kge_l1o[n] = kge_prime(debit_obs[:, n], debit_l1o[:, n])
        nse_l1o[n] = nash(debit_obs[:, n], debit_l1o[:, n])

    plot_results(kge, kge_l1o, nse, nse_l1o)

if __name__=="__main__":
    compare(50)
