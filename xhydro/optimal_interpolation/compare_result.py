import numpy as np
import xarray as xr
import sys
import functions.utilities as util
import functions.mathematical_algorithms as ma
import datetime as dt




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
def compare(percentileToPlot=50): #start_date, end_date, files,

    start_date = dt.datetime(2018, 11, 1)
    end_date = dt.datetime(2019, 1, 1)
    start_date = np.datetime64('1961-01-01')
    end_date = np.datetime64('2018-12-31')
    time = ((end_date - start_date) / np.timedelta64(1, 'D')) + 1

    obs_data_filename = 'data\\A20_HYDOBS.nc'
    sim_data_file = 'data\\A20_HYDREP.nc'
    l1o_data_file = 'data\\A20_ANALYS_DEBITJ_RESULTAT_VALIDATION_CROISEE_L1O.nc'
    station_validation_filename = "data\\stations_retenues_validation_croisee.csv"
    station_mapping_filename = "data\\Table_Correspondance_Station_Troncon.csv"

    print("Lecture des CSV")
    station_validation = util.read_csv_file(station_validation_filename)
    station_mapping = util.read_csv_file(station_mapping_filename)


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

    time_range = int(time)
    station_count = len(station_validation)
    debit_sim, debit_obs, debit_l1o = util.initialize_nan_arrays((time_range, station_count), 3)

    for i in range(0, station_count):
        print("Lecture des données..." + str(i + 1) + "/" + str(station_count))

        station_id = station_validation[i][0]
        associate_section = util.find_station_section(station_mapping, station_id)

        idx_section = util.find_index(sections_id, 'station_id', associate_section)
        idx_stat = util.find_index(stations_id, 'station_id', station_id)
        idx_stat_l1o = util.find_index(l1o_stations_id, 'station_id', station_id)

        sup_sim = da_sim[idx_section].item()
        sup_obs = da_obs[idx_stat].item()
        sup = da_l1o[idx_stat_l1o].item()

        debit_sim[:, i] = dis_sim[idx_section, 0:time_range].values[:] / sup_sim
        debit_obs[:, i] = dis_obs[idx_stat, 0:time_range].values[:] / sup_obs
        debit_l1o[:, i] = dis_l1o[idx_pct, idx_stat_l1o, 0:time_range].values[:] / sup

    kge, nse, kge_l1o, nse_l1o = util.initialize_nan_arrays(station_count, 4)

    for n in range(0, station_count):
        kge[n] = ma.kge_prime(debit_obs[:, n], debit_sim[:, n])
        nse[n] = ma.nash(debit_obs[:, n], debit_sim[:, n])
        kge_l1o[n] = ma.kge_prime(debit_obs[:, n], debit_l1o[:, n])
        nse_l1o[n] = ma.nash(debit_obs[:, n], debit_l1o[:, n])

    util.plot_results(kge, kge_l1o, nse, nse_l1o)

if __name__=="__main__":
    compare(50)