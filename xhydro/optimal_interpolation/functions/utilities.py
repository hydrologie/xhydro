"""Utilities required for managing data in the interpolation toolbox."""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def read_csv_file(csv_filename):
    """
    Read values from a CSV file and return them as a list.

    Parameters:
    - csv_filename (str): The name of the CSV file to be read.

    Returns:
    list: A list containing the values from the CSV file.
    """
    items = []
    with open(csv_filename, newline="") as csvfile:
        line = csvfile.readline()
        if ";" in line:
            delimiter = ";"
        else:
            delimiter = ","
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar="|")
        for row in reader:
            items.append(row)

    return items


def find_index(array, key, value):
    """
    Find the index of an element in a list based on a specified key-value pair.

    Parameters:
    - array (list): List containing the data.
    - key (str): The key to identify the element in the list.
    - value (str): The value associated with the key to search for.

    Returns:
    int: Returns the index of the element in the list where the key-value pair matches.
         Returns -1 if the element is not found.
    """
    return np.where(array[key].data == value.encode("UTF-8"))[0][0]


def convert_list_to_dict(t):
    """
    Convert lists to dictionaries.

    This function takes a list of key-value pairs and converts it into a dictionary.

    Parameters:
    - t (list): List of key-value pairs.

    Returns:
    dict: A dictionary created from the input list."""
    return {k: v for k, v in t}


def initialize_nan_arrays(dimensions, percentiles):
    """
    Initialize arrays with NaN values for later population.

    This function preallocates arrays filled with NaN values to the correct size for later data population.

    Parameters:
    - dimensions (int): The size of each array dimension.
    - percentiles (int): The number of arrays to initialize, representing percentiles.

    Returns:
    tuple: A tuple of preallocated arrays, each initialized with NaN values.
    """
    t = [0] * percentiles
    for i in range(percentiles):
        t[i] = np.empty(dimensions)
        t[i][:] = np.nan
    return tuple(t)


def find_station_section(stations, section_id):
    """
    Find the section associated with a given station in the data tables.

    This function searches for the association of a section with a specific station in the provided list.

    Parameters:
    - stations (list): A list containing station information.
    - section_id (string): The identifier of the section.

    Returns:
    string: Returns an empty string if the section is not found, otherwise, returns the key
    representing the association between the station and a section.
    """
    value = ""
    section_position = 0
    section_value = 1
    for i in range(len(stations)):
        if section_id == stations[i][section_position]:
            value = stations[i]

    return value[section_value]


def load_files(files):
    """
    Load data from files containing Hydrotel runs and observations.

    Parameters:
    - files (list): A list of file paths to be loaded.

    Returns:
    list: A list containing the loaded data from the specified files."""
    extract_files = [0] * len(files)
    count = 0
    for filepath in files:
        file_name, file_extension = os.path.splitext(filepath)
        if file_extension == ".csv":
            extract_files[count] = read_csv_file(filepath)
        elif file_extension == ".nc":
            extract_files[count] = xr.open_dataset(filepath)
        count += 1
    return extract_files


def plot_results(kge, kge_l1o, nse, nse_l1o):
    """
    Plots the results of model evaluation using various metrics.

    Parameters:
    - kge (float): Kling-Gupta Efficiency for the entire dataset.
    - kge_l1o (float): Kling-Gupta Efficiency for leave-one-out cross-validation.
    - nse (float): Nash-Sutcliffe Efficiency for the entire dataset.
    - nse_l1o (float): Nash-Sutcliffe Efficiency for leave-one-out cross-validation.

    Returns:
    None

    """
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


def general_ecf(h, par, form):
    """
    Define the form of the Error Covariance Function (ECF) equations.

    Parameters:
    - h (float or array): The distance or distances at which to evaluate the ECF.
    - par (list): List of parameters for the ECF equation.
    - form (int): The form of the ECF equation to use (1, 2, or other).

    Returns:
    float or array: The calculated ECF values based on the specified form.

    """
    if form == 1:
        return par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
    elif form == 2:
        return par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
    else:
        return par[0] * np.exp(-h / par[1])


def write_netcdf_debit(
    write_file, station_id, lon, lat, drain_area, time, percentile, discharge
):
    """
    Write discharge data to a NetCDF file.

    Parameters
    ----------
    write_file : str
        Name of the NetCDF file to be created.
    station_id : list
        List of station IDs.
    lon : list
        List of longitudes corresponding to each station.
    lat : list
        List of latitudes corresponding to each station.
    drain_area : list
        List of drainage areas corresponding to each station.
    time : list
        List of datetime objects representing time.
    percentile : list or None
        List of percentiles or None if not applicable.
    discharge : numpy.ndarray
        3D array of discharge data, dimensions (percentile, station, time).

    Notes
    -----
    - The function creates a NetCDF file using the provided data and saves it with the specified filename.
    - If the file already exists, it is deleted before creating a new one.
    - The function includes appropriate metadata and attributes for each variable.


    Returns
    -------

    """

    if os.path.exists(write_file):
        os.remove(write_file)

    # Convert time to days since reference
    reference_time = datetime(1970, 1, 1)
    time = [(t - reference_time).days for t in time]

    # Create dataset
    ds = xr.Dataset()
    discharge = np.array(discharge)
    axis_time = np.where(np.array(discharge.shape) == len(time))
    axis_stations = np.where(np.array(discharge.shape) == len(drain_area))

    # Prepare discharge data
    if percentile:
        axis_percentile = np.where(np.array(discharge.shape) == len(percentile))
        ds["Dis"] = (
            ["percentile", "station", "time"],
            np.transpose(
                discharge, (axis_percentile[0][0], axis_stations[0][0], axis_time[0][0])
            ),
        )
        ds["percentile"] = ("percentile", percentile)
    else:
        ds["Dis"] = (
            ["station", "time"],
            np.transpose(discharge, (axis_stations[0][0], axis_time[0][0])),
        )

    # Other variables
    ds["time"] = ("time", time)
    ds["lat"] = ("station", lat)
    ds["lon"] = ("station", lon)
    ds["drainage_area"] = ("station", drain_area)
    ds["station_id"] = ("station", station_id)

    # Time bounds
    ta = np.array(time)
    time_bnds = np.array([ta - 1, time]).T
    ds["time_bnds"] = (("time", "nbnds"), time_bnds)

    # Set attributes
    ds["time"].attrs = {
        "long_name": "time",
        "standard_name": "time",
        "units": "days since 1970-01-01 -05:00:00",
        "calendar": "standard",
        "axis": "T",
        "bounds": "time_bnds",
    }
    ds["Dis"].attrs = {
        "long_name": "discharge",
        "standard_name": "discharge",
        "units": "m3/s",
        "cell_methods": "time: mean",
        "coverage_content_type": "modelResult",
        "coordinates": "time station_id",
    }
    ds["lat"].attrs = {
        "long_name": "latitude_of_river_stretch_outlet",
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    }
    ds["lon"].attrs = {
        "long_name": "longitude_of_river_stretch_outlet",
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }
    ds["time_bnds"].attrs = {
        "units": "days since 1970-01-01 -05:00:00",
        "calendar": "standard",
    }
    ds["drainage_area"].attrs = {
        "long_name": "drainage_area_at_river_stretch_outlet",
        "standard_name": "drainage_area",
        "units": "km2",
        "coverage_content_type": "auxiliaryInformation",
        "coordinates": "lat lon station_id",
    }
    ds["station_id"].attrs = {"long_name": "Station ID", "cf_role": "timeseries_id"}

    # Write to file
    ds.to_netcdf(write_file)
