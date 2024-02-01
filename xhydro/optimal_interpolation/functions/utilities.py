"""Utilities required for managing data in the interpolation toolbox."""
import csv

import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr

def read_csv_file(csv_filename):
    """Retourne une liste qui contient les valeurs d'un fichier CSV.

    Arguments :
    csv_filename (string): Le nom du fichier CSV
    header (bool) : Le fichier CSV contient une entête
    Retourne :
    (list): Liste qui contient les valeurs du fichier
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
    """Find the index of an element in a list.

    Arguments :
    array (list): Liste qui contient les données
    value (string) : Élément à trouver dans la liste
    Retourne :
    (float): -1 si l'élément est introuvable ou l'indice de l'élément
    """
    return np.where(array[key].data == value.encode("UTF-8"))[0][0]


def convert_list_to_dict(t):
    """Convert lists to dictionnaries."""
    return {k: v for k, v in t}


def initialize_nan_arrays(dimensions, percentiles):
    """Preallocate arrays to nan of correct size for populating later."""
    t = [0] * percentiles
    for i in range(percentiles):
        t[i] = np.empty(dimensions)
        t[i][:] = np.nan
    return tuple(t)


def find_station_section(stations, section_id):
    """Find which section in the data tables corresponds to the station.

    Trouve l'association d'une section à une station.
    Arguments :
    stations (list): Liste qui contient les stations
    section_id (string) : L'indentificateur de la section
    Retourne :
    (string): Vide si la section est introuvable ou la clé d'association entre
    la station et une section.
    """
    value = ""
    section_position = 0
    section_value = 1
    for i in range(len(stations)):
        if section_id == stations[i][section_position]:
            value = stations[i]

    return value[section_value]

def load_files(files):
    """Load the files that contain the Hydrotel runs and observations."""
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
    """Code to plot results as per user request."""
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
    """Define the form of the ECF equations.

    We will use functools.partial to define functions instead of lambdas as it is
    more efficient and will allow parallelization. Therefore, define the ECF
    function shape here. New function.
    """
    if form == 1:
        return par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
    elif form == 2:
        return par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
    else:
        return par[0] * np.exp(-h / par[1])
