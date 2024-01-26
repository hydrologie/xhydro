import numpy as np
import csv
import matplotlib.pyplot as plt

"""
Retourne une liste qui contient les valeurs d'un fichier CSV
Arguments :
csv_filename (string): Le nom du fichier CSV
header (bool) : Le fichier CSV contient une entête
Retourne :
(list): Liste qui contient les valeurs du fichier
"""


def read_csv_file(csv_filename):
    items = []
    with open(csv_filename, newline='') as csvfile:
        line = csvfile.readline()
        if ';' in line:
            delimiter = ';'
        else:
            delimiter = ','
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in reader:
            items.append(row)

    return items

"""
Trouve l'indince d'un élément donné dans une liste.
Arguments :
array (list): Liste qui contient les données
value (string) : Élément à trouver dans la liste
Retourne :
(float): -1 si l'élément est introuvable ou l'indice de l'élément
Modifié pas mal pour vectoriser.
"""


def find_index(array, key, value):
    return np.where(array[key].data == value.encode('UTF-8'))[0][0]

def convert_list_to_dict(t):
    return dict((k, v) for k, v in t)

def initialize_nan_arrays(dimensions, quantities):
    t = [0] * quantities
    for i in range(quantities):
        t[i] = np.empty(dimensions)
        t[i][:] = np.nan
    return tuple(t)


"""
Trouve l'association d'une section à une station.
Arguments :
stations (list): Liste qui contient les stations
section_id (string) : L'indentificateur de la section
Retourne :
(string): Vide si la section est introuvable ou la clé d'association entre la station et une section.
"""
def find_station_section(stations, section_id):
    value = ""
    section_position = 0
    section_value = 1
    for i in range(len(stations)):
        if section_id == stations[i][section_position]:
            value = stations[i]
    return value[section_value]


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
We will use functools.partial to define functions instead of lambdas as it is
more efficient and will allow parallelization. Therefore, define the ECF
function shape here. New function.
"""


def general_ecf(h, par, form):
    if form == 1:
        return par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
    elif form == 2:
        return par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
    else:
        return par[0] * np.exp(-h / par[1])
