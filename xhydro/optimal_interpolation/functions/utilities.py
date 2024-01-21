import numpy as np
import csv

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
