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
Modifié pas mal pour vectoriser.
"""


def find_index(array, key):
    logical = array.station_id.data == key.encode('UTF-8')
    return np.where(logical)[0][0]
