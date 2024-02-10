# -*- coding: utf-8 -*-
"""
This library is a regroupment of typical hydrology functions.

Functions included in the library:
    - PET_oudin
"""

import numpy as np
import pandas as pd
from numpy import arccos, cos, sin


def PET_oudin(latitude, doy, tas):
    """
    PET_oudin

    INPUTS:
        latitude -- latitude of the catchment in degree (°)
        doy -- a vector of the whole time period with the corresponding day of
               year (January 1st is 1 and December 31 is 365/366)
        tas -- mean temperature of the whole time period (°C)

    This fucntion allows to compute the potentiel evapotranspitation (PET)
    using a simplified version of the Oudin formula that only needs latitude,
    the day of year and mean daily temperature.
    
    This code is written based on the Excel spreadsheet of IRSTEA which can be
    found at http://webgr.irstea.fr/modeles/modele-devapotranspiration/?lang=en
    """
    # Compute the required parameters
    teta = 0.4093 * sin(doy/58.1 - 1.405)
    cosGz = np.maximum(0.001, cos(latitude / 57.3 - teta))
    cosOM = np.maximum(-1, np.minimum(
            1 - cosGz / cos(latitude / 57.3) / cos(teta),
            1)
            )
    OM = arccos(cosOM)
    Eta = 1 + cos(doy / 58.1) / 30
    cosPz = cosGz + cos(latitude / 57.3) * cos(teta) * (sin(OM) / OM - 1)
    rad = 446 * OM * Eta * cosPz  # in MJ/m2/day

    # Compute potential evapotranspiration (PET)
    PET = np.maximum(0, rad * (tas + 5) / 28.5 / 100)  # mm/day

    return PET


def remove_feb29(dates):
    """
    remove_feb29

    INPUTS:
        dates -- vector of dates from which to remove all February 29

    This function simply removes all February 29 from a given date vector.
    """

    leap = []
    for each in dates:
        if each.month == 2  and each.day == 29:
            leap.append(each)

    dates = dates.drop(leap)

    return dates


if __name__ == '__main__':
    # =========================================================================
    # EXAMPLE: PET_oudin
    # =========================================================================
    latitude = 48.73
    doy = np.arange(1, 366 + 1)
    tas = np.repeat(32, 366)
    PET = PET_oudin(latitude, doy, tas)
    # The result should be equal to 1400.1 mmm/year like the Excel spreadsheet
    print('The total annual potential evapotranspiration (PET) is: ' +
          str(np.round(PET.sum(), 1)) + ' mm.')

    # =========================================================================
    # EXAMPLE: remove_feb_29
    # =========================================================================
    dates = pd.date_range(start='2005-1-1', end='2014-12-31', freq='D')
    dates = remove_feb29(dates)
