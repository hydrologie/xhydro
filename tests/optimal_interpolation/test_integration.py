import xhydro.optimal_interpolation.cross_validation as cv
from xhydro.optimal_interpolation import constants
import numpy as np
from pytest import approx

def test_cross_validation_execute():
    start_date = '2018-01-01'
    end_date = '2019-01-01'
    files = [
        constants.DATA_PATH + "Table_Info_Station_Hydro_2020.csv",
        constants.DATA_PATH + "Table_Correspondance_Station_Troncon.csv",
        constants.DATA_PATH + "stations_retenues_validation_croisee.csv",
        constants.DATA_PATH + 'A20_HYDOBS.nc',
        constants.DATA_PATH + 'A20_HYDREP.nc'
    ]

    result_flows = cv.execute(start_date, end_date, files, False)

    # Test some output flow values
    assert result_flows[0][-1, 3] == 10.13226473752605
    assert result_flows[0][-2, 4] == 14.197267554780344

    # To randomize to test direct values
    assert np.nanmean(result_flows[1][:, :]) == approx(33, 0.5)
    assert np.nanmean(result_flows[2][:, :]) == approx(59, 0.5)

    # Test the time range duration
    assert len(result_flows[0]) == 365
    assert len(result_flows[1]) == 365
    assert len(result_flows[2]) == 365

    # Test a different data range to verify that the last entry is different
    start_date = '2015-01-01'
    end_date = '2019-01-01'

    result_flows = cv.execute(start_date, end_date, files, False)

    # Test some output flow values
    assert result_flows[0][-1, 3] == 19.49652141729676
    assert result_flows[0][-1, 4] == 29.557635690780714

    # To randomize to test direct values
    assert np.nanmean(result_flows[1][:, :]) == approx(52, 0.5)
    assert np.nanmean(result_flows[2][:, :]) == approx(94, 0.5)

    # Test the time range duration
    assert len(result_flows[0]) == 1461
    assert len(result_flows[1]) == 1461
    assert len(result_flows[2]) == 1461

def test_cross_validation_execute_parralelize():
    start_date = '2018-01-01'
    end_date = '2019-01-01'
    files = [
        constants.DATA_PATH + "Table_Info_Station_Hydro_2020.csv",
        constants.DATA_PATH + "Table_Correspondance_Station_Troncon.csv",
        constants.DATA_PATH + "stations_retenues_validation_croisee.csv",
        constants.DATA_PATH + 'A20_HYDOBS.nc',
        constants.DATA_PATH + 'A20_HYDREP.nc'
    ]

    result_flows = cv.execute(start_date, end_date, files)

    # Test some output flow values
    assert result_flows[0][-1, 3] == 10.13226473752605
    assert result_flows[0][-2, 4] == 14.197267554780344
