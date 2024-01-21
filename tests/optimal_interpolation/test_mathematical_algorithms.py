import xhydro.optimal_interpolation.functions.mathematical_algorithms as ma
import numpy as np

def test_calculate_average_distance():
    x_points = np.array([[0, 3]])
    y_points = np.array([[0, 4]])

    result = ma.calculate_average_distance(x_points, y_points)

    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0, 0] == 0
    assert result[0, 1] == 5
    assert result[1, 0] == 5
    assert result[1, 1] == 0


    x_points = np.array([[0, 1], [0, 1]])
    y_points = np.array([[0, 1], [0, 1]])

    result = ma.calculate_average_distance(x_points,y_points)

    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0,0] == 0
    assert result[0,1] == 1.4142135623730951
    assert result[1,0] == 1.4142135623730951
    assert result[1,1] == 0

    x_points = np.array([[1, 2, 3], [4, 5, 6]])
    y_points = np.array([[1, 2, 3], [4, 5, 6]])

    result = ma.calculate_average_distance(x_points, y_points)

    assert len(result) == 3
    assert len(result[0]) == 3
    assert result[0, 0] == 0
    assert result[0, 1] == 1.4142135623730951
    assert result[0, 2] == 2.8284271247461903
    assert result[1, 0] == 1.4142135623730951
    assert result[1, 1] == 0
    assert result[1, 2] == 1.4142135623730951
    assert result[2, 0] == 2.8284271247461903
    assert result[2, 1] == 1.4142135623730951
    assert result[2, 2] == 0


def test_latlon_to_xy():
    lat = 10
    lon = -45

    result = ma.latlon_to_xy(lat, lon)
    assert result[0] == -4436.536575078841
    assert result[1] == 1106.312539916013

    lat = 0
    lon = 0
    lat0 = 10
    lon0 = -45

    result = ma.latlon_to_xy(lat, lon, lat0, lon0)
    assert result[0] == 4504.977302939495
    assert result[1] == -782.281099086326
