import numpy as np

import xhydro.optimal_interpolation.mathematical_algorithms as ma


def test_calculate_average_distance():
    """Test calculating average distance between points."""
    # Between 2 points
    x_points = np.array([[0, 3]])
    y_points = np.array([[0, 4]])

    # Calculate distance
    result = ma.calculate_average_distance(x_points, y_points)

    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0, 0] == 0
    assert result[0, 1] == 5
    assert result[1, 0] == 5
    assert result[1, 1] == 0

    # Test between 2 sets of vectors
    x_points = np.array([[0, 1], [0, 1]])
    y_points = np.array([[0, 1], [0, 1]])

    # Calculate distance
    result = ma.calculate_average_distance(x_points, y_points)

    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0, 0] == 0
    np.testing.assert_almost_equal(result[0, 1], 1.4142135623730951, 10)
    np.testing.assert_almost_equal(result[1, 0], 1.4142135623730951, 10)
    assert result[1, 1] == 0

    # Test between 2 sets of points in 3D
    x_points = np.array([[1, 2, 3], [4, 5, 6]])
    y_points = np.array([[1, 2, 3], [4, 5, 6]])

    # Calculate distance
    result = ma.calculate_average_distance(x_points, y_points)

    assert len(result) == 3
    assert len(result[0]) == 3
    assert result[0, 0] == 0
    np.testing.assert_almost_equal(result[0, 1], 1.4142135623730951, 10)
    np.testing.assert_almost_equal(result[0, 2], 2.8284271247461903, 10)
    np.testing.assert_almost_equal(result[1, 0], 1.4142135623730951, 10)
    assert result[1, 1] == 0
    np.testing.assert_almost_equal(result[1, 2], 1.4142135623730951, 10)
    np.testing.assert_almost_equal(result[2, 0], 2.8284271247461903, 10)
    np.testing.assert_almost_equal(result[2, 1], 1.4142135623730951, 10)
    assert result[2, 2] == 0


def test_latlon_to_xy():
    """Test conversion between latitude longitude and x-y coordinates."""
    lat = 10
    lon = -45

    # Do the conversion
    result = ma.latlon_to_xy(lat, lon)
    np.testing.assert_almost_equal(result[0], -4436.536575078841, 4)
    np.testing.assert_almost_equal(result[1], 1106.312539916013, 4)

    # Test using a reference
    lat = 0
    lon = 0
    lat0 = 10
    lon0 = -45

    # Do the conversion
    result = ma.latlon_to_xy(lat, lon, lat0=lat0, lon0=lon0)
    np.testing.assert_almost_equal(result[0], 4504.977302939495, 4)
    np.testing.assert_almost_equal(result[1], -782.281099086326, 4)
