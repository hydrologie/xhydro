import numpy as np
import pytest
import xarray as xr

from xhydro.PMP import (
    calculate_PMP,
    calculate_PMP_ensemble,
    calculate_PMP_ensemble_stats,
    hershfield_PMP,
)


def test_hershfield_PMP():
    xm = 100
    sx = 20
    km = 15
    expected_PMP = 400
    result = hershfield_PMP(xm, sx, km)
    assert np.isclose(result, expected_PMP, rtol=1e-5)


def test_calculate_PMP():
    precip = xr.DataArray(
        np.random.rand(10, 5, 5),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2000-01-01", periods=10),
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 5),
        },
    )
    result = calculate_PMP(precip)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("lat", "lon")
    assert np.all(result.values >= precip.max("time").values)


def test_calculate_PMP_ensemble():
    ensemble = xr.DataArray(
        np.random.rand(3, 10, 5, 5),
        dims=["realization", "time", "lat", "lon"],
        coords={
            "realization": [1, 2, 3],
            "time": pd.date_range("2000-01-01", periods=10),
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 5),
        },
    )
    result = calculate_PMP_ensemble(ensemble)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("realization", "lat", "lon")
    assert np.all(result.values >= ensemble.max("time").values)


def test_calculate_PMP_ensemble_stats():
    ensemble = xr.DataArray(
        np.random.rand(3, 10, 5, 5),
        dims=["realization", "time", "lat", "lon"],
        coords={
            "realization": [1, 2, 3],
            "time": pd.date_range("2000-01-01", periods=10),
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 5),
        },
    )
    result = calculate_PMP_ensemble_stats(ensemble)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"mean", "std", "min", "max"}
    assert all(var.dims == ("lat", "lon") for var in result.data_vars.values())


@pytest.mark.parametrize("km", [-1, 0, 100])
def test_hershfield_PMP_edge_cases(km):
    xm = 100
    sx = 20
    result = hershfield_PMP(xm, sx, km)
    assert np.isfinite(result)
    assert result >= xm


def test_calculate_PMP_constant_precip():
    precip = xr.DataArray(
        np.full((10, 5, 5), 100),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2000-01-01", periods=10),
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 5),
        },
    )
    result = calculate_PMP(precip)
    assert np.all(result.values == 100)


def test_calculate_PMP_ensemble_single_realization():
    ensemble = xr.DataArray(
        np.random.rand(1, 10, 5, 5),
        dims=["realization", "time", "lat", "lon"],
        coords={
            "realization": [1],
            "time": pd.date_range("2000-01-01", periods=10),
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 5),
        },
    )
    result = calculate_PMP_ensemble(ensemble)
    assert result.dims == ("realization", "lat", "lon")
    assert result.shape == (1, 5, 5)


def test_calculate_PMP_ensemble_stats_extreme_values():
    ensemble = xr.DataArray(
        np.random.rand(3, 10, 5, 5) * 1000,
        dims=["realization", "time", "lat", "lon"],
        coords={
            "realization": [1, 2, 3],
            "time": pd.date_range("2000-01-01", periods=10),
            "lat": np.linspace(0, 1, 5),
            "lon": np.linspace(0, 1, 5),
        },
    )
    result = calculate_PMP_ensemble_stats(ensemble)
    assert np.all(result.max.values >= result.mean.values)
    assert np.all(result.min.values <= result.mean.values)
    assert np.all(result.std.values >= 0)
