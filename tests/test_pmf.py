import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xhydro.pmf.pmf import (
    fix_pmp_year,
    maximize_snow,
    place_pmf_inputs,
    remove_precip,
    separate_pr,
)


@pytest.fixture
def sample_data():
    # Create sample xarray Dataset for testing
    data = xr.Dataset(
        {
            "precipitation": (("time", "lat", "lon"), np.random.rand(20, 5, 5)),
            "temperature_min": (("time", "lat", "lon"), np.random.rand(20, 5, 5)),
            "temperature_max": (("time", "lat", "lon"), np.random.rand(20, 5, 5) + 2),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=20),
            "lat": np.linspace(-10, 10, 5),
            "lon": np.linspace(-10, 10, 5),
        },
    )
    return data


@pytest.fixture
def sample_pmp():
    # Create sample xarray Dataset for testing
    data = xr.Dataset(
        {
            "precipitation": (("time", "lat", "lon"), np.random.rand(5, 5, 5) + 22),
            "temperature_min": (("time", "lat", "lon"), np.random.rand(5, 5, 5) + 10),
            "temperature_max": (("time", "lat", "lon"), np.random.rand(5, 5, 5) + 15),
        },
        coords={
            "time": pd.date_range("2020-01-05", periods=5),
            "lat": np.linspace(-10, 10, 5),
            "lon": np.linspace(-10, 10, 5),
        },
    )
    return data


@pytest.fixture
def sample_data_with_catchments():
    # Create sample xarray Dataset for testing with two catchments and two centers
    data = xr.Dataset(
        {
            "precipitation": (
                ("time", "catchment", "center"),
                np.random.rand(10, 2, 2),
            ),
            "temperature_min": (
                ("time", "catchment", "center"),
                np.random.rand(10, 2, 2),
            ),
            "temperature_max": (
                ("time", "catchment", "center"),
                np.random.rand(10, 2, 2) + 2,
            ),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=10),
            "catchment": ["ws_1", "ws_2"],
            "center": ["c1", "c2"],
        },
    )
    return data


def test_remove_precip(sample_data):
    time = pd.date_range(start="2020-01-03", periods=3, freq="D")
    result = remove_precip(sample_data.precipitation, time)
    assert result.sel(time=time).sum() == 0


def test_remove_precip_multi_dim(sample_data_with_catchments):
    time = pd.date_range(start="2020-01-03", periods=3, freq="D")
    result = remove_precip(sample_data_with_catchments.precipitation, time)
    assert result.sel(time=time).sum() == 0  # Add specific assertions based on expected behavior


def test_separate_pr(sample_data):
    result = separate_pr(sample_data, "precipitation", "temperature_min", "temperature_max")
    #  check for precipitation_snow
    assert "precipitation_snow" in result
    assert "precipitation_rain" in result
    assert result.precipitation_rain.attrs["t_trans"] == 0
    assert "DINGMAN " in result.precipitation_rain.attrs["history"]


def test_separate_pr_ubc(sample_data):
    result = separate_pr(sample_data, "precipitation", "temperature_min", "temperature_max", algo="UBC")
    assert result.precipitation_rain.attrs["t_trans"] == 0
    assert "UBC " in result.precipitation_rain.attrs["history"]
    assert result.precipitation_rain.attrs["delta_t"] == 4


def test_fix_pmp_year(sample_data):
    result = fix_pmp_year(sample_data["precipitation"], 2020)
    assert all(result.time.dt.year == 2020)


def test_place_pmf_inputs_multidim(sample_pmp, sample_data):
    result = place_pmf_inputs(sample_pmp, sample_data)
    assert (result.precipitation.sel(time=sample_pmp.time) >= 20).all()
    assert (result.temperature_min.sel(time=sample_pmp.time) >= 10).all()
    assert (result.temperature_max.sel(time=sample_pmp.time) >= 15).all()
    assert (result.precipitation.drop_sel(time=sample_pmp.time) <= 1).all()
    assert (result.temperature_min.drop_sel(time=sample_pmp.time) <= 1).all()
    assert (result.temperature_max.drop_sel(time=sample_pmp.time) <= 3).all()


def test_place_pmf_inputs_multidim_size_one(sample_pmp, sample_data):
    result = place_pmf_inputs(sample_pmp.isel(lon=0, lat=0), sample_data.isel(lon=0, lat=0))
    assert (result.precipitation.sel(time=sample_pmp.time) >= 20).all()
    assert (result.temperature_min.sel(time=sample_pmp.time) >= 10).all()
    assert (result.temperature_max.sel(time=sample_pmp.time) >= 15).all()
    assert (result.precipitation.drop_sel(time=sample_pmp.time) <= 1).all()
    assert (result.temperature_min.drop_sel(time=sample_pmp.time) <= 1).all()
    assert (result.temperature_max.drop_sel(time=sample_pmp.time) <= 3).all()


def test_place_pmf_inputs_simgledim(sample_pmp, sample_data):
    result = place_pmf_inputs(sample_pmp.isel(lon=0, lat=0).squeeze(), sample_data.isel(lon=0, lat=0).squeeze())
    assert (result.precipitation.sel(time=sample_pmp.time) >= 20).all()
    assert (result.temperature_min.sel(time=sample_pmp.time) >= 10).all()
    assert (result.temperature_max.sel(time=sample_pmp.time) >= 15).all()
    assert (result.precipitation.drop_sel(time=sample_pmp.time) <= 1).all()
    assert (result.temperature_min.drop_sel(time=sample_pmp.time) <= 1).all()
    assert (result.temperature_max.drop_sel(time=sample_pmp.time) <= 3).all()


def test_maximize_snow(sample_data):
    result = maximize_snow(sample_data, {}, 2020, 0.5)
    assert result is not None  # Add specific assertions based on expected behavior
