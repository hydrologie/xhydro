"""Pytest configuration for xHydro tests."""

# noqa: D100
from os.path import commonpath
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries
from xclim.testing.utils import nimbus as _nimbus

from xhydro.testing.helpers import (
    TESTDATA_BRANCH,
    TESTDATA_CACHE_DIR,
    TESTDATA_REPO_URL,
)
from xhydro.testing.helpers import deveraux as _deveraux


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory):
    data_dir = Path(tmp_path_factory.getbasetemp().joinpath("data"))
    data_dir.mkdir(exist_ok=True)
    yield data_dir


@pytest.fixture(autouse=True, scope="session")
def deveraux(threadsafe_data_dir, worker_id) -> pooch.Pooch:
    return _deveraux(
        repo=TESTDATA_REPO_URL,
        branch=TESTDATA_BRANCH,
        cache_dir=(
            TESTDATA_CACHE_DIR if worker_id == "master" else threadsafe_data_dir
        ),
    )


@pytest.fixture(autouse=True, scope="session")
def nimbus(threadsafe_data_dir, worker_id) -> pooch.Pooch:
    kwargs = {}
    if worker_id != "master":
        kwargs["cache_dir"] = threadsafe_data_dir
    return _nimbus(**kwargs)


@pytest.fixture
def era5_example(nimbus):
    # Prepare a dataset with the required fields
    file = nimbus.fetch("ERA5/daily_surface_cancities_1990-1993.nc")
    ds = xr.open_dataset(file)[["huss", "pr", "snw"]]
    ds = ds.rename({"huss": "hus"})

    # Fake Geopotential field
    np.random.seed(42)
    zg = timeseries(
        np.random.rand(1461) * 1000,
        variable="geopotential",
        start="1990-01-01",
        freq="D",
    ).expand_dims(location=ds.location)
    zg.attrs = {
        "units": "m",
        "long_name": "Geopotential Height",
        "standard_name": "geopotential_height",
    }
    ds["zg"] = zg

    # Expand the dataset to have a 3D field
    ds = xr.concat(
        [
            ds.expand_dims(plev=[1000]),
            (ds * 1.1).expand_dims(plev=[1500]),
            (ds * 1.2).expand_dims(plev=[2000]),
        ],
        dim="plev",
    )
    ds["plev"].attrs = {
        "units": "Pa",
        "long_name": "pressure level",
        "standard_name": "air_pressure",
        "axis": "Z",
        "positive": "down",
    }

    # Fake orography field
    ds["orog"] = xr.DataArray(
        np.array([800, 100, 95, 25, 450]),
        dims=["location"],
        coords={"location": ds.location},
        attrs={
            "units": "m",
            "long_name": "Orography",
            "standard_name": "surface_altitude",
        },
    )
    return ds


@pytest.fixture(scope="session")
def oi_data(deveraux):

    # Get data with pooch
    oi_data = "optimal_interpolation/OI_data.zip"
    test_data_path = deveraux.fetch(oi_data, pooch.Unzip())
    common = Path(commonpath(test_data_path))

    # Correct files to get them into the correct shape.
    df = pd.read_csv(common / "Info_Station.csv", sep=None, dtype=str, engine="python")
    qobs = xr.open_dataset(common / "A20_HYDOBS_TEST.nc")
    qobs = qobs.assign(
        {"centroid_lat": ("station", df["Latitude Centroide BV"].astype(np.float32))}
    )
    qobs = qobs.assign(
        {"centroid_lon": ("station", df["Longitude Centroide BV"].astype(np.float32))}
    )
    qobs = qobs.assign({"classement": ("station", df["Classement"].astype(np.float32))})
    qobs = qobs.assign(
        {"station_id": ("station", qobs["station_id"].values.astype(str))}
    )
    qobs = qobs.assign({"streamflow": (("station", "time"), qobs["Dis"].values)})

    df = pd.read_csv(
        common / "Correspondance_Station.csv", sep=None, dtype=str, engine="python"
    )
    station_correspondence = xr.Dataset(
        {
            "reach_id": ("station", df["troncon_id"]),
            "station_id": ("station", df["No.Station"]),
        }
    )

    qsim = xr.open_dataset(common / "A20_HYDREP_TEST.nc")
    qsim = qsim.assign(
        {"station_id": ("station", qsim["station_id"].values.astype(str))}
    )
    qsim = qsim.assign({"streamflow": (("station", "time"), qsim["Dis"].values)})
    qsim["station_id"].values[
        143
    ] = "SAGU99999"  # Forcing to change due to double value wtf.
    qsim["station_id"].values[
        7
    ] = "BRKN99999"  # Forcing to change due to double value wtf.

    flow_l1o = xr.open_dataset(
        common / "A20_ANALYS_FLOWJ_RESULTS_CROSS_VALIDATION_L1O_TEST.nc"
    )
    flow_l1o = flow_l1o.assign(
        {"station_id": ("station", flow_l1o["station_id"].values.astype(str))}
    )
    flow_l1o = flow_l1o.assign(
        {"streamflow": (("percentile", "station", "time"), flow_l1o["Dis"].values)}
    )
    tt = flow_l1o["time"].dt.round(freq="D")
    flow_l1o = flow_l1o.assign_coords(time=tt.values)

    # Load data
    df_validation = pd.read_csv(
        common / "stations_retenues_validation_croisee.csv",
        sep=None,
        dtype=str,
        engine="python",
    )
    data = {
        "flow_l1o": flow_l1o,
        "observation_stations": list(df_validation["No_station"]),
        "qobs": qobs,
        "qsim": qsim,
        "station_correspondence": station_correspondence,
    }

    return data


@pytest.fixture(scope="session")
def corrected_oi_data(deveraux):

    # Get data with pooch
    oi_data = "optimal_interpolation/OI_data_corrected.zip"
    test_data_path = deveraux.fetch(oi_data, pooch.Unzip())
    common = Path(commonpath(test_data_path))

    # Load data
    df_validation = pd.read_csv(
        common / "stations_retenues_validation_croisee.csv",
        sep=None,
        dtype=str,
        engine="python",
    )
    data = {
        "flow_l1o": xr.open_dataset(
            common / "A20_ANALYS_FLOWJ_RESULTS_CROSS_VALIDATION_L1O_TEST_corrected.nc"
        ),
        "observation_stations": list(df_validation["No_station"]),
        "qobs": xr.open_dataset(common / "A20_HYDOBS_TEST_corrected.nc"),
        "qsim": xr.open_dataset(common / "A20_HYDREP_TEST_corrected.nc"),
        "station_correspondence": xr.open_dataset(common / "station_correspondence.nc"),
    }

    return data


@pytest.fixture(scope="session")
def data_fre(threadsafe_data_dir, deveraux):
    paths_sea = deveraux.fetch(
        "extreme_value_analysis/sea-levels_fremantle.zarr.zip",
        processor=pooch.Unzip(),
    )

    ds_sea = xr.open_zarr(Path(paths_sea[0]).parents[0]).compute()

    return ds_sea


@pytest.fixture(scope="session")
def data_rain(threadsafe_data_dir, deveraux):
    paths_rain = deveraux.fetch(
        "extreme_value_analysis/rainfall_excedance.zarr.zip",
        processor=pooch.Unzip(),
    )

    ds_rain = xr.open_zarr(Path(paths_rain[0]).parents[0]).compute()

    return ds_rain
