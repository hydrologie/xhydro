"""Pytest configuration for xHydro tests."""

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries
from xclim.testing.utils import open_dataset as _open_dataset


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory) -> Path:
    yield Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def open_dataset(threadsafe_data_dir):
    # FIXME: This is a temporary fix against the latest xclim-testdata release. It should be removed once xclim itself is updated.
    def _open_session_scoped_file(
        file: str | os.PathLike, branch: str = "v2023.12.14", **xr_kwargs
    ):
        xr_kwargs.setdefault("engine", "h5netcdf")
        return _open_dataset(
            file, cache_dir=threadsafe_data_dir, branch=branch, **xr_kwargs
        )

    return _open_session_scoped_file


@pytest.fixture
def era5_example(open_dataset):
    # Prepare a dataset with the required fields
    ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")[
        ["huss", "pr", "snw"]
    ]
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
