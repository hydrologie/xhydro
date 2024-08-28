"""Test suite for hydrological modelling in hydrological_modelling.py"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xscen.testing import datablock_3d

from xhydro.modelling import (
    format_input,
    get_hydrological_model_inputs,
    hydrological_model,
)


class TestHydrologicalModelling:
    def test_hydrological_modelling(self):
        """Test the hydrological models as they become online"""
        # Test the dummy model
        model_config = {
            "precip": np.array([10, 11, 12, 13, 14, 15]),
            "temperature": np.array([10, 3, -5, 1, 15, 0]),
            "qobs": np.array([120, 130, 140, 150, 160, 170]),
            "drainage_area": np.array([10]),
            "model_name": "Dummy",
            "parameters": np.array([5, 5, 5]),
        }
        qsim = hydrological_model(model_config).run()
        np.testing.assert_array_equal(qsim["streamflow"].values[3], 3500.00)

    def test_import_unknown_model(self):
        """Test for unknown model"""
        with pytest.raises(NotImplementedError):
            model_config = {"model_name": "fake_model"}
            _ = hydrological_model(model_config).run()

    def test_missing_name(self):
        with pytest.raises(ValueError, match="The model name must be provided"):
            model_config = {"parameters": [1, 2, 3]}
            hydrological_model(model_config).run()


class TestHydrologicalModelRequirements:
    def test_get_unknown_model_requirements(self):
        """Test for required inputs for models with unknown name"""
        with pytest.raises(NotImplementedError):
            model_name = "fake_model"
            _ = get_hydrological_model_inputs(model_name)

    @pytest.mark.parametrize("model_name", ["Dummy", "Hydrotel", "GR4JCN"])
    def test_get_model_requirements(self, model_name):
        """Test for required inputs for models"""
        expected_keys = {"Dummy": (6, 6), "Hydrotel": (8, 3), "GR4JCN": (5, 5)}

        all_config, _ = get_hydrological_model_inputs(model_name)
        assert len(all_config.keys()) == expected_keys[model_name][0]

        all_config, _ = get_hydrological_model_inputs(model_name, required_only=True)
        assert len(all_config.keys()) == expected_keys[model_name][1]


class TestFormatInputs:
    @pytest.mark.parametrize("lons", ["180", "360"])
    def test_hydrotel(self, tmpdir, lons):
        ds = datablock_3d(
            np.array(
                np.tile(
                    [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
                    (365 * 3, 1, 1),
                )
            ),
            "tasmax",
            "lon",
            10 if lons == "180" else 190,
            "lat",
            15,
            30,
            30,
            as_dataset=True,
        )
        ds["tasmin"] = datablock_3d(
            np.array(
                np.tile(
                    [[8, 9, 10, 11, 12, 13], [8, 9, 10, 11, 12, 13]], (365 * 3, 1, 1)
                )
            ),
            "tasmin",
            "lon",
            10 if lons == "180" else 190,
            "lat",
            15,
            30,
            30,
        )
        ds["pr"] = datablock_3d(
            np.array(
                np.tile(
                    [
                        [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006],
                        [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006],
                    ],
                    (365 * 3, 1, 1),
                )
            ),
            "pr",
            "lon",
            10 if lons == "180" else 190,
            "lat",
            15,
            30,
            30,
        )
        # Add a z coordinate
        ds["z"] = xr.ones_like(ds["tasmax"].isel(time=0)).drop_vars("time") * 100
        ds = ds.assign_coords({"z": ds["z"]})
        ds["z"].attrs = {
            "units": "m",
            "long_name": "Elevation",
            "standard_name": "height",
            "axis": "Z",
        }

        ds_out, cfg = format_input(ds, "Hydrotel", save_as=tmpdir / "meteo.nc")

        ds_loaded = xr.open_dataset(tmpdir / "meteo.nc")
        # Time will differ when xarray reads the file
        assert (
            ds_out.isel(time=0)
            .drop_vars("time")
            .equals(ds_loaded.isel(time=0).drop_vars("time"))
        )
        assert Path.exists(tmpdir / "meteo.nc.config")

        assert cfg["TYPE (STATION/GRID/GRID_EXTENT)"] == "STATION"
        assert cfg["STATION_DIM_NAME"] == "station"
        assert cfg["LATITUDE_NAME"] == "lat"
        assert cfg["LONGITUDE_NAME"] == "lon"
        assert cfg["ELEVATION_NAME"] == "z"
        assert cfg["TIME_NAME"] == "time"
        assert cfg["TMIN_NAME"] == "tasmin"
        assert cfg["TMAX_NAME"] == "tasmax"
        assert cfg["PRECIP_NAME"] == "pr"

        assert "station" in ds_out.dims
        assert ("lon" not in ds_out.dims) and ("lon" in ds_out.coords)
        np.testing.assert_array_equal(
            ds_out.lon, np.tile([10, 40, 70, 100, 130, 160], 2)
        )

        assert len(ds_out.station) == len(ds.lon) * len(ds.lat)
        assert ds_out.tasmax.attrs["units"] == "°C"
        np.testing.assert_array_almost_equal(
            ds_loaded.isel(station=0).tasmax.values,
            ds.isel(lon=0, lat=0).tasmax.values - 273.15,
        )
        np.testing.assert_array_equal(
            ds_loaded.isel(station=0).tasmin.values,
            ds.isel(lon=0, lat=0).tasmin.values - 273.15,
        )
        assert ds_out.pr.attrs["units"] == "mm"
        np.testing.assert_array_equal(
            ds_loaded.isel(station=0).pr.values, ds.isel(lon=0, lat=0).pr.values * 86400
        )

        assert ds_out.time.attrs["units"] == "days since 1970-01-01 00:00:00"
        np.testing.assert_array_equal(ds_out.time[0], 11139)
        np.testing.assert_array_equal(
            ds_loaded.time[0], pd.Timestamp("2000-07-01").to_datetime64()
        )

    def test_badmodel(self):
        ds = xr.Dataset()
        with pytest.raises(
            NotImplementedError, match="The model 'BadModel' is not recognized."
        ):
            _ = format_input(ds, "BadModel")


class TestDummyModel:
    def test_inputs(self):
        model_config = {
            "model_name": "Dummy",
            "precip": np.array([10, 11, 12, 13, 14, 15]),
            "temperature": np.array([10, 3, -5, 1, 15, 0]),
            "qobs": np.array([120, 130, 140, 150, 160, 170]),
            "drainage_area": np.array([10]),
            "parameters": np.array([5, 5, 5]),
        }
        dummy = hydrological_model(model_config)
        ds_in = dummy.get_inputs()
        np.testing.assert_array_equal(ds_in.precip, model_config["precip"])
        np.testing.assert_array_equal(ds_in.temperature, model_config["temperature"])
        assert len(ds_in.time) == len(model_config["precip"])

    def test_streamflow(self):
        model_config = {
            "model_name": "Dummy",
            "precip": np.array([10, 11, 12, 13, 14, 15]),
            "temperature": np.array([10, 3, -5, 1, 15, 0]),
            "qobs": np.array([120, 130, 140, 150, 160, 170]),
            "drainage_area": np.array([10]),
            "parameters": np.array([5, 5, 5]),
        }
        dummy = hydrological_model(model_config)
        ds_out = dummy.get_streamflow()
        np.testing.assert_array_equal(ds_out["streamflow"].values[3], 3500.00)
        assert dummy.qsim.equals(ds_out)
        assert dummy.get_streamflow().equals(ds_out)
