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

    @pytest.mark.parametrize("model_name", ["Hydrotel", "GR4JCN"])
    def test_get_model_requirements(self, model_name):
        """Test for required inputs for models"""
        expected_keys = {"Hydrotel": (8, 4), "GR4JCN": (5, 5)}

        all_config, _ = get_hydrological_model_inputs(model_name)
        assert len(all_config.keys()) == expected_keys[model_name][0]

        all_config, _ = get_hydrological_model_inputs(model_name, required_only=True)
        assert len(all_config.keys()) == expected_keys[model_name][1]


class TestFormatInputs:
    # Create a dataset with a few issues:
    # tasminnn instead of tasmin, but attributes are fine
    # precip and z have no standard_name, but a recognized variable names
    ds_bad = datablock_3d(
        np.array(
            np.tile(
                [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
                (365 * 3, 1, 1),
            )
        ),
        "tasmax",
        "lon",
        10,
        "lat",
        15,
        30,
        30,
        start="2000-01-01",
        as_dataset=True,
    )
    ds_bad["tasminnn"] = datablock_3d(
        np.array(
            np.tile([[8, 9, 10, 11, 12, 13], [8, 9, 10, 11, 12, 13]], (365 * 3, 1, 1))
        ),
        "tasmin",
        "lon",
        10,
        "lat",
        15,
        30,
        30,
        start="2000-01-01",
    )
    ds_bad["precip"] = datablock_3d(
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
        10,
        "lat",
        15,
        30,
        30,
        start="2000-01-01",
    )
    ds_bad["precip"].attrs = {"units": "kg m-2 s-1"}
    ds_bad["precip"] = ds_bad["precip"].where(ds_bad["precip"] > 0.0001)
    # Add an elevation coordinate
    ds_bad["z"] = xr.ones_like(ds_bad["tasmax"].isel(time=0)).drop_vars("time") * 100
    ds_bad = ds_bad.assign_coords({"z": ds_bad["z"]})
    ds_bad["z"].attrs = {
        "units": "m",
    }

    # Create a rotated dataset with a few issues
    ds_bad_rotated = datablock_3d(
        np.array(
            np.tile(
                [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
                (365 * 3, 1, 1),
            )
        ),
        "tasmax",
        "rlon",
        10,
        "rlat",
        15,
        30,
        30,
        start="2000-01-01",
        as_dataset=True,
    )

    ds_bad_rotated["tasminnn"] = datablock_3d(
        np.array(
            np.tile([[8, 9, 10, 11, 12, 13], [8, 9, 10, 11, 12, 13]], (365 * 3, 1, 1))
        ),
        "tasmin",
        "rlon",
        10,
        "rlat",
        15,
        30,
        30,
        start="2000-01-01",
    )

    ds_bad_rotated["precip"] = datablock_3d(
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
        "rlon",
        10,
        "rlat",
        15,
        30,
        30,
        start="2000-01-01",
    )

    ds_bad_rotated["precip"].attrs = {"units": "kg m-2 s-1"}
    ds_bad_rotated["precip"] = ds_bad_rotated["precip"].where(
        ds_bad_rotated["precip"] > 0.0001
    )

    # Add an elevation coordinate
    ds_bad_rotated["z"] = (
        xr.ones_like(ds_bad_rotated["tasmax"].isel(time=0)).drop_vars("time") * 100
    )
    ds_bad_rotated = ds_bad_rotated.assign_coords({"z": ds_bad_rotated["z"]})
    ds_bad_rotated["z"].attrs = {"units": "m"}

    @pytest.mark.parametrize(("lons", "ds"), [("180", ds_bad), ("360", ds_bad_rotated)])
    def test_hydrotel(self, tmpdir, lons, ds):
        ds = ds.copy()
        if lons == "360":
            with xr.set_options(keep_attrs=True):
                ds["lon"] = ds["lon"] + 180

        ds_out, cfg = format_input(ds, "Hydrotel", save_as=tmpdir / "meteo.nc")

        ds_loaded = xr.open_dataset(tmpdir / "meteo.nc")
        # Time will differ when xarray reads the file
        assert (
            ds_out.isel(time=0)
            .drop_vars("time")
            .equals(ds_loaded.isel(time=0).drop_vars("time"))
        )
        assert Path(tmpdir / "meteo.nc.config").is_file()

        assert cfg["TYPE (STATION/GRID/GRID_EXTENT)"] == "STATION"
        assert cfg["STATION_DIM_NAME"] == "station"
        assert cfg["LATITUDE_NAME"] == "lat"
        assert cfg["LONGITUDE_NAME"] == "lon"
        assert cfg["ELEVATION_NAME"] == "orog"
        assert cfg["TIME_NAME"] == "time"
        assert cfg["TMIN_NAME"] == "tasmin"
        assert cfg["TMAX_NAME"] == "tasmax"
        assert cfg["PRECIP_NAME"] == "pr"

        assert "station" in ds_out.dims
        assert ("lon" not in ds_out.dims) and ("lon" in ds_out.coords)
        if lons == "180":
            np.testing.assert_array_equal(
                ds_out.lon, np.tile([40, 70, 100, 130, 160], 2)
            )
            assert len(ds_out.station) == len(ds.lon) * len(ds.lat) - 2
            np.testing.assert_array_almost_equal(
                ds_loaded.isel(station=1).tasmax.values,
                ds.isel(lon=2, lat=0).tasmax.values - 273.15,
            )
            np.testing.assert_array_equal(
                ds_loaded.isel(station=1).tasmin.values,
                ds.isel(lon=2, lat=0).tasminnn.values - 273.15,
            )
            np.testing.assert_array_equal(
                ds_loaded.isel(station=1).pr.values,
                ds.isel(lon=2, lat=0).precip.values * 86400,
            )
        else:
            np.testing.assert_array_almost_equal(
                ds_out.lon,
                [
                    -33.464092,
                    -9.042494,
                    10.730389,
                    32.514941,
                    60.663006,
                    11.873653,
                    21.311321,
                    33.950654,
                    49.820166,
                    69.003495,
                ],
            )
            np.testing.assert_array_almost_equal(
                ds_loaded.isel(station=1).tasmax.values,
                ds.isel(rlon=2, rlat=0).tasmax.values - 273.15,
            )
            np.testing.assert_array_equal(
                ds_loaded.isel(station=1).tasmin.values,
                ds.isel(rlon=2, rlat=0).tasminnn.values - 273.15,
            )
            np.testing.assert_array_equal(
                ds_loaded.isel(station=1).pr.values,
                ds.isel(rlon=2, rlat=0).precip.values * 86400,
            )
        assert ds_out.tasmax.attrs["units"] == "degC"
        assert ds_out.pr.attrs["units"] == "mm"

        assert ds_out.time.attrs["units"] == "days since 1970-01-01 00:00:00"
        np.testing.assert_array_equal(ds_out.time[0], 10957)
        np.testing.assert_array_equal(
            ds_loaded.time[0], pd.Timestamp("2000-01-01").to_datetime64()
        )

    def test_hydrotel_calendars(self, tmpdir):
        ds = self.ds_bad.copy()
        ds = ds.convert_calendar("365_day")

        with pytest.raises(
            ValueError,
            match="is not supported by Hydrotel.",
        ):
            ds_out, _ = format_input(ds, "Hydrotel", convert_calendar_missing=False)
        with pytest.warns(
            UserWarning,
            match="NaNs will need to be filled manually",
        ):
            ds_out, _ = format_input(ds, "Hydrotel")
        np.testing.assert_array_equal(ds_out.tasmin.isel(station=1, time=59), np.nan)
        np.testing.assert_array_equal(ds_out.tasmin.isel(station=1).isnull().sum(), 1)

        ds_out2, _ = format_input(ds, "Hydrotel", convert_calendar_missing=999)
        np.testing.assert_array_equal(ds_out2.tasmin.isel(station=1, time=59), 999)
        np.testing.assert_array_equal(ds_out2.tasmin.isel(station=1).isnull().sum(), 0)

        ds_out3, _ = format_input(
            ds,
            "Hydrotel",
            convert_calendar_missing={"tasmin": "interpolate", "tasmax": 999, "pr": 0},
        )
        np.testing.assert_array_equal(
            ds_out3.tasmin.isel(station=1, time=59),
            np.mean(
                [
                    ds_out3.tasmin.isel(station=1, time=58),
                    ds_out3.tasmin.isel(station=1, time=60),
                ]
            ),
        )
        np.testing.assert_array_equal(ds_out3.tasmax.isel(station=1, time=59), 999)
        np.testing.assert_array_equal(ds_out3.pr.isel(station=1, time=59), 0)
        np.testing.assert_array_equal(ds_out3.tasmin.isel(station=1).isnull().sum(), 0)

        assert format_input(
            ds,
            "Hydrotel",
            convert_calendar_missing={
                "tasmin": "interpolate",
                "tasmax": "interpolate",
                "pr": 0,
            },
        )[0].equals(
            format_input(
                ds,
                "Hydrotel",
                convert_calendar_missing=True,
            )[0]
        )

    def test_hydrotel_error(self):
        ds = self.ds_bad.copy()
        ds = ds.drop_vars("tasmax")
        with pytest.raises(
            ValueError, match="The dataset is missing the following required variables"
        ):
            _ = format_input(ds, "Hydrotel")

    @pytest.mark.parametrize(("lons", "ds"), [("180", ds_bad), ("360", ds_bad_rotated)])
    def test_raven(self, tmpdir, lons, ds):
        ds = ds.copy()

        if lons == "360":
            with xr.set_options(keep_attrs=True):
                ds["lon"] = ds["lon"] + 180

            # Change temperature to tmean
            ds = ds.rename({"tasmax": "tmean"})
            ds["tmean"].attrs = {
                "standard_name": "air_temperature",
                "cell_methods": "time: mean",
                "units": "K",
            }
            ds = ds.drop_vars(["tasminnn"])

        ds_out, cfg = format_input(ds, "HBVEC", save_as=tmpdir / "meteo.nc")

        ds_loaded = xr.open_dataset(tmpdir / "meteo.nc")

        assert ds_out.equals(ds_loaded)

        if "rlon" in ds.dims:
            spatial = ["rlon", "rlat"]
            assert tuple(ds_out.tas.sizes.keys()) == (*spatial, "time")
        else:
            spatial = ["lon", "lat"]
            assert tuple(ds_out.tasmin.sizes.keys()) == (*spatial, "time")
            assert tuple(ds_out.tasmax.sizes.keys()) == (*spatial, "time")
        assert tuple(ds_out.pr.sizes.keys()) == (*spatial, "time")

        if "rlon" in ds.dims:
            assert ("rlon" in ds_out.dims) and ("rlon" in ds_out.coords)
            assert ("rlat" in ds_out.dims) and ("rlat" in ds_out.coords)
            assert ("lon" not in ds_out.dims) and ("lon" in ds_out.coords)
            assert ("lat" not in ds_out.dims) and ("lat" in ds_out.coords)
            np.testing.assert_array_equal(
                ds_out.rlon, np.tile([10, 40, 70, 100, 130, 160], 1)
            )
        else:
            assert ("lon" in ds_out.dims) and ("lon" in ds_out.coords)
            assert ("lat" in ds_out.dims) and ("lat" in ds_out.coords)
            np.testing.assert_array_equal(
                ds_out.lon, np.tile([10, 40, 70, 100, 130, 160], 1)
            )

        if lons == "180":
            assert ds_out.tasmin.attrs["units"] == "degC"
            assert ds_out.tasmax.attrs["units"] == "degC"
            np.testing.assert_array_almost_equal(
                ds_loaded.isel(lon=2, lat=0).tasmax.values,
                ds.isel(lon=2, lat=0).tasmax.values - 273.15,
            )
            np.testing.assert_array_equal(
                ds_loaded.isel(lon=2, lat=0).tasmin.values,
                ds.isel(lon=2, lat=0).tasminnn.values - 273.15,
            )
            np.testing.assert_array_equal(
                ds_loaded.isel(lon=2, lat=0).pr.values,
                ds.isel(lon=2, lat=0).precip.values * 86400,
            )
        else:
            assert ds_out.tas.attrs["units"] == "degC"
            np.testing.assert_array_almost_equal(
                ds_loaded.isel(rlon=2, rlat=0).tas.values,
                ds.isel(rlon=2, rlat=0).tmean.values - 273.15,
            )
            np.testing.assert_array_equal(
                ds_loaded.isel(rlon=2, rlat=0).pr.values,
                ds.isel(rlon=2, rlat=0).precip.values * 86400,
            )

        assert ds_out.pr.attrs["units"] == "mm"

        np.testing.assert_array_equal(
            ds_loaded.time[0], pd.Timestamp("2000-01-01").to_datetime64()
        )

    def test_raven_1dspatial(self):
        ds = self.ds_bad.copy()
        ds = ds.stack({"station": ("lon", "lat")})
        ds = ds.drop_vars(["lon", "lat"]).reset_coords()
        ds = ds.assign_coords(
            {"station": np.arange(len(ds.station)), "lon": 0, "lat": 0}
        )

        with pytest.raises(
            ValueError,
            match="The dataset does not contain a dimension with t",
        ):
            format_input(ds, "HBVEC")

        ds["station"].attrs["cf_role"] = "timeseries_id"
        ds_out, _ = format_input(ds, "HBVEC", save_as=None)

        assert all(
            var in ds_out
            for var in ["station", "time", "tasmax", "tasmin", "pr", "orog"]
        )
        assert "time" not in ds_out.orog.dims

        ds = ds.isel(station=0).expand_dims("station")
        ds["station"].attrs = {}
        ds_out2, _ = format_input(ds, "HBVEC", save_as=None)
        assert all(var in ds_out2 for var in ["time", "tasmax", "tasmin", "pr", "orog"])
        assert "station" not in ds_out2.dims

    def test_raven_hbvec_calendars(self, tmpdir):
        ds = self.ds_bad_rotated.copy()
        ds = ds.convert_calendar("365_day")

        ds_out, _ = format_input(ds, "HBVEC", convert_calendar_missing=False)
        assert ds_out.time.dt.calendar == "noleap"

    def test_raven_hbvec_error(self):
        ds = self.ds_bad_rotated.copy()
        ds = ds.drop_vars("tasmax")
        with pytest.raises(
            ValueError, match="The dataset is missing the required variables"
        ):
            _ = format_input(ds, "HBVEC")

    def test_badmodel(self):
        ds = xr.Dataset()
        with pytest.raises(
            NotImplementedError, match="The model 'BadModel' is not recognized."
        ):
            _ = format_input(ds, "BadModel")
