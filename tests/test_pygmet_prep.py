import tempfile

import numpy as np
import pandas as pd
import xarray as xr
from numpy.ma.testutils import assert_almost_equal

from xhydro.pygmet.make_toml_config_pygmet import write_config_toml
from xhydro.pygmet.make_toml_settings_pygmet import write_settings_toml
from xhydro.pygmet.subsample_vector_format_stations import isel_every_about_n
from xhydro.pygmet.transform_to_station_order import convert_2d_nc_to_1d_stations, make_target_pygmet_grid


def test_make_pygmet_config():
    params = {
        "case_name": "My_case_01",
        "num_processes": 10,
        "modelsettings_file": "model.settings_xhydro.toml",
        "input_stn_all": "./stations.nc",
        "infile_grid_domain": "./grid_domain.nc",
        "outpath_parent": "./output_dir",
        "date_start": "1970-01-01",
        "date_end": "2024-12-31",
        "input_vars": ["precip", "tmin", "tmax"],
        "target_vars": ["precip", "tmin", "tmax"],
        "target_vars_WithProbability": ["precip"],
        "minRange_vars": [0.0, -70.0, -70.0],
        "maxRange_vars": [250.0, 70.0, 70.0],
        "transform_vars": ["boxcox", "", ""],
        "predictor_name_static_stn": ["latitude", "longitude"],
        "predictor_name_static_grid": ["latitude", "longitude"],
        "ensemble_end": 100,
        "clen": [150, 800, 800],  # codespell:ignore clen
        "auto_corr_method": ["direct", "anomaly", "anomaly"],
        "target_vars_max_constrain": ["precip"],
    }
    tempout = tempfile.mkdtemp()
    outpath = tempout + "/model.config_xhydro.toml"
    write_config_toml(outpath, params, strict=True)


def test_make_pygmet_settings():
    params = {
        "stn_lat_name": "latitude",
        "stn_lon_name": "longitude",
        "grid_lat_name": "latitude",
        "grid_lon_name": "longitude",
        "grid_mask_name": "mask",
        "dynamic_grid_lat_name": "latitude",
        "dynamic_grid_lon_name": "longitude",
        "nearstn_min": 2,
        "nearstn_max": 16,
        "try_radius": 200,
        "initial_distance": 200,
    }
    tempout = tempfile.mkdtemp()
    outpath = tempout + "/model.settings_xhydro.toml"
    write_settings_toml(outpath, params, strict=True)


class TestPreparePygmetInputs:
    # Prepare the data for the subset OI precip grid.
    # Coordinates
    percentile = np.array([50], dtype=np.int64)
    time = pd.date_range("1970-01-01", periods=15, freq="D")
    latitude = np.round(50.0 - 0.1 * np.arange(10), 1).astype(np.float64)
    longitude = np.round(-78.0 + 0.1 * np.arange(10), 1).astype(np.float64)

    # Repeatable
    rng = np.random.default_rng(12345)  # seed = repeatable
    tp = rng.random((len(percentile), len(time), len(latitude), len(longitude))).astype(np.float64)

    # Build dataset
    ds = xr.Dataset(
        data_vars={"tp": (("percentile", "time", "latitude", "longitude"), tp)},
        coords={
            "percentile": percentile,
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    # Write the file to temp location
    tmpfile = tempfile.mkdtemp()
    path_nc_oi_precip = tmpfile + "/subset_oi_tp.nc"
    ds.to_netcdf(path_nc_oi_precip)

    # Now prepare the data for the subset temperature grid.
    # Coordinates
    time = pd.date_range("1970-01-01", periods=15, freq="D")  # datetime64[ns]
    longitude = np.round(-78.0 + 0.1 * np.arange(10), 1).astype(np.float64)  # -78.0 ... -77.1
    latitude = np.round(50.0 - 0.1 * np.arange(10), 1).astype(np.float64)  # 50.0 ... 49.1

    # Repeatable
    rng = np.random.default_rng(12345)

    shape_3d = (len(longitude), len(latitude), len(time))

    # Base mean temperature field
    tmean = rng.normal(loc=12.0, scale=6.0, size=shape_3d).astype(np.float32)

    # Positive daily range
    trange = rng.uniform(3.0, 12.0, size=shape_3d).astype(np.float32)
    tasmin = (tmean - 0.5 * trange).astype(np.float32)
    tasmax = (tmean + 0.5 * trange).astype(np.float32)

    # Altitude (m), repeatable
    altitude = rng.uniform(0, 1200, size=(len(longitude), len(latitude))).astype(np.float64)

    # Build dataset
    ds = xr.Dataset(
        data_vars={
            "tasmax": (("longitude", "latitude", "time"), tasmax),
            "tasmin": (("longitude", "latitude", "time"), tasmin),
            "altitude": (("longitude", "latitude"), altitude),
        },
        coords={
            "longitude": longitude,
            "latitude": latitude,
            "time": time,
        },
    )

    # Write file to temp location as well.
    tmpfile_temperature = tempfile.mkdtemp()
    path_nc_grid_temperature = tmpfile_temperature + "/subset_grid_temperature.nc"
    ds.to_netcdf(path_nc_grid_temperature)

    # Intermediate data for reordered stations
    stn = np.arange(100, dtype=np.int64)
    time = pd.date_range("1970-01-01", periods=15, freq="D")  # datetime64[ns]
    # Build station lat/lon from a 10x10 grid (same values/ranges as previous grid example)
    lon_1d = np.round(-78.0 + 0.1 * np.arange(10), 1).astype(np.float64)
    lat_1d = np.round(50.0 - 0.1 * np.arange(10), 1).astype(np.float64)

    # IMPORTANT: indexing="ij" matches grid dim order (longitude, latitude)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d, indexing="ij")  # both shape (10, 10)

    # Flatten to stations (100 points). Order="C" = latitude varies fastest within each longitude.
    longitude = lon2d.ravel(order="C")
    latitude = lat2d.ravel(order="C")

    # Station IDs
    stnid = (stn).astype(np.int64)

    # 2) Repeatable data
    rng = np.random.default_rng(12345)  # seed -> repeatable
    shape = (len(stn), len(time))

    # Precip
    precip = rng.gamma(shape=2.0, scale=2.0, size=shape).astype(np.float64)

    # Temperatures
    tmean = rng.normal(loc=12.0, scale=6.0, size=shape).astype(np.float32)  # mean °C
    trange = rng.uniform(3.0, 12.0, size=shape).astype(np.float32)  # daily range °C (>0)
    tmin = (tmean - 0.5 * trange).astype(np.float32)
    tmax = (tmean + 0.5 * trange).astype(np.float32)

    # 3) Build Dataset
    ds_stn = xr.Dataset(
        data_vars={
            "precip": (("stn", "time"), precip),
            "tmax": (("stn", "time"), tmax),
            "tmin": (("stn", "time"), tmin),
            "latitude": (("stn",), latitude),
            "longitude": (("stn",), longitude),
            "stnid": (("stn",), stnid),
        },
        coords={
            "stn": stn,
            "time": time,
        },
    )

    # Write file to temp location as well.
    tmpfile_reordered = tempfile.mkdtemp()
    path_nc_grid_reordered = tmpfile_reordered + "/subset_reordered_grids_to_stations.nc"
    ds_stn.to_netcdf(path_nc_grid_reordered)

    # tempout folder
    tempout = tempfile.mkdtemp()

    def test_convert_2d_to_1d(self):
        # Run the code
        convert_2d_nc_to_1d_stations(
            self.path_nc_oi_precip,
            self.path_nc_grid_temperature,
            self.tempout + "/subset_reordered_grids_to_stations.nc/",
        )

        ds = xr.open_dataset(self.tempout + "/subset_reordered_grids_to_stations.nc/")
        assert_almost_equal(ds.precip.isel(stn=5, time=5).values, 0.2762543, 5)
        assert_almost_equal(ds.tmax.isel(stn=6, time=5).values, 13.288241, 5)
        assert ds.stn.shape[0] == 100
        assert ds.stn[-1] == 99

    def test_subsample_stations_for_pygmet(self):
        # Take roughly 1 out of every ~10 stations
        isel_every_about_n(
            path_to_nc=self.path_nc_grid_reordered,
            dim="stn",
            outpath=self.tempout + "/subset_oi_vector_format_subsampled.nc/",
            rng=42,
            threshold=0.1,
            n=10,
            jitter=3,
        )

        ds = xr.open_dataset(self.tempout + "/subset_oi_vector_format_subsampled.nc/")
        assert_almost_equal(ds.precip.isel(stn=5, time=5).values, 3.057068, 5)
        assert_almost_equal(ds.tmax.isel(stn=6, time=5).values, 10.385146, 5)
        assert ds.stn.shape[0] == 11
        assert ds.stn[-1] == 96

    def test_make_target_pygmet_grid(self):
        make_target_pygmet_grid(
            self.path_nc_grid_temperature,
            self.tempout + "/new_grid_output_shape.nc",
        )

        ds = xr.open_dataset(self.tempout + "/new_grid_output_shape.nc")
        assert ds.y[0] == 50
        assert ds.mask[5, 5].values == 1
