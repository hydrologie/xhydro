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

    write_config_toml("model.config_xhydro.toml", params, strict=True)


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

    write_settings_toml("model.settings_xhydro.toml", params, strict=True)


def test_convert_2d_to_1d():
    path_nc_oi_precip = "./subset_oi_tp.nc"
    path_nc_grid_temperature = "./subset_grid_temperature.nc"
    outpath = "./subset_reordered_grids_to_stations.nc/"

    convert_2d_nc_to_1d_stations(
        path_nc_oi_precip,
        path_nc_grid_temperature,
        outpath,
    )


def test_subsample_stations_for_pygmet():
    path_to_nc = "./subset_reordered_grids_to_stations.nc"
    outpath = "./subset_oi_vector_format_subsampled.nc"

    # Take roughly 1 out of every ~10 stations
    isel_every_about_n(
        path_to_nc=path_to_nc,
        dim="stn",
        outpath=outpath,
        rng=42,
        threshold=0.1,
        n=10,
        jitter=3,
    )


def test_make_target_pygmet_grid():
    ds_in_path = "./subset_grid_temperature.nc"
    outpath = "./new_grid_output_shape.nc"
    make_target_pygmet_grid(ds_in_path, outpath)
