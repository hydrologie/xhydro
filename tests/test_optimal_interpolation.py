import datetime as dt
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pooch
import xarray as xr

import xhydro.optimal_interpolation.compare_result as cr
import xhydro.optimal_interpolation.optimal_interpolation_fun as opt
from xhydro.optimal_interpolation.ECF_climate_correction import general_ecf


class TestOptimalInterpolationIntegrationCorrectedFiles:

    # Set Github URL for getting files for tests
    GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
    BRANCH_OR_COMMIT_HASH = "main"

    # Get data with pooch
    test_data_path = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/optimal_interpolation/OI_data_corrected.zip",
        known_hash="md5:acdf90b78b53595eb97ff0e84fc07aa8",
    )

    # Extract to a cache path. Easier this way than with the pooch Unzip method, as that one forces the outputs to be
    # a list of files including full path, which makes it harder to attribute the paths to each variable we need below.
    directory_to_extract_to = Path(
        test_data_path
    ).parent  # Extract to the same directory as the zip file
    with ZipFile(test_data_path, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    # Read-in all the files and set to paths that we can access later.
    corresponding_station_file = directory_to_extract_to / "station_correspondence.nc"
    selected_station_file = (
        directory_to_extract_to / "stations_retenues_validation_croisee.csv"
    )
    flow_obs_info_file = directory_to_extract_to / "A20_HYDOBS_TEST_corrected.nc"
    flow_sim_info_file = directory_to_extract_to / "A20_HYDREP_TEST_corrected.nc"
    flow_l1o_info_file = (
        directory_to_extract_to
        / "A20_ANALYS_FLOWJ_RESULTS_CROSS_VALIDATION_L1O_TEST_corrected.nc"
    )

    qobs = xr.open_dataset(flow_obs_info_file)
    qsim = xr.open_dataset(flow_sim_info_file)
    flow_l1o = xr.open_dataset(flow_l1o_info_file)

    station_correspondence = xr.open_dataset(corresponding_station_file)
    df_validation = pd.read_csv(selected_station_file, sep=None, dtype=str)
    observation_stations = list(df_validation["No_station"])

    # Start and end dates for the simulation. Short period for the test.
    start_date = dt.datetime(2018, 11, 1)
    end_date = dt.datetime(2019, 1, 1)

    # Set some variables to use in the tests
    ratio_var_bg = 0.15
    percentiles = [25.0, 50.0, 75.0]
    variogram_bins = 10

    form = 3
    hmax_divider = 2.0
    p1_bnds = [0.95, 1]
    hmax_mult_range_bnds = [0.05, 3]

    def test_cross_validation_execute(self):
        """Test the cross validation of optimal interpolation."""
        # Get the required times only
        qobs = self.qobs.sel(time=slice(self.start_date, self.end_date))
        qsim = self.qsim.sel(time=slice(self.start_date, self.end_date))

        # Run the code and obtain the resulting flows.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=False,
            max_cores=1,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(
            ds["streamflow"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["streamflow"]
            .isel(station_id=10, time=slice(0, 20), percentile=1)
            .data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 10, 31)
        end_date = dt.datetime(2018, 12, 31)

        ds = opt.execute_interpolation(
            self.qobs.sel(time=slice(start_date, end_date)),
            self.qsim.sel(time=slice(start_date, end_date)),
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=False,
            max_cores=1,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        # Verify results
        np.testing.assert_almost_equal(
            ds["streamflow"].isel(station_id=10, time=10, percentile=1).data,
            21.48871,
            1,
        )
        np.testing.assert_almost_equal(
            ds["streamflow"]
            .isel(station_id=10, time=slice(0, 20), percentile=1)
            .data.mean(),
            14.98491,
            1,
        )
        assert len(ds["time"].data) == (end_date - start_date).days + 1

    def test_cross_validation_execute_parallel(self):
        """Test the parallel version of the optimal interpolation cross validation."""
        # Run the interpolation and obtain the resulting flows.
        ds = opt.execute_interpolation(
            self.qobs.sel(time=slice(self.start_date, self.end_date)),
            self.qsim.sel(time=slice(self.start_date, self.end_date)),
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=True,
            max_cores=3,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(
            ds["streamflow"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["streamflow"]
            .isel(station_id=10, time=slice(0, 20), percentile=1)
            .data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

    def test_operational_optimal_interpolation_run(self):
        """Test the operational version of the optimal interpolation code."""
        # Run the interpolation and get flows
        ds = opt.execute_interpolation(
            self.qobs.sel(time=slice(self.start_date, self.end_date)),
            self.qsim.sel(time=slice(self.start_date, self.end_date)),
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=False,
            max_cores=1,
            leave_one_out_cv=False,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(
            ds["streamflow"].isel(station_id=160, time=10, percentile=1).data,
            32.432376,
            1,
        )
        np.testing.assert_almost_equal(
            ds["streamflow"]
            .isel(station_id=160, time=slice(0, 20), percentile=1)
            .data.mean(),
            26.801498,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

    def test_compare_result_compare(self):
        start_date = dt.datetime(2018, 11, 1)
        end_date = dt.datetime(2018, 12, 30)

        cr.compare(
            qobs=self.qobs.sel(time=slice(start_date, end_date)),
            qsim=self.qsim.sel(time=slice(start_date, end_date)),
            flow_l1o=self.flow_l1o.sel(time=slice(start_date, end_date)),
            station_correspondence=self.station_correspondence,
            observation_stations=self.observation_stations,
            show_comparison=False,
        )

    def test_optimal_interpolation_single_time_dim(self):
        """Test the OI for data with no time dimension such as indicators."""
        # Get the required times only
        qobs = self.qobs.sel(time=dt.datetime(2018, 12, 20))
        qsim = self.qsim.sel(time=dt.datetime(2018, 12, 20))

        # TODO: Generate better data to make sure results compute accurately
        # Run the code and ensure dataset is of correct size and code does not crash.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=False,
            max_cores=1,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        assert "time" not in ds
        assert len(ds.percentile) == 3

    def test_optimal_interpolation_no_time_dim(self):
        """Test the OI for data with no time dimension such as indicators."""
        # Get the required times only
        qobs = self.qobs.isel(time=10).drop("time")
        qsim = self.qsim.isel(time=10).drop("time")

        # TODO: Generate better data to make sure results compute accurately
        # Run the code and ensure dataset is of correct size and code does not crash.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=False,
            max_cores=1,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        assert "time" not in ds
        assert len(ds.percentile) == 3


class TestOptimalInterpolationIntegrationOriginalDEHFiles:

    # Set Github URL for getting files for tests
    GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
    BRANCH_OR_COMMIT_HASH = "main"

    # Get data with pooch
    test_data_path = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/optimal_interpolation/OI_data.zip",
        known_hash="md5:1ab72270023366d0410eb6972d1e2656",
    )

    # Extract to a cache path. Easier this way than with the pooch Unzip method, as that one forces the outputs to be
    # a list of files including full path, which makes it harder to attribute the paths to each variable we need below.
    directory_to_extract_to = Path(
        test_data_path
    ).parent  # Extract to the same directory as the zip file
    with ZipFile(test_data_path, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    # Read-in all the files and set to paths that we can access later.
    station_info_file = directory_to_extract_to / "OI_data/Info_Station.csv"
    corresponding_station_file = (
        directory_to_extract_to / "OI_data/Correspondance_Station.csv"
    )
    selected_station_file = (
        directory_to_extract_to / "OI_data/stations_retenues_validation_croisee.csv"
    )
    flow_obs_info_file = directory_to_extract_to / "OI_data/A20_HYDOBS_TEST.nc"
    flow_sim_info_file = directory_to_extract_to / "OI_data/A20_HYDREP_TEST.nc"
    flow_l1o_info_file = (
        directory_to_extract_to
        / "OI_data/A20_ANALYS_FLOWJ_RESULTS_CROSS_VALIDATION_L1O_TEST.nc"
    )

    # Correct files to get them into the correct shape.
    df = pd.read_csv(station_info_file, sep=None, dtype=str)
    qobs = xr.open_dataset(flow_obs_info_file)
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

    df = pd.read_csv(corresponding_station_file, sep=None, dtype=str)
    station_correspondence = xr.Dataset(
        {
            "reach_id": ("station", df["troncon_id"]),
            "station_id": ("station", df["No.Station"]),
        }
    )

    qsim = xr.open_dataset(flow_sim_info_file)
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

    df_validation = pd.read_csv(selected_station_file, sep=None, dtype=str)
    observation_stations = list(df_validation["No_station"])

    flow_l1o = xr.open_dataset(flow_l1o_info_file)
    flow_l1o = flow_l1o.assign(
        {"station_id": ("station", flow_l1o["station_id"].values.astype(str))}
    )
    flow_l1o = flow_l1o.assign(
        {"streamflow": (("percentile", "station", "time"), flow_l1o["Dis"].values)}
    )
    tt = flow_l1o["time"].dt.round(freq="D")
    flow_l1o = flow_l1o.assign_coords(time=tt.values)

    # Now we are all in dataset format!

    # Start and end dates for the simulation. Short period for the test.
    start_date = dt.datetime(2018, 11, 1)
    end_date = dt.datetime(2019, 1, 1)

    # Set some variables to use in the tests
    ratio_var_bg = 0.15
    percentiles = [25.0, 50.0, 75.0]
    variogram_bins = 10

    form = 3
    hmax_divider = 2.0
    p1_bnds = [0.95, 1]
    hmax_mult_range_bnds = [0.05, 3]

    def test_cross_validation_execute(self):
        """Test the cross validation of optimal interpolation."""
        # Get the required times only
        qobs = self.qobs.sel(time=slice(self.start_date, self.end_date))
        qsim = self.qsim.sel(time=slice(self.start_date, self.end_date))

        # Run the code and obtain the resulting flows.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=False,
            max_cores=1,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(
            ds["streamflow"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["streamflow"]
            .isel(station_id=10, time=slice(0, 20), percentile=1)
            .data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 10, 31)
        end_date = dt.datetime(2018, 12, 31)

        ds = opt.execute_interpolation(
            self.qobs.sel(time=slice(start_date, end_date)),
            self.qsim.sel(time=slice(start_date, end_date)),
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=False,
            max_cores=1,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        # Verify results
        np.testing.assert_almost_equal(
            ds["streamflow"].isel(station_id=10, time=10, percentile=1).data,
            21.48871,
            1,
        )
        np.testing.assert_almost_equal(
            ds["streamflow"]
            .isel(station_id=10, time=slice(0, 20), percentile=1)
            .data.mean(),
            14.98491,
            1,
        )
        assert len(ds["time"].data) == (end_date - start_date).days + 1

    def test_cross_validation_execute_parallel(self):
        """Test the parallel version of the optimal interpolation cross validation."""
        # Run the interpolation and get flows
        ds = opt.execute_interpolation(
            self.qobs.sel(time=slice(self.start_date, self.end_date)),
            self.qsim.sel(time=slice(self.start_date, self.end_date)),
            self.station_correspondence,
            self.observation_stations,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            variogram_bins=self.variogram_bins,
            parallelize=True,
            max_cores=3,
            leave_one_out_cv=True,
            form=self.form,
            hmax_divider=self.hmax_divider,
            p1_bnds=self.p1_bnds,
            hmax_mult_range_bnds=self.hmax_mult_range_bnds,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(
            ds["streamflow"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["streamflow"]
            .isel(station_id=10, time=slice(0, 20), percentile=1)
            .data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

    def test_compare_result_compare(self):
        start_date = dt.datetime(2018, 11, 1)
        end_date = dt.datetime(2018, 12, 30)

        cr.compare(
            qobs=self.qobs.sel(time=slice(start_date, end_date)),
            qsim=self.qsim.sel(time=slice(start_date, end_date)),
            flow_l1o=self.flow_l1o.sel(time=slice(start_date, end_date)),
            station_correspondence=self.station_correspondence,
            observation_stations=self.observation_stations,
            show_comparison=False,
        )


class TestOptimalInterpolationFunction:
    def test_optimal_interpolation_test_data(self):
        """Test the optimal interpolation code for a single day."""
        # Run the code and obtain the resulting flows.
        ecf_fun = partial(general_ecf, form=4)
        v_est, var_est, _ = opt.optimal_interpolation(
            lat_est=np.array([-0.5, -0.25, 0.25, 0.5]),
            lon_est=np.array([-0.5, -0.25, 0.25, 0.5]),
            lat_obs=np.array([-1.0, 0.0, 1.0]),
            lon_obs=np.array([-1.0, 0.0, 1.0]),
            ecf=partial(ecf_fun, par=[1.0, 0.5]),
            bg_var_obs=np.array([1.0, 1.0, 1.0]),
            bg_var_est=np.array([1.0, 1.0, 1.0, 1.0]),
            var_obs=np.array([0.25, 0.25, 0.25]),
            bg_departures=np.array([0.2, 0.3, 0.1]),
            bg_est=np.array([1.0, 4.0, 3.0, 5.0]),
            precalcs={},
        )

        np.testing.assert_almost_equal(v_est[2], 3.0004557853394282, 2)
        np.testing.assert_almost_equal(var_est[0], 0.9999999682102936, 2)


class TestEcfFunctions:
    def test_general_ecf(self):
        h = np.array([0, 1, 2])
        param = np.array([0.5, 50])

        # Test the three forms for the general_ecf function
        assert np.allclose(
            general_ecf(h, param, form=1), np.array([0.5, 0.49990132, 0.49961051])
        )
        assert np.allclose(
            general_ecf(h, param, form=2), np.array([0.5, 0.49990001, 0.49960016])
        )
        assert np.allclose(
            general_ecf(h, param, form=3), np.array([0.5, 0.49009934, 0.48039472])
        )
