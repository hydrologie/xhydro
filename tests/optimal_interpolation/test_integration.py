import datetime as dt
import tempfile
from pathlib import Path
from zipfile import ZipFile
import xarray as xr
import pandas as pd

import numpy as np
import pooch

import xhydro.optimal_interpolation.compare_result as cr
import xhydro.optimal_interpolation.cross_validation as cv


class TestOptimalInterpolationIntegrationCorrectedFiles:

    # Set Github URL for getting files for tests
    GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
    BRANCH_OR_COMMIT_HASH = "optimal-interpolation"

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
    selected_station_file = directory_to_extract_to / "stations_retenues_validation_croisee.csv"
    flow_obs_info_file = directory_to_extract_to / "A20_HYDOBS_TEST_corrected.nc"
    flow_sim_info_file = directory_to_extract_to / "A20_HYDREP_TEST_corrected.nc"
    flow_l1o_info_file = directory_to_extract_to / "A20_ANALYS_FLOWJ_RESULTS_CROSS_VALIDATION_L1O_TEST_corrected.nc"

    flow_obs = xr.open_dataset(flow_obs_info_file)
    flow_sim = xr.open_dataset(flow_sim_info_file)
    flow_l1o = xr.open_dataset(flow_l1o_info_file)

    station_correspondence = xr.open_dataset(corresponding_station_file)
    df_validation = pd.read_csv(selected_station_file, sep=None, dtype=str)
    crossvalidation_stations = list(df_validation["No_station"])

    # Path to file to be written to
    tmpdir = tempfile.mkdtemp()
    write_file = tmpdir + "/" + "Test_OI_results.nc"

    # Start and end dates for the simulation. Short period for the test.
    start_date = dt.datetime(2018, 11, 1)
    end_date = dt.datetime(2019, 1, 1)

    # Set some variables to use in the tests
    ratio_var_bg = 0.15
    percentiles = [0.25, 0.50, 0.75]
    iterations = 10

    def test_cross_validation_execute(self):
        """Test the cross validation of optimal interpolation."""

        # Get the required times only
        flow_obs = self.flow_obs.sel(time=slice(self.start_date, self.end_date))
        flow_sim = self.flow_sim.sel(time=slice(self.start_date, self.end_date))

        # Run the code and obtain the resulting flows.
        result_flows = cv.execute(
            flow_obs,
            flow_sim,
            self.station_correspondence,
            self.crossvalidation_stations,
            write_file=self.write_file,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            iterations=self.iterations,
            parallelize=False,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(result_flows[1][-1, 0], 7.9, 2)
        np.testing.assert_almost_equal(result_flows[1][-2, 0], 8.04, 2)

        # To randomize to test direct values
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.3669, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 51.26, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[1]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[2]) == (self.end_date - self.start_date).days + 1

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 10, 31)
        end_date = dt.datetime(2018, 12, 31)

        result_flows = cv.execute(
            self.flow_obs.sel(time=slice(start_date, end_date)),
            self.flow_sim.sel(time=slice(start_date, end_date)),
            self.station_correspondence,
            self.crossvalidation_stations,
            write_file=self.write_file,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            iterations=self.iterations,
            parallelize=False,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(result_flows[1][-1, 0], 8.0505, 2)
        np.testing.assert_almost_equal(result_flows[1][-2, 0], 8.3878, 2)

        # To randomize to test direct values
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.82, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 52.28, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (end_date - start_date).days + 1
        assert len(result_flows[1]) == (end_date - start_date).days + 1
        assert len(result_flows[2]) == (end_date - start_date).days + 1

    def test_cross_validation_execute_parallel(self):
        """Test the parallel version of the optimal interpolation cross validation."""

        # Run the interpolation and get flows
        result_flows = cv.execute(
            self.flow_obs.sel(time=slice(self.start_date, self.end_date)),
            self.flow_sim.sel(time=slice(self.start_date, self.end_date)),
            self.station_correspondence,
            self.crossvalidation_stations,
            write_file=self.write_file,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            iterations=self.iterations,
            parallelize=True,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(result_flows[1][-1, 0], 7.9, 2)
        np.testing.assert_almost_equal(result_flows[1][-2, 0], 8.04, 2)

        # To randomize to test direct values
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.3669, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 51.26, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[1]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[2]) == (self.end_date - self.start_date).days + 1

    def test_compare_result_compare(self):
        start_date = dt.datetime(2018, 11, 1)
        end_date = dt.datetime(2018, 12, 30)

        cr.compare(
            flow_obs=self.flow_obs.sel(time=slice(start_date, end_date)),
            flow_sim=self.flow_sim.sel(time=slice(start_date, end_date)),
            flow_l1o=self.flow_l1o.sel(time=slice(start_date, end_date)),
            station_correspondence=self.station_correspondence,
            crossvalidation_stations=self.crossvalidation_stations,
            show_comparison=False)


class TestOptimalInterpolationIntegrationOriginalDEHFiles:

    # Set Github URL for getting files for tests
    GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
    BRANCH_OR_COMMIT_HASH = "optimal-interpolation"

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
    flow_obs = xr.open_dataset(flow_obs_info_file)
    flow_obs = flow_obs.assign({'centroid_lat': ('station', df["Latitude Centroide BV"].astype(np.float32))})
    flow_obs = flow_obs.assign({'centroid_lon': ('station', df["Longitude Centroide BV"].astype(np.float32))})
    flow_obs = flow_obs.assign({'classement': ('station', df["Classement"].astype(np.float32))})
    flow_obs = flow_obs.assign({'station_id': ('station', flow_obs["station_id"].values.astype(str))})
    flow_obs = flow_obs.assign({'streamflow': (('station', 'time'), flow_obs["Dis"].values)})

    df = pd.read_csv(corresponding_station_file, sep=None, dtype=str)
    station_correspondence = xr.Dataset(
        {'reach_id': ('station', df["troncon_id"]), "station_id": ("station", df["No.Station"])}
    )

    flow_sim = xr.open_dataset(flow_sim_info_file)
    flow_sim = flow_sim.assign({'station_id': ('station', flow_sim["station_id"].values.astype(str))})
    flow_sim = flow_sim.assign({'streamflow': (('station', 'time'), flow_sim["Dis"].values)})
    flow_sim["station_id"].values[143] = 'SAGU99999'  # Forcing to change due to double value wtf.
    flow_sim["station_id"].values[7] = 'BRKN99999'  # Forcing to change due to double value wtf.

    df_validation = pd.read_csv(selected_station_file, sep=None, dtype=str)
    crossvalidation_stations = list(df_validation["No_station"])

    flow_l1o = xr.open_dataset(flow_l1o_info_file)
    flow_l1o = flow_l1o.assign({'station_id': ('station', flow_l1o["station_id"].values.astype(str))})
    flow_l1o = flow_l1o.assign({'streamflow': (('percentile', 'station', 'time'), flow_l1o["Dis"].values)})
    tt = flow_l1o["time"].dt.round(freq="D")
    flow_l1o = flow_l1o.assign_coords(time=tt.values)

    # Now we are all in dataset format!

    # Path to file to be written to
    tmpdir = tempfile.mkdtemp()
    write_file = tmpdir + "/" + "Test_OI_results.nc"

    # Start and end dates for the simulation. Short period for the test.
    start_date = dt.datetime(2018, 11, 1)
    end_date = dt.datetime(2019, 1, 1)

    # Set some variables to use in the tests
    ratio_var_bg = 0.15
    percentiles = [0.25, 0.50, 0.75]
    iterations = 10

    def test_cross_validation_execute(self):
        """Test the cross validation of optimal interpolation."""

        # Get the required times only
        flow_obs = self.flow_obs.sel(time=slice(self.start_date, self.end_date))
        flow_sim = self.flow_sim.sel(time=slice(self.start_date, self.end_date))

        # Run the code and obtain the resulting flows.
        result_flows = cv.execute(
            flow_obs,
            flow_sim,
            self.station_correspondence,
            self.crossvalidation_stations,
            write_file=self.write_file,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            iterations=self.iterations,
            parallelize=False,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(result_flows[1][-1, 0], 7.9, 2)
        np.testing.assert_almost_equal(result_flows[1][-2, 0], 8.04, 2)

        # To randomize to test direct values
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.3669, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 51.26, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[1]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[2]) == (self.end_date - self.start_date).days + 1

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 10, 31)
        end_date = dt.datetime(2018, 12, 31)

        result_flows = cv.execute(
            self.flow_obs.sel(time=slice(start_date, end_date)),
            self.flow_sim.sel(time=slice(start_date, end_date)),
            self.station_correspondence,
            self.crossvalidation_stations,
            write_file=self.write_file,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            iterations=self.iterations,
            parallelize=False,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(result_flows[1][-1, 0], 8.0505, 2)
        np.testing.assert_almost_equal(result_flows[1][-2, 0], 8.3878, 2)

        # To randomize to test direct values
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.82, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 52.28, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (end_date - start_date).days + 1
        assert len(result_flows[1]) == (end_date - start_date).days + 1
        assert len(result_flows[2]) == (end_date - start_date).days + 1

    def test_cross_validation_execute_parallel(self):
        """Test the parallel version of the optimal interpolation cross validation."""

        # Run the interpolation and get flows
        result_flows = cv.execute(
            self.flow_obs.sel(time=slice(self.start_date, self.end_date)),
            self.flow_sim.sel(time=slice(self.start_date, self.end_date)),
            self.station_correspondence,
            self.crossvalidation_stations,
            write_file=self.write_file,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            iterations=self.iterations,
            parallelize=True,
        )

        # Test some output flow values
        np.testing.assert_almost_equal(result_flows[1][-1, 0], 7.9, 2)
        np.testing.assert_almost_equal(result_flows[1][-2, 0], 8.04, 2)

        # To randomize to test direct values
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.3669, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 51.26, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[1]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[2]) == (self.end_date - self.start_date).days + 1

    def test_compare_result_compare(self):
        start_date = dt.datetime(2018, 11, 1)
        end_date = dt.datetime(2018, 12, 30)

        cr.compare(
            flow_obs=self.flow_obs.sel(time=slice(start_date, end_date)),
            flow_sim=self.flow_sim.sel(time=slice(start_date, end_date)),
            flow_l1o=self.flow_l1o.sel(time=slice(start_date, end_date)),
            station_correspondence=self.station_correspondence,
            crossvalidation_stations=self.crossvalidation_stations,
            show_comparison=False)
