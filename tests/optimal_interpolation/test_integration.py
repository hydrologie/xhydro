import datetime as dt
import tempfile

import numpy as np

import xhydro.optimal_interpolation.compare_result as cr
import xhydro.optimal_interpolation.cross_validation as cv
from xhydro.optimal_interpolation.functions.testdata import get_file


class Test_optimal_interpolation_integration:
    # Set Github URL for getting files for tests
    git_url = "https://github.com/Mayetea/xhydro-testdata"
    branch = "optimal_interpolation"
    dataf = "data/optimal_interpolation/"

    # Prepare files. Get them on the public data repo.
    station_info_file = get_file(
        name=dataf + "Info_Station.csv", github_url=git_url, branch=branch
    )
    corresponding_station_file = get_file(
        name=dataf + "Correspondance_Station.csv", github_url=git_url, branch=branch
    )
    selected_station_file = get_file(
        name=dataf + "stations_retenues_validation_croisee.csv",
        github_url=git_url,
        branch=branch,
    )
    flow_obs_info_file = get_file(
        name=dataf + "A20_HYDOBS_TEST.nc", github_url=git_url, branch=branch
    )
    flow_sim_info_file = get_file(
        name=dataf + "A20_HYDREP_TEST.nc", github_url=git_url, branch=branch
    )
    flow_l1o_info_file = get_file(
        name=dataf + "A20_ANALYS_FLOWJ_RESULTS_CROSS_VALIDATION_L1O_TEST.nc",
        github_url=git_url,
        branch=branch,
    )

    # Make a list with these files paths, required for the code.
    files = [
        station_info_file,
        corresponding_station_file,
        selected_station_file,
        flow_obs_info_file,
        flow_sim_info_file,
    ]

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

        # Run the code and obtain the resulting flows.
        result_flows = cv.execute(
            self.start_date,
            self.end_date,
            self.files,
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
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.35, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 51.26, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[1]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[2]) == (self.end_date - self.start_date).days + 1

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 10, 31)
        end_date = dt.datetime(2018, 12, 31)

        result_flows = cv.execute(
            start_date,
            end_date,
            self.files,
            write_file=self.write_file,
            ratio_var_bg=self.ratio_var_bg,
            percentiles=self.percentiles,
            iterations=self.iterations,
            parallelize=False,
        )

        # TODO: CHECK WHY SOME DAYS HAVE ALL NANS LIKE: result_flows[0][27]
        # Test some output flow values
        np.testing.assert_almost_equal(result_flows[1][-1, 0], 8.05, 2)
        np.testing.assert_almost_equal(result_flows[1][-2, 0], 8.4, 2)

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
            self.start_date,
            self.end_date,
            self.files,
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
        np.testing.assert_almost_equal(np.nanmean(result_flows[0][:, :]), 29.35, 2)
        np.testing.assert_almost_equal(np.nanmean(result_flows[2][:, :]), 51.26, 2)

        # Test the time range duration
        assert len(result_flows[0]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[1]) == (self.end_date - self.start_date).days + 1
        assert len(result_flows[2]) == (self.end_date - self.start_date).days + 1

    def test_compare_result_compare(self):
        files = [
            self.selected_station_file,
            self.corresponding_station_file,
            self.flow_obs_info_file,
            self.flow_sim_info_file,
            self.flow_l1o_info_file,
        ]

        cr.compare(self.start_date, self.end_date, files, show_comparaison=False)
