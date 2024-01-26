import datetime as dt

import numpy as np
from pytest import approx

import xhydro.optimal_interpolation.cross_validation as cv
from xhydro.optimal_interpolation.functions.testdata import get_file


class test_optimal_interpolation_integration:
    # Set Github URL for getting files for tests
    git_url = "https://github.com/Mayetea/xhydro-testdata"
    branch = "optimal_interpolation"
    dataf = "data/optimal_interpolation/"

    # Prepare files. Get them on the public data repo.
    station_info_file = get_file(
        name="Info_Station.csv", github_url=git_url, branch=branch
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

    # Make a list with these files paths, required for the code.
    files = [
        station_info_file,
        corresponding_station_file,
        selected_station_file,
        flow_obs_info_file,
        flow_sim_info_file,
    ]

    # Start and end dates for the simulation. Short period for the test.
    start_date = dt.datetime(2018, 11, 1)
    end_date = dt.datetime(2019, 1, 1)

    def test_cross_validation_execute(self, files, start_date, end_date):
        """test the cross validation of optimal interpolation."""

        # Run the code and obtain the resulting flows.
        result_flows = cv.execute(start_date, end_date, files, parallelize=False)

        # Test some output flow values
        assert result_flows[0][-1, 0] == 8.042503657491906
        assert result_flows[0][-2, 0] == 8.377341430781929

        # To randomize to test direct values
        assert np.nanmean(result_flows[1][:, :]) == approx(33, 0.5)
        assert np.nanmean(result_flows[2][:, :]) == approx(59, 0.5)

        # Test the time range duration
        assert len(result_flows[0]) == 61
        assert len(result_flows[1]) == 61
        assert len(result_flows[2]) == 61

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 1, 1)
        end_date = dt.datetime(2018, 1, 31)

        result_flows = cv.execute(start_date, end_date, files, parallelize=False)

        # Test some output flow values
        assert result_flows[0][-1, 0] == 7.884402090442147
        assert result_flows[0][-2, 0] == 8.170066622988958

        # To randomize to test direct values
        assert np.nanmean(result_flows[1][:, :]) == approx(33, 0.5)
        assert np.nanmean(result_flows[2][:, :]) == approx(59, 0.5)

        # Test the time range duration
        assert len(result_flows[0]) == 31
        assert len(result_flows[1]) == 31
        assert len(result_flows[2]) == 31

    def test_cross_validation_execute_parralelize(self, files, start_date, end_date):
        """Test the parallel version of the optimal interpolation cross validation."""

        # Run the interpolation and get flows
        result_flows = cv.execute(start_date, end_date, files)

        # Test some output flow values
        assert result_flows[0][-1, 0] == 7.884402090442147
        assert result_flows[0][-2, 0] == 8.170066622988958

        # To randomize to test direct values
        assert np.nanmean(result_flows[1][:, :]) == approx(33, 0.5)
        assert np.nanmean(result_flows[2][:, :]) == approx(59, 0.5)

        # Test the time range duration
        assert len(result_flows[0]) == 61
        assert len(result_flows[1]) == 61
        assert len(result_flows[2]) == 61
