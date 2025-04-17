import datetime as dt
from functools import partial

import numpy as np
import pytest

import xhydro.optimal_interpolation.compare_result as cr
import xhydro.optimal_interpolation.optimal_interpolation_fun as opt
from xhydro.optimal_interpolation.ECF_climate_correction import general_ecf


class TestOptimalInterpolationIntegrationCorrectedFiles:

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

    def test_cross_validation_execute(self, corrected_oi_data):
        """Test the cross validation of optimal interpolation."""
        # Get the required times only
        qobs = (
            corrected_oi_data["qobs"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"})
        )
        qsim = (
            corrected_oi_data["qsim"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"})
        )

        # Run the code and obtain the resulting flows.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            corrected_oi_data["station_correspondence"],
            corrected_oi_data["observation_stations"],
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
            ds["q"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["q"].isel(station_id=10, time=slice(0, 20), percentile=1).data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 10, 31)
        end_date = dt.datetime(2018, 12, 31)

        ds = opt.execute_interpolation(
            corrected_oi_data["qobs"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            corrected_oi_data["qsim"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            corrected_oi_data["station_correspondence"],
            corrected_oi_data["observation_stations"],
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
            ds["q"].isel(station_id=10, time=10, percentile=1).data,
            21.48871,
            1,
        )
        np.testing.assert_almost_equal(
            ds["q"].isel(station_id=10, time=slice(0, 20), percentile=1).data.mean(),
            14.98491,
            1,
        )
        assert len(ds["time"].data) == (end_date - start_date).days + 1

    # FIXME: Not sure what's going on here. This test is failing on conda-forge.
    @pytest.mark.xfail(
        reason="test reports that num processes is not more than one on conda-forge."
    )
    def test_cross_validation_execute_parallel(self, corrected_oi_data):
        """Test the parallel version of the optimal interpolation cross validation."""
        # Run the interpolation and obtain the resulting flows.
        ds = opt.execute_interpolation(
            corrected_oi_data["qobs"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"}),
            corrected_oi_data["qsim"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"}),
            corrected_oi_data["station_correspondence"],
            corrected_oi_data["observation_stations"],
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
            ds["q"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["q"].isel(station_id=10, time=slice(0, 20), percentile=1).data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

    def test_operational_optimal_interpolation_run(self, corrected_oi_data):
        """Test the operational version of the optimal interpolation code."""
        # Run the interpolation and get flows
        ds = opt.execute_interpolation(
            corrected_oi_data["qobs"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"}),
            corrected_oi_data["qsim"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"}),
            corrected_oi_data["station_correspondence"],
            corrected_oi_data["observation_stations"],
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
            ds["q"].isel(station_id=160, time=10, percentile=1).data,
            32.432376,
            1,
        )
        np.testing.assert_almost_equal(
            ds["q"].isel(station_id=160, time=slice(0, 20), percentile=1).data.mean(),
            26.801498,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

    def test_compare_result_compare(self, corrected_oi_data):
        start_date = dt.datetime(2018, 11, 1)
        end_date = dt.datetime(2018, 12, 30)

        cr.compare(
            qobs=corrected_oi_data["qobs"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            qsim=corrected_oi_data["qsim"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            flow_l1o=corrected_oi_data["flow_l1o"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            station_correspondence=corrected_oi_data["station_correspondence"],
            observation_stations=corrected_oi_data["observation_stations"],
            show_comparison=False,
        )

    def test_optimal_interpolation_single_time_dim(self, corrected_oi_data):
        """Test the OI for data with no time dimension such as indicators."""
        # Get the required times only
        qobs = (
            corrected_oi_data["qobs"]
            .sel(time=dt.datetime(2018, 12, 20))
            .rename({"streamflow": "q"})
        )
        qsim = (
            corrected_oi_data["qsim"]
            .sel(time=dt.datetime(2018, 12, 20))
            .rename({"streamflow": "q"})
        )

        # TODO: Generate better data to make sure results compute accurately
        # Run the code and ensure dataset is of correct size and code does not crash.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            corrected_oi_data["station_correspondence"],
            corrected_oi_data["observation_stations"],
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

    def test_optimal_interpolation_no_time_dim(self, corrected_oi_data):
        """Test the OI for data with no time dimension such as indicators."""
        # Get the required times only
        qobs = (
            corrected_oi_data["qobs"]
            .isel(time=10)
            .drop_vars("time")
            .rename({"streamflow": "q"})
        )
        qsim = (
            corrected_oi_data["qsim"]
            .isel(time=10)
            .drop_vars("time")
            .rename({"streamflow": "q"})
        )

        # TODO: Generate better data to make sure results compute accurately
        # Run the code and ensure dataset is of correct size and code does not crash.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            corrected_oi_data["station_correspondence"],
            corrected_oi_data["observation_stations"],
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

    def test_cross_validation_execute(self, oi_data):
        """Test the cross validation of optimal interpolation."""
        # Get the required times only
        qobs = (
            oi_data["qobs"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"})
        )
        qsim = (
            oi_data["qsim"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"})
        )

        # Run the code and obtain the resulting flows.
        ds = opt.execute_interpolation(
            qobs,
            qsim,
            oi_data["station_correspondence"],
            oi_data["observation_stations"],
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
            ds["q"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["q"].isel(station_id=10, time=slice(0, 20), percentile=1).data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

        # Test a different data range to verify that the last entry is different
        start_date = dt.datetime(2018, 10, 31)
        end_date = dt.datetime(2018, 12, 31)

        ds = opt.execute_interpolation(
            oi_data["qobs"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            oi_data["qsim"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            oi_data["station_correspondence"],
            oi_data["observation_stations"],
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
            ds["q"].isel(station_id=10, time=10, percentile=1).data,
            21.48871,
            1,
        )
        np.testing.assert_almost_equal(
            ds["q"].isel(station_id=10, time=slice(0, 20), percentile=1).data.mean(),
            14.98491,
            1,
        )
        assert len(ds["time"].data) == (end_date - start_date).days + 1

    # FIXME: Not sure what's going on here. This test is failing on conda-forge.
    @pytest.mark.xfail(
        reason="test reports that num processes is not more than one on conda-forge."
    )
    def test_cross_validation_execute_parallel(self, oi_data):
        """Test the parallel version of the optimal interpolation cross validation."""
        # Run the interpolation and get flows
        ds = opt.execute_interpolation(
            oi_data["qobs"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"}),
            oi_data["qsim"]
            .sel(time=slice(self.start_date, self.end_date))
            .rename({"streamflow": "q"}),
            oi_data["station_correspondence"],
            oi_data["observation_stations"],
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
            ds["q"].isel(station_id=10, time=10, percentile=1).data,
            21.21767,
            1,
        )
        np.testing.assert_almost_equal(
            ds["q"].isel(station_id=10, time=slice(0, 20), percentile=1).data.mean(),
            15.06389,
            1,
        )
        assert len(ds["time"].data) == (self.end_date - self.start_date).days + 1

    def test_compare_result_compare(self, oi_data):
        start_date = dt.datetime(2018, 11, 1)
        end_date = dt.datetime(2018, 12, 30)

        cr.compare(
            qobs=oi_data["qobs"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            qsim=oi_data["qsim"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            flow_l1o=oi_data["flow_l1o"]
            .sel(time=slice(start_date, end_date))
            .rename({"streamflow": "q"}),
            station_correspondence=oi_data["station_correspondence"],
            observation_stations=oi_data["observation_stations"],
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
