import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xhydro.indicators import hydro_signatures as xh


@pytest.fixture
def q_series():
    def _q_series(values, start="1/1/2000", units="m3 s-1"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="q",
            attrs={
                "standard_name": "water_volume_transport_in_river_channel",
                "units": units,
            },
        )

    return _q_series


@pytest.fixture
def pr_series():
    """Return precipitation time series."""

    def _pr_series(values, start="1/1/2000", units="mm/hr"):
        coords = pd.date_range(start, periods=len(values), freq="D")
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "precipitation",
                "units": units,
            },
        )

    return _pr_series


@pytest.fixture
def area_series():
    def _area_series(values, units="km2"):
        return xr.DataArray(
            values,
            name="area",
            attrs={
                "standard_name": "cell_area",
                "units": units,
            },
        )

    return _area_series


class TestFDCSlope:
    def test_simple(self, q_series):
        # 5 years of increasing data with slope of 1
        q = np.arange(1, 1826)

        # Create a daily time index
        q = q_series(q)
        print(q)
        out = xh.flow_duration_curve_slope(q)
        np.testing.assert_allclose(out, 2.097932, atol=1e-15)


# Expected: ( np.log(1825 / 3) - np.log(1825 * 2 / 3) ) / .33


class TestTotRR:
    def test_simple(self, q_series, area_series, pr_series):
        # 1 years of daily data
        q = np.ones(365) * 10
        pr = np.ones(365) * 20

        # 30 days with low flows, ratio should stay the same
        q[300:330] = 5
        pr[270:300] = 10
        a = 1000
        a = area_series(a)

        # Create a daily time index
        q = q_series(q)
        pr = pr_series(pr, units="mm/hr")

        out = xh.total_runoff_ratio(q, a, pr)
        np.testing.assert_allclose(out, 0.0018, atol=1e-15)


class TestElastIndex:
    def test_simple(self, q_series, pr_series):
        # 5 years of increasing data with slope of 1
        q = np.arange(1, 1826)
        pr = np.arange(1, 1826)

        # Create a daily time index
        q = q_series(q)
        pr = pr_series(pr, units="mm/hr")

        out = xh.elasticity_index(q, pr)
        np.testing.assert_allclose(out, 0.999997, rtol=1e-6, atol=0)  # not exactly 1 due to epsilon
        # print(type(out))


class TestHurstExpNoise:
    def test_simple(self, q_series, pr_series):
        # daily time index
        np.random.seed(0)
        q = np.random.randn(365 * 10)  # 10 years of random daily flows
        q = q_series(q)

        out = xh.hurst_exp(q)  # returns a value close to 0.5 representing noise.
        print(out)

        assert 0.3 <= out <= 0.6, f"H={out:.3f} out of expected range"


class TestHurstExp:
    def test_simple(self, q_series, pr_series):
        # daily time index
        q = np.arange(1, 1826)
        q = q_series(q)

        out = xh.hurst_exp(q)  # returns very high value due to artificial input
        print(out)
        np.testing.assert_allclose(out, 1.499721, rtol=1e-6, atol=0)
