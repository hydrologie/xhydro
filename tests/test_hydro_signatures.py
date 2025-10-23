import numpy as np
import pytest
from xhydro import modelling as xh
from xclim import land
from xclim.core.units import convert_units_to
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

class TestAnnualMaxima:
    def test_simple(self, q_series):
        # 2 years of daily data
        a = np.zeros(365 * 2)

        # Year 1: 1 day peak
        a[50:51] = 20

        # Year 2: 2 days with peaks
        a[400:401] = 5
        a[401:402] = 6

        # Create a daily time index
        q = q_series(a)

        out = xh.annual_maxima(q)

        out
        # Year 1: expect maxima 20, DOY = 51
        # Year 2: expect maxima 6, DOY = 36
        # Year 3 (due to water year resampling) : expect maxima 0, DOY = c aka october 1st the start of water year
        np.testing.assert_array_equal(out["peak_flow"].values, [20.0, 6.0, 0.0])
        np.testing.assert_array_equal(out["peak_doy"].values, [51, 36, 274])


class TestFDCSlope:
    def test_simple(self, q_series):
        # 5 years of increasing data with slope of 1
        q = np.arange(1, 1826)

        # Create a daily time index
        q = q_series(q)

        out = xh.fdc_slope(q)
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
        pr = pr_series(pr, units = "mm/hr")

        out = xh.total_runoff_ratio(q, a, pr)
        np.testing.assert_allclose(out, 0.0018, atol=1e-15)

class TestElastIndex:
    def test_simple(self, q_series, pr_series):
        #5 years of increasing data with slope of 1
        q = np.arange(1, 1826)
        pr = np.arange(1, 1826)

         # Create a daily time index
        q = q_series(q)
        pr = pr_series(pr, units="mm/hr")

        out = xh.elasticity_index(q, pr)
        np.testing.assert_allclose(out, 1.000672, rtol=1e-6, atol=0) #not exactly 1 due to epsilon

