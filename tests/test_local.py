import numpy as np
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro.frequency_analysis as xhfa
from xhydro.frequency_analysis.local import get_plotting_positions


class TestFit:
    def test_fit(self):
        ds = timeseries(
            np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
            variable="streamflow",
            start="2001-01-01",
            freq="YS",
            as_dataset=True,
        )
        params = xhfa.local.fit(ds, distributions=["gamma", "pearson3"])

        np.testing.assert_array_equal(params.dparams, ["a", "skew", "loc", "scale"])
        np.testing.assert_array_equal(params.scipy_dist, ["gamma", "pearson3"])
        assert params.streamflow.attrs["scipy_dist"] == ["gamma", "pearson3"]
        assert params.streamflow.attrs["estimator"] == "Maximum likelihood"
        assert params.streamflow.attrs["long_name"] == "Distribution parameters"
        assert (
            params.streamflow.attrs["description"] == "Parameters of the distributions"
        )

        np.testing.assert_array_almost_equal(
            params.streamflow,
            [
                [9.95357815e00, np.nan, -3.07846650e01, 1.56498193e01],
                [np.nan, -2.25674044e-05, 1.25012261e02, 4.74238877e01],
            ],
        )

    def test_default(self):
        ds = timeseries(
            np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
            variable="streamflow",
            start="2001-01-01",
            freq="YS",
            as_dataset=True,
        )
        params = xhfa.local.fit(ds)
        np.testing.assert_array_equal(
            params.dparams, ["a", "c", "skew", "loc", "scale"]
        )
        np.testing.assert_array_equal(
            params.scipy_dist,
            [
                "expon",
                "gamma",
                "genextreme",
                "genpareto",
                "gumbel_r",
                "pearson3",
                "weibull_min",
            ],
        )

    @pytest.mark.parametrize("miny", [10, 15])
    def test_min_years(self, miny):
        ds = timeseries(
            np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
            variable="streamflow",
            start="2001-01-01",
            freq="YS",
            as_dataset=True,
        )
        params = xhfa.local.fit(ds, distributions=["gamma"], min_years=miny)
        np.testing.assert_array_almost_equal(
            params.streamflow,
            (
                [[9.95357815e00, -3.07846650e01, 1.56498193e01]]
                if miny == 10
                else [[np.nan, np.nan, np.nan]]
            ),
        )


@pytest.mark.parametrize("mode", ["max", "min", "foo"])
def test_quantiles(mode):
    ds = timeseries(
        np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
        variable="streamflow",
        start="2001-01-01",
        freq="YS",
        as_dataset=True,
    )
    params = xhfa.local.fit(ds, distributions=["gamma", "pearson3"])

    if mode == "foo":
        with pytest.raises(ValueError):
            xhfa.local.parametric_quantiles(params, [10, 20], mode=mode)
    else:
        rp = xhfa.local.parametric_quantiles(params, [10, 20], mode=mode)

        np.testing.assert_array_equal(rp.return_period, [10, 20])
        np.testing.assert_array_equal(
            rp.p_quantile, [0.1, 0.05] if mode == "min" else [0.9, 0.95]
        )
        np.testing.assert_array_equal(rp.scipy_dist, ["gamma", "pearson3"])
        assert rp.streamflow.attrs["long_name"] == "Return period"
        assert (
            rp.streamflow.attrs["description"]
            == f"Return period ({mode}) estimated with statistic distributions"
        )
        assert rp.streamflow.attrs["cell_methods"] == "dparams: ppf"
        assert rp.streamflow.attrs["mode"] == mode

        ans = (
            [[190.66041057, 214.08102761], [185.78830382, 203.01731036]]
            if mode == "max"
            else [[66.00067153, 53.58658639], [64.23598869, 47.00660287]]
        )
        np.testing.assert_array_almost_equal(rp.streamflow, ans)


def test_criteria():
    ds = timeseries(
        np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
        variable="streamflow",
        start="2001-01-01",
        freq="YS",
        as_dataset=True,
    )
    params = xhfa.local.fit(ds, distributions=["gamma", "pearson3"])
    crit = xhfa.local.criteria(ds, params)
    crit_with_otherdim = xhfa.local.criteria(
        ds.expand_dims("otherdim"), params.expand_dims("otherdim")
    )
    np.testing.assert_array_almost_equal(
        crit.streamflow, crit_with_otherdim.streamflow.squeeze("otherdim")
    )

    np.testing.assert_array_equal(crit.scipy_dist, ["gamma", "pearson3"])
    np.testing.assert_array_equal(crit.criterion, ["aic", "bic", "aicc"])
    assert crit.streamflow.attrs["long_name"] == "Information criteria"
    assert (
        crit.streamflow.attrs["description"]
        == "Information criteria for the distribution parameters."
    )
    assert crit.streamflow.attrs["scipy_dist"] == ["gamma", "pearson3"]
    assert all(
        attr not in crit.streamflow.attrs
        for attr in ["estimator", "method", "min_years"]
    )

    np.testing.assert_array_almost_equal(
        crit.streamflow,
        [
            [118.19066549, 118.58856076, 118.63510993],
            [118.12140939, 118.51930466, 118.56585383],
        ],
    )


class TestGetPlottingPositions:
    def test_default(self):
        data = timeseries(
            np.array([50, 65, 80, 95]),
            variable="streamflow",
            start="2001-01-01",
            freq="YS",
            as_dataset=True,
        )
        expected = [1.16666667, 1.61538462, 2.625, 7.0]
        result = get_plotting_positions(data)
        np.testing.assert_array_almost_equal(result.streamflow_pp, expected)
        np.testing.assert_array_almost_equal(result.streamflow, data.streamflow)

        data_2d = xr.concat([data, data], dim="id")
        result = get_plotting_positions(data_2d)
        np.testing.assert_array_almost_equal(result.streamflow_pp, [expected, expected])
        np.testing.assert_array_equal(result.streamflow, data_2d.streamflow)

    def test_nan(self):
        data = timeseries(
            np.array([50, np.nan, 80, 95]),
            variable="streamflow",
            start="2001-01-01",
            freq="YS",
            as_dataset=True,
        )
        expected = [1.23076923, np.nan, 2.0, 5.33333333]
        result = get_plotting_positions(data)
        np.testing.assert_array_almost_equal(result.streamflow_pp, expected)
        np.testing.assert_array_almost_equal(result.streamflow, data.streamflow)

        data_2d = xr.concat([data, data], dim="id")
        result = get_plotting_positions(data_2d)
        np.testing.assert_array_almost_equal(result.streamflow_pp, [expected, expected])
        np.testing.assert_array_equal(result.streamflow, data_2d.streamflow)

    def test_return_period(self):
        data = timeseries(
            np.array([50, 65, 80, 95]),
            variable="streamflow",
            start="2001-01-01",
            freq="YS",
            as_dataset=True,
        )
        expected = [0.14285714, 0.38095238, 0.61904762, 0.85714286]
        result = get_plotting_positions(data, return_period=False)
        np.testing.assert_array_almost_equal(result.streamflow_pp, expected)
        np.testing.assert_array_almost_equal(result.streamflow, data.streamflow)

        data_2d = xr.concat([data, data], dim="id")
        result = get_plotting_positions(data_2d, return_period=False)
        np.testing.assert_array_almost_equal(result.streamflow_pp, [expected, expected])
        np.testing.assert_array_equal(result.streamflow, data_2d.streamflow)

    def test_alpha_beta(self):
        data = timeseries(
            np.array([50, 65, 80, 95]),
            variable="streamflow",
            start="2001-01-01",
            freq="YS",
            as_dataset=True,
        )
        expected = [1.25, 1.66666667, 2.5, 5.0]
        result = get_plotting_positions(data, alpha=0, beta=0)
        np.testing.assert_array_almost_equal(result.streamflow_pp, expected)
        np.testing.assert_array_almost_equal(result.streamflow, data.streamflow)

        data_2d = xr.concat([data, data], dim="id")
        result = get_plotting_positions(data_2d, alpha=0, beta=0)
        np.testing.assert_array_almost_equal(result.streamflow_pp, [expected, expected])
        np.testing.assert_array_equal(result.streamflow, data_2d.streamflow)
