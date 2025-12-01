import numpy as np
import pytest
import xarray as xr


@pytest.mark.requires_julia
class Testfit:
    def test_stationary_gev(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(data_fre, dist="genextreme", method="ml", variables=["SeaLevel"], dim="Year").compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)

        np.testing.assert_array_equal(p.coords["dparams"], np.array(["shape", "loc", "scale"]))
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])
        np.testing.assert_array_almost_equal(
            p.to_array(),
            np.array(
                [
                    [0.21742187, 1.48234069, 0.14127096],
                    [0.09241566, 1.44956008, 0.12044046],
                    [0.34242808, 1.5151213, 0.16570414],
                ]
            ),
        )

    def test_stationary_gumbel(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(data_fre, dist="gumbel_r", method="ml", variables=["SeaLevel"], dim="Year").compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_equal(p.coords["dparams"], np.array(["loc", "scale"]))
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])
        np.testing.assert_array_almost_equal(
            p.to_array(),
            np.array(
                [
                    [1.4662802, 0.13941243],
                    [1.43506132, 0.11969685],
                    [1.49749908, 0.1623754],
                ]
            ),
        )

    def test_stationary_pareto(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_rain,
            dist="genpareto",
            method="ml",
            variables=["Exceedance"],
            dim="Date",
        ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p["Exceedance_lower"].attrs["method"] == "Maximum likelihood"
        np.testing.assert_array_equal(p["Exceedance_lower"] < p["Exceedance"], True)
        np.testing.assert_array_equal(p["Exceedance"] < p["Exceedance_upper"], True)
        np.testing.assert_array_equal(p.coords["dparams"], np.array(["scale", "shape"]))
        np.testing.assert_array_equal(list(p.data_vars), ["Exceedance", "Exceedance_lower", "Exceedance_upper"])
        np.testing.assert_array_almost_equal(
            p.to_array(),
            np.array(
                [
                    [7.44028211, 0.18449556],
                    [5.78001519, -0.01385888],
                    [9.57744851, 0.38284999],
                ]
            ),
        )

    def test_non_stationary_cov(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_fre,
            dist="genextreme",
            method="ml",
            variables=["SeaLevel"],
            dim="Year",
            locationcov=["Year"],
        ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_equal(
            p.coords["dparams"],
            np.array(["shape", "loc", "loc_Year_covariate", "scale"]),
        )
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])
        np.testing.assert_array_almost_equal(
            p.to_array(),
            np.array(
                [
                    [1.25309158e-01, -2.47280133e00, 2.03216816e-03, 1.24326357e-01],
                    [-1.13711503e-02, -4.44908773e00, 1.01747569e-03, 1.05446536e-01],
                    [2.61989466e-01, -4.96514942e-01, 3.04686063e-03, 1.46586541e-01],
                ]
            ),
        )

    def test_non_stationary_cov2(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_fre,
            dist="genextreme",
            method="ml",
            variables=["SeaLevel"],
            dim="Year",
            locationcov=["Year", "SOI"],
            scalecov=["Year", "SOI"],
        ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_equal(
            p.coords["dparams"],
            np.array(
                [
                    "shape",
                    "loc",
                    "loc_Year_covariate",
                    "loc_SOI_covariate",
                    "scale",
                    "scale_Year_covariate",
                    "scale_SOI_covariate",
                ]
            ),
        )
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [
                        2.22927569e-01,
                        -2.21521222e00,
                        1.90673206e-03,
                        6.59271322e-02,
                        8.63461296e02,
                        9.95445509e-01,
                        1.30358008e00,
                    ],
                    [
                        7.44795732e-02,
                        -4.01218652e00,
                        9.86301542e-04,
                        3.15126826e-02,
                        6.20280622e-02,
                        9.90579605e-01,
                        1.04008160e00,
                    ],
                    [
                        3.71375565e-01,
                        -4.18237918e-01,
                        2.82716257e-03,
                        1.00341582e-01,
                        12019808.185356595,
                        1.00033532e00,
                        1.63383432e00,
                    ],
                ]
            ),
        )

    def test_non_stationary_cov_pwm(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="Probability weighted moment parameter estimation cannot have covariates \\['Year', 'SOI', 'Year2', 'SOI2', 'Year3', 'SOI3'\\]",
        ):
            evap.fit(
                data_fre,
                dist="genextreme",
                method="pwm",
                variables=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year2", "SOI2"],
                shapecov=["Year3", "SOI3"],
            ).compute(scheduler="processes")

    def test_fewer_entries_than_param_param(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.warns(
            UserWarning,
            match="The fitting data contains fewer entries than the number of parameters for the given distribution. "
            "Returned parameters are numpy.nan",
        ):
            p = evap.fit(
                data_fre.isel(Year=slice(1, 2)),
                dist="genextreme",
                method="ml",
                variables=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year", "SOI"],
            ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        np.testing.assert_array_equal(np.isnan(p.to_array()), True)
        np.testing.assert_array_equal(
            p.coords["dparams"],
            np.array(
                [
                    "shape",
                    "loc",
                    "loc_Year_covariate",
                    "loc_SOI_covariate",
                    "scale",
                    "scale_Year_covariate",
                    "scale_SOI_covariate",
                ]
            ),
        )
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])

    def test_fewer_entries_than_param_rtnlev(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.warns(
            UserWarning,
            match="The fitting data contains fewer entries than the number of parameters for the given distribution. "
            "Returned parameters are numpy.nan",
        ):
            p = evap.return_level(
                data_fre.isel(Year=slice(0, 5)),
                dist="genextreme",
                method="ml",
                variables=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year", "SOI"],
            ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["Year", "return_period"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        np.testing.assert_array_equal(np.isnan(p.to_array()), True)
        np.testing.assert_array_equal(p.coords["Year"], np.array([1897, 1898, 1899, 1900, 1901]))
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])


@pytest.mark.requires_julia
class TestRtnlv:
    def test_stationary(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_fre,
            dist="genextreme",
            method="ml",
            variables=["SeaLevel"],
            dim="Year",
            return_period=100,
        ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["return_period"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        assert p.to_array().shape[1] == 1
        assert ((p["SeaLevel_lower"] < p["SeaLevel"]) & (p["SeaLevel"] < p["SeaLevel_upper"])).values.all()
        np.testing.assert_array_equal(p.coords["return_period"], np.array([100]))
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])
        np.testing.assert_array_almost_equal(p.to_array(), np.array([[1.89310525], [1.81018661], [1.9760239]]))

    def test_non_stationary_cov(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_fre,
            dist="genextreme",
            method="ml",
            variables=["SeaLevel"],
            dim="Year",
            locationcov=["Year", "SOI"],
            return_period=100,
        ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["Year", "return_period"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        assert len(data_fre["SeaLevel"]) == p.to_array().shape[1]
        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_equal(p.coords["Year"].values[:3], np.array([1897, 1898, 1899]))
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])
        np.testing.assert_array_almost_equal(
            p.to_array()[:, :5],
            np.array(
                [
                    [1.74899503, 1.81870904, 1.79847138, 1.75642726, 1.7972477],
                    [1.64648473, 1.72510362, 1.70486945, 1.65503958, 1.70372846],
                    [1.85150534, 1.91231447, 1.89207332, 1.85781493, 1.89076694],
                ]
            ),
        )

    def test_recover_nan(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        data_fre_nan = data_fre.copy(deep=True)
        data_fre_nan["SeaLevel"].isel(Year=slice(1, 3))[:] = np.nan
        data_fre_nan["SOI"].isel(Year=slice(5, 6))[:] = np.nan

        p = evap.return_level(
            data_fre_nan,
            dist="genextreme",
            method="ml",
            variables=["SeaLevel"],
            dim="Year",
            locationcov=["Year", "SOI"],
            return_period=100,
        ).compute(scheduler="processes")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["Year", "return_period"]
        assert p["SeaLevel_lower"].attrs["method"] == "Maximum likelihood"
        assert len(data_fre["SeaLevel"]) == p.to_array().shape[1]
        np.testing.assert_array_equal(p.coords["Year"].values[:3], np.array([1897, 1898, 1899]))
        np.testing.assert_array_equal(list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"])
        np.testing.assert_array_almost_equal(
            p.to_array()[:, :7],
            np.array(
                [
                    [
                        1.75452062,
                        np.nan,
                        np.nan,
                        1.76138234,
                        1.80296834,
                        np.nan,
                        1.8271513,
                    ],
                    [
                        1.64450866,
                        np.nan,
                        np.nan,
                        1.65251713,
                        1.7003416,
                        np.nan,
                        1.72609747,
                    ],
                    [
                        1.86453259,
                        np.nan,
                        np.nan,
                        1.87024755,
                        1.90559507,
                        np.nan,
                        1.92820514,
                    ],
                ]
            ),
        )


@pytest.mark.requires_julia
class TestGEV:
    def test_ml_param(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(data_fre, dist="genextreme", method="ML", dim="Year", variables=["SeaLevel"]).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)

        np.testing.assert_array_almost_equal(
            p.to_array(),
            np.array(
                [
                    [0.21742187, 1.48234069, 0.14127096],
                    [0.09241566, 1.44956008, 0.12044046],
                    [0.34242808, 1.5151213, 0.16570414],
                ]
            ),
        )

    def test_pwm_param(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_fre,
            dist="genextreme",
            method="PWM",
            dim="Year",
            variables=["SeaLevel"],
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_almost_equal(p.to_array().values[0], np.array([0.19631625, 1.4807491, 0.13907873]))

    def test_bayes_param(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_fre,
            dist="genextreme",
            method="BAYES",
            dim="Year",
            variables=["SeaLevel"],
            niter=50,
            warmup=20,
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)

    def test_ml_rtnlv(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(data_fre, dist="genextreme", method="ML", dim="Year", variables=["SeaLevel"]).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_almost_equal(p.to_array().values, np.array([[1.89310525], [1.81018661], [1.9760239]]))

    def test_pwm_rtnlv(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_fre,
            dist="genextreme",
            method="PWM",
            dim="Year",
            variables=["SeaLevel"],
        ).compute(scheduler="processes")

        np.testing.assert_array_almost_equal(p.to_array().values[0], np.array([1.90204717]))
        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)

    def test_bayes_rtnlv(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_fre,
            dist="genextreme",
            method="BAYES",
            dim="Year",
            variables=["SeaLevel"],
            niter=50,
            warmup=20,
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)


@pytest.mark.requires_julia
class TestGumbel:
    def test_ml_param(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(data_fre, dist="gumbel_r", method="ML", dim="Year", variables=["SeaLevel"]).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [1.4662802, 0.13941243],
                    [1.43506132, 0.11969685],
                    [1.49749908, 0.1623754],
                ]
            ),
        )

    def test_pwm_param(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(data_fre, dist="gumbel_r", method="PWM", dim="Year", variables=["SeaLevel"]).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_almost_equal(p.to_array().values[0], np.array([1.46903519, 0.1195187]))

    def test_bayes_param(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_fre,
            dist="gumbel_r",
            method="BAYES",
            dim="Year",
            variables=["SeaLevel"],
            niter=50,
            warmup=20,
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)

    def test_ml_rtnlv(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(data_fre, dist="gumbel_r", method="ML", dim="Year", variables=["SeaLevel"]).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)
        np.testing.assert_array_almost_equal(p.to_array(), np.array([[2.10759817], [1.99555233], [2.21964401]]))

    def test_pwm_rtnlv(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(data_fre, dist="gumbel_r", method="PWM", dim="Year", variables=["SeaLevel"]).compute(scheduler="processes")

        np.testing.assert_array_almost_equal(p.to_array().values[0], np.array([2.01883904]))
        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)

    def test_bayes_rtnlv(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_fre,
            dist="gumbel_r",
            method="BAYES",
            dim="Year",
            variables=["SeaLevel"],
            niter=50,
            warmup=20,
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["SeaLevel_lower"] < p["SeaLevel"], True)
        np.testing.assert_array_equal(p["SeaLevel"] < p["SeaLevel_upper"], True)


@pytest.mark.requires_julia
class TestPareto:
    def test_ml_param(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_rain,
            dist="genpareto",
            method="ML",
            dim="Date",
            variables=["Exceedance"],
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["Exceedance_lower"] < p["Exceedance"], True)
        np.testing.assert_array_equal(p["Exceedance"] < p["Exceedance_upper"], True)
        np.testing.assert_array_almost_equal(
            p.to_array(),
            np.array(
                [
                    [7.44028211, 0.18449556],
                    [5.78001519, -0.01385888],
                    [9.57744851, 0.38284999],
                ]
            ),
        )

    def test_pwm_param(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_rain,
            dist="genpareto",
            method="PWM",
            dim="Date",
            variables=["Exceedance"],
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["Exceedance_lower"] < p["Exceedance"], True)
        np.testing.assert_array_equal(p["Exceedance"] < p["Exceedance_upper"], True)
        np.testing.assert_array_almost_equal(p.to_array().values[0], np.array([7.29901897, 0.19651587]))

    def test_bayes_param(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.fit(
            data_rain,
            dist="genpareto",
            method="BAYES",
            variables=["Exceedance"],
            dim="Date",
            niter=50,
            warmup=20,
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["Exceedance_lower"] < p["Exceedance"], True)
        np.testing.assert_array_equal(p["Exceedance"] < p["Exceedance_upper"], True)

    def test_ml_rtnlv(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_rain,
            dist="genpareto",
            method="ML",
            dim="Date",
            variables=["Exceedance"],
            threshold_pareto=30,
            nobs_pareto=17531,
            nobsperblock_pareto=365,
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["Exceedance_lower"] < p["Exceedance"], True)
        np.testing.assert_array_equal(p["Exceedance"] < p["Exceedance_upper"], True)
        np.testing.assert_array_almost_equal(
            p.to_array(),
            np.array([[106.32726047], [65.48118115], [147.1733398]]),
        )

    def test_pwm_rtnlv(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_rain,
            dist="genpareto",
            method="PWM",
            dim="Date",
            variables=["Exceedance"],
            threshold_pareto=30,
            nobs_pareto=17531,
            nobsperblock_pareto=365,
        ).compute(scheduler="processes")
        np.testing.assert_array_almost_equal(p.to_array().values[0], np.array([107.99657119]))
        np.testing.assert_array_equal(p["Exceedance_lower"] < p["Exceedance"], True)
        np.testing.assert_array_equal(p["Exceedance"] < p["Exceedance_upper"], True)

    def test_bayes_rtnlv(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        p = evap.return_level(
            data_rain,
            dist="genpareto",
            method="BAYES",
            dim="Date",
            variables=["Exceedance"],
            threshold_pareto=30,
            nobs_pareto=17531,
            nobsperblock_pareto=365,
            niter=50,
            warmup=20,
        ).compute(scheduler="processes")

        np.testing.assert_array_equal(p["Exceedance_lower"] < p["Exceedance"], True)
        np.testing.assert_array_equal(p["Exceedance"] < p["Exceedance_upper"], True)


@pytest.mark.requires_julia
class TestError:
    dataset = xr.Dataset(
        {"n": ("pos", [1, -1, 1, -1, 1, -1]), "n2": ("pos", [2, -2, 2, -2, 2, -2])},
        coords={"pos": [0, 1, 2, 3, 4, -5]},
    )

    def test_non_stationary_cov_pwm(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="Probability weighted moment parameter estimation cannot have covariates \\['Year', 'SOI', 'Year2', 'SOI2', 'Year3', 'SOI3'\\]",
        ):
            evap.return_level(
                data_fre,
                dist="genextreme",
                method="pwm",
                variables=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year2", "SOI2"],
                shapecov=["Year3", "SOI3"],
            ).compute(scheduler="processes")

    def test_extremefit_rtnlv_error(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(ValueError, match="Unrecognized distribution: XXX"):
            evap.return_level(
                data_rain,
                dist="XXX",
                method="BAYES",
                dim="Date",
                variables=["Exceedance"],
                threshold_pareto=30,
                nobs_pareto=17531,
                nobsperblock_pareto=365,
                niter=50,
                warmup=20,
            ).compute(scheduler="processes")

    def test_extremefit_param_error(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(ValueError, match="Unrecognized method: YYY"):
            evap.return_level(
                data_rain,
                dist="genpareto",
                method="YYY",
                dim="Date",
                variables=["Exceedance"],
                threshold_pareto=30,
                nobs_pareto=17531,
                nobsperblock_pareto=365,
            ).compute(scheduler="processes")

    def test_confinc_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.warns(UserWarning, match="There was an error in computing confidence interval."):
            p = evap.fit(
                data_fre,
                dist="genpareto",
                method="ML",
                variables=["SeaLevel"],
                dim="Year",
            ).compute(scheduler="processes")

        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array([[2.87959631, -1.49978975], [np.nan, np.nan], [np.nan, np.nan]]),
        )

    def test_confinc_error_nostat(self):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.warns(
            UserWarning,
            match="There was an error in fitting the data to a genpareto distribution using ML. Returned parameters are numpy.nan",
        ):
            p = evap.fit(
                self.dataset,
                dist="genpareto",
                method="ML",
                dim="pos",
                scalecov=["n2"],
                variables=["n"],
            ).compute(scheduler="processes")
            np.testing.assert_array_equal(
                p.to_array(),
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
            )

    def test_confinc_error_nostat_rtnlv(self):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.warns(
            UserWarning,
            match="There was an error in fitting the data to a genpareto distribution using BAYES. Returned parameters are numpy.nan",
        ):
            p = evap.fit(
                self.dataset,
                dist="genpareto",
                method="BAYES",
                dim="pos",
                scalecov=["n2"],
                variables=["n"],
                niter=50,
                warmup=20,
            ).compute(scheduler="processes")

            np.testing.assert_array_equal(
                p.to_array(),
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
            )

    def test_rtnlv_error_nostat_rtnlv(self):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.warns(
            UserWarning,
            match="There was an error in fitting the data to a genextreme distribution using BAYES. Returned parameters are numpy.nan",
        ):
            p = evap.return_level(
                self.dataset,
                dist="genextreme",
                method="BAYES",
                dim="pos",
                scalecov=["n2"],
                variables=["n"],
                niter=50,
                warmup=20,
            ).compute(scheduler="processes")

            np.testing.assert_array_equal(
                p.to_array(),
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            )

    def test_dist_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(ValueError, match="Unrecognized distribution: XXX"):
            evap.fit(data_fre, dist="XXX", method="ml", variables=["SeaLevel"], dim="Year").compute(scheduler="processes")

    def test_method_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(ValueError, match="Unrecognized method: YYY"):
            evap.fit(
                data_fre,
                dist="genextreme",
                method="YYY",
                variables=["SeaLevel"],
                dim="Year",
            ).compute(scheduler="processes")

    def test_vars_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="XXXX is not a variable in the Dataset. Dataset variables are: \\['SOI', 'SeaLevel'\\]",
        ):
            evap.fit(data_fre, dist="genextreme", variables=["XXXX"], dim="Year").compute(scheduler="processes")

    def test_conflev_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="Confidence level must be strictly smaller than 1 and strictly larger than 0",
        ):
            evap.fit(
                data_fre,
                dist="genextreme",
                variables=["SeaLevel"],
                dim="Year",
                confidence_level=2,
            ).compute(scheduler="processes")

    def test_gumbel_cov_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="Gumbel distribution has no shape parameter and thus cannot have shape covariates \\['Year'\\]",
        ):
            evap.fit(
                data_fre,
                dist="gumbel_r",
                method="ml",
                variables=["SeaLevel"],
                dim="Year",
                shapecov=["Year"],
            ).compute(scheduler="processes")

    def test_pareto_cov_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="Pareto distribution has no location parameter and thus cannot have location covariates \\['Year'\\]",
        ):
            evap.return_level(
                data_fre,
                dist="genpareto",
                method="ml",
                variables=["SeaLevel"],
                dim="Year",
                locationcov=["Year"],
            ).compute(scheduler="processes")

    def test_rtmper_error(self, data_fre):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="Return period has to be strictly larger than 0. Current return period value is -1",
        ):
            evap.return_level(
                data_fre,
                dist="genextreme",
                variables=["SeaLevel"],
                dim="Year",
                return_period=-1,
            ).compute(scheduler="processes")

    def test_para_pareto_error(self, data_rain):
        evap = pytest.importorskip("xhydro.extreme_value_analysis.parameterestimation")

        with pytest.raises(
            ValueError,
            match="'threshold_pareto', 'nobs_pareto', and 'nobsperblock_pareto' must be defined when using dist 'genpareto'.",
        ):
            evap.return_level(
                data_rain,
                dist="genpareto",
                method="ml",
                variables=["Exceedance"],
                dim="Date",
            ).compute(scheduler="processes")
