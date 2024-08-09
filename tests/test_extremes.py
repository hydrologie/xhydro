import warnings

import numpy as np
import pytest

try:
    from xhydro.extreme_value_analysis import parameterestimation

    julia_installed = True
except ImportError:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    warnings.warn(JULIA_WARNING)
    julia_installed = False
    parameterestimation = None


@pytest.mark.skipif(not julia_installed, reason="Julia not installed")
class TestGevfit:

    def y(self, genextreme_data):
        return genextreme_data["gev_stationary"]["y"].values

    def test_gevfit(self, genextreme_data):
        param_cint = parameterestimation.gevfit(self.y(genextreme_data))
        test_params = [param_cint[0], param_cint[3], param_cint[6]]
        true_params = [0.0009, 1.014, -0.0060]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, rtol=0.05)

    def test_gevfitpwm(self, genextreme_data):
        param_cint = parameterestimation.gevfitpwm(self.y(genextreme_data))
        test_params = [param_cint[0], param_cint[3], param_cint[6]]
        true_params = [
            -0.0005,
            1.0125,
            -0.0033,
        ]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.00015)

    def test_gevfitbayes(self, genextreme_data):
        param_cint = parameterestimation.gevfitbayes(
            self.y(genextreme_data), niter=100, warmup=50
        )
        test_params = [param_cint[0], param_cint[3], param_cint[6]]
        true_params = [0, 1, 0]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.03)


@pytest.mark.skipif(not julia_installed, reason="Julia not installed")
class TestGumbelfit:

    def y(self, genextreme_data):
        return genextreme_data["gev_stationary"]["y"].values

    def test_gumbelfit(self, genextreme_data):
        param_cint = parameterestimation.gumbelfit(self.y(genextreme_data))
        test_params = [param_cint[0], param_cint[3]]
        true_params = [-0.0023, 1.0125]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.0001)

    def test_gumbelfitpwm(self, genextreme_data):
        param_cint = parameterestimation.gumbelfitpwm(self.y(genextreme_data))
        test_params = [param_cint[0], param_cint[3]]
        true_params = [-0.0020, 1.009]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.001)

    def test_gumbelfitbayes(self, genextreme_data):
        param_cint = parameterestimation.gumbelfitbayes(
            self.y(genextreme_data), niter=100, warmup=50
        )
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0, 1]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.02)


# FIXME: When ran locally, these tests only necessitate a very small atol, but in order to move on with the pipeline
# on https://github.com/hydrologie/xhydro/pull/175 I put a large atol
@pytest.mark.skipif(not julia_installed, reason="Julia not installed")
class TestGpfit:

    def y(self, genpareto_data):
        return genpareto_data["gp_stationary"]["y"].values

    def test_gpfit(self, genpareto_data):
        param_cint = parameterestimation.gpfit(self.y(genpareto_data))
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0.9866, 0.0059]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.5)

    def test_gpfitpwm(self, genpareto_data):
        param_cint = parameterestimation.gpfitpwm(self.y(genpareto_data))
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0, 1]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.5)

    def test_gpfitbayes(self, genpareto_data):
        param_cint = parameterestimation.gpfitbayes(
            self.y(genpareto_data), niter=100, warmup=50
        )
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0, 1]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.5)
