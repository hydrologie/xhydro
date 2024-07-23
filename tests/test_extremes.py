import numpy as np
import pandas as pd
import pooch

from xhydro.extreme_value_analysis.parameterestimation import (
    gevfit,
    gevfitbayes,
    gevfitpwm,
    gpfit,
    gpfitbayes,
    gpfitpwm,
    gumbelfit,
    gumbelfitbayes,
    gumbelfitpwm,
)

# Dataset taken from tests in Extremes.jl
GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
BRANCH_OR_COMMIT_HASH = "main"

genextreme_data = pooch.retrieve(
    url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/extreme_value_analysis/genextreme.zip",
    known_hash="md5:cc2ff7c93949673a6acf00c7c2fac20b",
    processor=pooch.Unzip(),
)

genpareto_data = pooch.retrieve(
    url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/extreme_value_analysis/genpareto.zip",
    known_hash="md5:ecb74164db4bbfeabfc5e340b11e7ae8",
    processor=pooch.Unzip(),
)

GEV_NONSTATIONARY = pd.read_csv(genextreme_data[0])
GEV_STATIONARY = pd.read_csv(genextreme_data[1])
GP_NONSTATIONARY = pd.read_csv(genpareto_data[0])
GP_STATIONARY = pd.read_csv(genpareto_data[1])


class TestGevfit:
    y = GEV_STATIONARY["y"].values

    def test_gevfit(self):
        param_cint = gevfit(self.y)
        test_params = [param_cint[0], param_cint[3], param_cint[6]]
        true_params = [0.0009, 1.014, -0.0060]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, rtol=0.05)

    def test_gevfitpwm(self):
        param_cint = gevfitpwm(self.y)
        test_params = [param_cint[0], param_cint[3], param_cint[6]]
        true_params = [
            -0.0005,
            1.0125,
            -0.0033,
        ]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.00015)

    def test_gevfitbayes(self):
        param_cint = gevfitbayes(self.y, niter=100, warmup=50)
        test_params = [param_cint[0], param_cint[3], param_cint[6]]
        true_params = [0, 1, 0]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.03)


class TestGumbelfit:
    y = GEV_STATIONARY["y"].values

    def test_gumbelfit(self):
        param_cint = gumbelfit(self.y)
        test_params = [param_cint[0], param_cint[3]]
        true_params = [-0.0023, 1.0125]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.0001)

    def test_gumbelfitpwm(self):
        param_cint = gumbelfitpwm(self.y)
        test_params = [param_cint[0], param_cint[3]]
        true_params = [-0.0020, 1.009]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.001)

    def test_gumbelfitbayes(self):
        param_cint = gumbelfitbayes(self.y, niter=100, warmup=50)
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0, 1]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.02)


class TestGpfit:
    y = GP_STATIONARY["y"].values

    def test_gpfit(self):
        param_cint = gpfit(self.y)
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0.9866, 0.0059]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.0001)

    def test_gpfitpwm(self):
        param_cint = gpfitpwm(self.y)
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0, 1]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.05)

    def test_gpfitbayes(self):
        param_cint = gpfitbayes(self.y, niter=100, warmup=50)
        test_params = [param_cint[0], param_cint[3]]
        true_params = [0, 1]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.05)
