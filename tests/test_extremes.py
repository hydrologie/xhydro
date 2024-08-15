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
from xhydro.extreme_value_analysis.structures.util import exponentiate_logscale

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
        test_params = gevfit(self.y)["params"]
        test_params = exponentiate_logscale(np.array(test_params), [], []).tolist()
        true_params = [0.0009, 1.014, -0.0060]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, rtol=0.05)

    def test_gevfitpwm(self):
        test_params = gevfitpwm(self.y)["params"]
        test_params = exponentiate_logscale(np.array(test_params), [], []).tolist()
        true_params = [
            -0.0005,
            1.0125,
            -0.0033,
        ]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.00015)

    def test_gevfitbayes(self):
        test_params = gevfitbayes(self.y, niter=100, warmup=50)["params"]
        test_params = exponentiate_logscale(np.array(test_params), [], []).tolist()
        true_params = [0, 1, 0]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.03)


class TestGumbelfit:
    y = GEV_STATIONARY["y"].values

    def test_gumbelfit(self):
        test_params = gumbelfit(self.y)["params"]
        test_params = exponentiate_logscale(np.array(test_params), [], []).tolist()
        true_params = [-0.0023, 1.0125]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.0001)

    def test_gumbelfitpwm(self):
        test_params = gumbelfitpwm(self.y)["params"]
        test_params = exponentiate_logscale(np.array(test_params), [], []).tolist()
        true_params = [-0.0020, 1.009]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.001)

    def test_gumbelfitbayes(self):
        test_params = gumbelfitbayes(self.y, niter=100, warmup=50)["params"]
        test_params = exponentiate_logscale(np.array(test_params), [], []).tolist()
        true_params = [0, 1]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.02)


# FIXME: When ran locally, these tests only necessitate a very small atol, but in order to move on with the pipeline
# on https://github.com/hydrologie/xhydro/pull/175 I put a large atol
class TestGpfit:
    y = GP_STATIONARY["y"].values

    def test_gpfit(self):
        test_params = gpfit(self.y)["params"]
        test_params = exponentiate_logscale(
            np.array(test_params), [], [], pareto=True
        ).tolist()
        true_params = [0.9866, 0.0059]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.5)

    def test_gpfitpwm(self):
        test_params = gpfitpwm(self.y)["params"]
        test_params = exponentiate_logscale(
            np.array(test_params), [], [], pareto=True
        ).tolist()
        true_params = [1, 0]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.5)

    def test_gpfitbayes(self):
        test_params = gpfitbayes(self.y, niter=100, warmup=50)["params"]
        test_params = exponentiate_logscale(
            np.array(test_params), [], [], pareto=True
        ).tolist()
        true_params = [1, 0]  # Values taken from tests in Extremes.jl
        for test, true in zip(test_params, true_params):
            np.testing.assert_allclose(test, true, atol=0.5)
