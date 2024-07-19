import numpy as np
import pandas as pd

from src.xhydro.extreme_value_analysis import (
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

GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
# TODO: branch and folder names are the same until the PR https://github.com/hydrologie/xhydro-testdata/pull/13
# is accepted and branch becomes 'main'
BRANCH_OR_COMMIT_HASH = "extreme_value_analysis"
FOLDER = "extreme_value_analysis"
FILE_URLS = {
    "gev_stationary": f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/{FOLDER}/gev_stationary.csv",
    "gev_nonstationary": f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/{FOLDER}/gev_nonstationary.csv",
    "gp_stationary": f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/{FOLDER}/gp_stationary.csv",
    "gp_nonstationary": f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/{FOLDER}/gp_nonstationary.csv",
}


class TestGevfit:
    # Dataset taken from tests in Extremes.jl
    csv_url = FILE_URLS["gev_stationary"]
    df = pd.read_csv(csv_url)
    y = df["y"].values

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
    # Dataset taken from tests in Extremes.jl
    csv_url = FILE_URLS["gev_stationary"]
    df = pd.read_csv(csv_url)
    y = df["y"].values

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
    # Dataset taken from tests in Extremes.jl
    csv_url = FILE_URLS["gp_stationary"]
    df = pd.read_csv(csv_url)
    y = df["y"].values

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
