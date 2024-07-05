import numpy as np
import pytest
import xarray as xr
import pandas as pd
from scipy.stats import genextreme, gumbel_r, genpareto # to generate samples of given distribution
from xhydro_temp.extreme_value_analysis.parameterestimation import *


class TestGevfit:
    (shape_true, loc_true, scale_true) = (0.1, 55, 125)
    df = pd.read_csv('genextreme.csv')
    y = df['y'].values

    df_small = pd.read_csv('genextreme_small.csv') # smaller dataset for bayes inference
    y_small = df_small['y'].values
    def test_gevfit(self):
        (shape_test, loc_test, scale_test) = tuple(np.roll(gevfit(self.y),1))
        np.testing.assert_allclose(shape_test, -self.shape_true, rtol=0.05) #TODO: remove "-" in front of shape_true once xclim issue is resolved
        np.testing.assert_allclose(loc_test, self.loc_true, rtol=0.07) #TODO: dont have different tolerances for each parameter
        np.testing.assert_allclose(scale_test, self.scale_true, rtol=0.05)

    def test_gevfitpwm(self):
        (shape_test, loc_test, scale_test) = tuple(np.roll(gevfitpwm(self.y),1))
        np.testing.assert_allclose(shape_test, -self.shape_true, rtol=0.05) #TODO: remove "-" in front of shape_true once xclim issue is resolved
        np.testing.assert_allclose(loc_test, self.loc_true, rtol=0.07) #TODO: dont have different tolerances for each parameter
        np.testing.assert_allclose(scale_test, self.scale_true, rtol=0.05)

    def test_gevfitbayes(self):
        (shape_test, loc_test, scale_test) = tuple(np.roll(gevfitbayes(self.y_small),1))
        np.testing.assert_allclose(shape_test, -self.shape_true, rtol=0.11) #TODO: remove "-" in front of shape_true once xclim issue is resolved + dont have different tolerances for each parameter
        np.testing.assert_allclose(loc_test, self.loc_true, rtol=0.07) #TODO: dont have different tolerances for each parameter
        np.testing.assert_allclose(scale_test, self.scale_true, rtol=0.05)



class TestGumbelfit:
    (loc_true, scale_true) = (80, 30)
    df = pd.read_csv('gumbel_r.csv')
    y = df['y'].values

    df_small = pd.read_csv('gumbel_r.csv') # smaller dataset for bayes inference
    y_small = df_small['y'].values
    def test_gumbelfit(self):
        (loc_test, scale_test) = tuple(gumbelfit(self.y))
        np.testing.assert_allclose(loc_test, self.loc_true, rtol=0.05)
        np.testing.assert_allclose(scale_test, self.scale_true, rtol=0.05)

    def test_gumbelfitpwm(self):
        (loc_test, scale_test) = tuple(gumbelfitpwm(self.y))
        np.testing.assert_allclose(loc_test, self.loc_true, rtol=0.05)
        np.testing.assert_allclose(scale_test, self.scale_true, rtol=0.05)

    def test_gumbelfitbayes(self):
        (loc_test, scale_test) = tuple(gumbelfitbayes(self.y_small))
        np.testing.assert_allclose(loc_test, self.loc_true, rtol=0.05)
        np.testing.assert_allclose(scale_test, self.scale_true, rtol=0.05)

class TestGpfit:
    (shape_true, loc_true, scale_true) = (0.1, 55, 125)
    df = pd.read_csv('genpareto.csv')
    y = df['y'].values

    df_small = pd.read_csv('genpareto_small.csv') # smaller dataset for bayes inference
    y_small = df_small['y'].values

    def test_gpfit(self):
        self.y = values_above_threshold(self.y, 0.05)
        (shape_test, loc_test, scale_test) = tuple(np.roll(gpfit(self.y), 1))

        print(self.shape_true, self.loc_true, self.scale_true)
        print(shape_test, loc_test, scale_test)
        # np.testing.assert_allclose(shape_test, -self.shape_true, rtol=0.05) #TODO: remove "-" in front of shape_true once xclim issue is resolved
        # np.testing.assert_allclose(loc_test, self.loc_true, rtol=0.07) #TODO: dont have different tolerances for each parameter
        # np.testing.assert_allclose(scale_test, self.scale_true, rtol=0.05)

    def test_gpfitpwm(self):


        pass
