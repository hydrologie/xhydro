import warnings

import numpy as np
import pandas as pd
import pytest
import scipy
import xarray as xr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

try:
    from lmoments3.distr import KappaGen
except ImportError:
    warnings.warn("lmoments3 is not installed. Please install it")
    lmoments3 = None

from xhydro.frequency_analysis.regional import (
    _moment_l_vector,
    calc_h_z,
    calculate_rp_from_afr,
    cluster_indices,
    fit_pca,
    get_group_from_fit,
    get_groups_indices,
)


class TestRegionalFrequencyAnalysis:

    @pytest.fixture
    def sample_data(self):
        return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    @pytest.fixture
    def sample_dataset(self):
        data = np.random.rand(100, 5)
        return xr.Dataset({"data": (("time", "Station"), data)})

    def test_cluster_indices(self):
        clusters = np.array([0, 1, 0, 2, 1])
        expected = [0, 2]
        result = cluster_indices(clusters, 0)
        np.testing.assert_array_equal(result, expected)

    def test_get_group_from_fit(self):
        df = np.array(
            [
                [2.67035877, 1.75919216, 0.76206293],
                [1.06732866, 2.05965305, 0.38237151],
                [-2.03313717, 2.72214445, -1.10333638],
                [0.23552091, 2.30462729, -0.480144],
                [2.28301367, 1.79640619, 0.05511961],
            ]
        )
        station = np.array(["020302", "020404", "020502", "020602", "020802"])
        components = [0, 1, 2]
        data = xr.DataArray(
            data=df, coords=[station, components], dims=["Station", "components"]
        )
        expected = [["020404", "020602"], ["020502"], ["020302", "020802"]]
        result = get_group_from_fit(AgglomerativeClustering, {"n_clusters": 3}, data)
        # return result
        assert len(result) == len(expected)
        for i, list_st in enumerate(result):
            assert (list_st == expected[i]).all()

    def test_fit_pca(self, sample_dataset):
        data_pca, pca_obj = fit_pca(sample_dataset, n_components=3)
        assert isinstance(data_pca, xr.DataArray)
        assert isinstance(pca_obj, PCA)
        assert data_pca.shape[1] == 3

    def test__moment_l_vector(self, sample_data):
        result = _moment_l_vector(sample_data)
        count = 0
        for element in result:
            count += len(element)
        assert isinstance(result, list)
        assert len(result) == 3
        assert count == 18

    # def test_moment_l(self, sample_data):
    #     result = moment_l(sample_data[0])
    #     assert isinstance(result, dict)
    #     assert set(result.keys()) == {'l1', 'l2', 'l3', 'l4', 't', 't3', 't4'}

    # def test_moment_l_empty_data(self):
    #     with pytest.raises(ValueError):
    #         moment_l(np.array([]))

    def test_fit_pca_invalid_components(self, sample_dataset):
        with pytest.raises(ValueError):
            fit_pca(sample_dataset, n_components=10)

    def test_cluster_indices_no_matches(self):
        clusters = np.array([1, 1, 1])
        result = cluster_indices(clusters, 0)
        assert len(result) == 0

    @pytest.fixture
    def sample_ds_groups(self):
        time = pd.date_range("2020-01-01", periods=54)
        data = np.array(
            [
                [
                    np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            190.0,
                            82.3,
                            84.0,
                            81.0,
                            92.2,
                            81.9,
                            65.7,
                            86.4,
                            115.0,
                            64.3,
                            53.8,
                            65.0,
                            69.7,
                            95.67,
                            91.2,
                            96.42,
                            68.79,
                            93.95,
                            93.35,
                            125.1,
                            51.88,
                            75.86,
                            114.9,
                            143.7,
                            74.74,
                            121.2,
                            157.4,
                            87.05,
                            112.5,
                            182.7,
                            150.1,
                            137.0,
                            159.3,
                            89.38,
                            71.53,
                            99.27,
                            62.36,
                            68.75,
                            100.3,
                            139.9,
                            112.0,
                            102.5,
                            69.8,
                            68.45,
                            105.4,
                        ]
                    ),
                    np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            52.52,
                            47.39,
                            49.46,
                            33.54,
                            31.47,
                            38.14,
                            28.91,
                            29.95,
                            50.08,
                            54.95,
                            39.66,
                            38.66,
                            22.52,
                            41.65,
                            30.85,
                        ]
                    ),
                    np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            353.0,
                            246.0,
                            166.0,
                            213.0,
                            427.0,
                            100.0,
                            140.0,
                            234.0,
                            171.0,
                            307.2,
                            317.1,
                            266.5,
                            141.6,
                            np.nan,
                            316.0,
                            197.1,
                            87.83,
                            140.1,
                            221.6,
                            157.3,
                            136.1,
                            262.4,
                            247.3,
                            153.2,
                            216.2,
                            332.3,
                            260.2,
                            177.1,
                            244.2,
                            223.8,
                            132.2,
                            163.1,
                            121.5,
                            180.3,
                            181.5,
                            231.6,
                            208.2,
                            204.7,
                            111.8,
                            133.8,
                            186.1,
                        ]
                    ),
                ]
            ]
        )
        ds = xr.Dataset(
            {"Qp": (("group_id", "id", "time"), data)},
            coords={"time": time, "id": ["A", "B", "C"], "group_id": ["G1"]},
        )
        ds["id"].attrs["cf_role"] = "timeseries_id"
        ds["Qp"].attrs["units"] = "m^3 s-1"
        return ds

    @pytest.fixture
    def sample_ds_moments_groups(self):
        lmom = ["l1", "l2", "l3", "tau", "tau3", "tau4"]
        data = data = np.array(
            [
                [
                    np.array(
                        [
                            9.89722137e01,
                            1.84435835e01,
                            4.20631717e00,
                            1.86351126e-01,
                            2.28063986e-01,
                            1.31944789e-01,
                        ]
                    ),
                    np.array(
                        [
                            3.93166656e01,
                            5.80247754e00,
                            2.04072105e-01,
                            1.47583154e-01,
                            3.51698226e-02,
                            8.15386981e-03,
                        ]
                    ),
                    np.array(
                        [
                            2.07748245e02,
                            4.20967562e01,
                            6.46589504e00,
                            2.02633511e-01,
                            1.53596040e-01,
                            1.26061870e-01,
                        ]
                    ),
                ]
            ]
        )
        ds = xr.Dataset(
            {"Qp": (("group_id", "id", "lmom"), data)},
            coords={"lmom": lmom, "id": ["A", "B", "C"], "group_id": ["G1"]},
        )
        ds["id"].attrs["cf_role"] = "timeseries_id"
        return ds

    @pytest.fixture
    def sample_kappa3(self):
        return KappaGen()

    def test_calc_h_z_output_structure(
        self, sample_ds_groups, sample_ds_moments_groups, sample_kappa3
    ):
        result = calc_h_z(sample_ds_groups, sample_ds_moments_groups, sample_kappa3)
        assert isinstance(result, xr.Dataset)
        assert "crit" in result.coords

    def test_calc_h_z_dimensions(
        self, sample_ds_groups, sample_ds_moments_groups, sample_kappa3
    ):
        sample_ds_groups = xr.concat(
            [sample_ds_groups, sample_ds_groups], dim="group_id"
        )
        sample_ds_moments_groups = xr.concat(
            [sample_ds_moments_groups, sample_ds_moments_groups], dim="group_id"
        )
        result = calc_h_z(sample_ds_groups, sample_ds_moments_groups, sample_kappa3)
        assert result.group_id.count().values == 2

    def test_calc_h_z_values(
        self, sample_ds_groups, sample_ds_moments_groups, sample_kappa3
    ):
        result = calc_h_z(
            sample_ds_groups, sample_ds_moments_groups, sample_kappa3, seed=42
        )
        np.testing.assert_almost_equal(0.42279565, result.sel(crit="H").Qp)
        np.testing.assert_almost_equal(0.2568702, result.sel(crit="Z").Qp)

    def test_calc_h_z_values_error(
        self, sample_ds_groups, sample_ds_moments_groups, sample_kappa3
    ):
        sample_ds_moments_groups = -sample_ds_moments_groups
        result = calc_h_z(sample_ds_groups, sample_ds_moments_groups, sample_kappa3)
        assert np.isnan(result.sel(crit="H").Qp)
        assert np.isnan(result.sel(crit="Z").Qp)

    def test_calc_h_z_nan(
        self, sample_ds_groups, sample_ds_moments_groups, sample_kappa3
    ):
        a = np.empty((1, 3, 6))
        a[:] = np.nan
        sample_ds_moments_groups["Qp"] = (["group_id", "id", "lmom"], a)
        result = calc_h_z(sample_ds_groups, sample_ds_moments_groups, sample_kappa3)
        assert np.isnan(result.sel(crit="H").Qp)
        assert np.isnan(result.sel(crit="Z").Qp)

    def test_calculate_rp_from_afr(self, sample_ds_groups, sample_ds_moments_groups):
        result = calculate_rp_from_afr(
            sample_ds_groups, sample_ds_moments_groups, [100, 1000, 10000]
        )
        np.testing.assert_almost_equal(
            197.83515837, result.Qp.sel(return_period=100, id="A")
        )
        np.testing.assert_almost_equal(
            98.87950615, result.Qp.sel(return_period=1000, id="B")
        )

    def test_calculate_rp_from_afr_with_l1(
        self, sample_ds_groups, sample_ds_moments_groups
    ):
        l1 = sample_ds_moments_groups.sel(lmom="l1").dropna(dim="id", how="all") * 1.1
        result = calculate_rp_from_afr(
            sample_ds_groups, sample_ds_moments_groups, [100, 1000, 10000], l1=l1
        )
        np.testing.assert_almost_equal(
            217.618674207, result.Qp.sel(return_period=100, id="A")
        )
        np.testing.assert_almost_equal(
            108.767456765, result.Qp.sel(return_period=1000, id="B")
        )
