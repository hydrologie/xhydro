import numpy as np
import pytest
import xarray as xr

import xhydro as xh


# Smoke test for xscen functions that are imported into xhydro
def test_xscen_imported():
    assert all(
        callable(getattr(xh.cc, f))
        for f in [
            "climatological_op",
            "compute_deltas",
            "ensemble_stats",
            "produce_horizon",
        ]
    )


class TestSampledIndicators:
    @pytest.mark.parametrize("delta_type", ["absolute", "percentage", "foo"])
    def test_sampled_indicators_type(self, delta_type):
        ds = xr.DataArray(
            np.arange(1, 8), coords={"percentile": [1, 10, 20, 50, 80, 90, 99]}
        ).to_dataset(name="QMOYAN")
        deltas = xr.DataArray(
            np.arange(-10, 55, 5),
            coords={"realization": np.arange(13)},
        ).to_dataset(name="QMOYAN")
        if delta_type in ["absolute", "percentage"]:
            out = xh.cc.sampled_indicators(
                ds, deltas, delta_type, n=10, seed=42, return_dist=True
            )

            np.testing.assert_array_equal(
                out[0]["percentile"], [1, 10, 20, 50, 80, 90, 99]
            )
            assert all(
                chosen in np.arange(1, 8) for chosen in np.unique(out[1].QMOYAN.values)
            )
            assert all(
                chosen in np.arange(-10, 55, 5)
                for chosen in np.unique(out[2].QMOYAN.values)
            )

            if delta_type == "absolute":
                assert (
                    np.min(out[3].QMOYAN) >= 1 - 10
                )  # Min of historical minus min of deltas
                assert (
                    np.max(out[3].QMOYAN) <= 7 + 50
                )  # Max of historical plus max of deltas
                np.testing.assert_array_almost_equal(
                    out[0]["QMOYAN"].values, [-3.0, -3.0, 14.6, 40.0, 46.2, 51.6, 56.46]
                )
            else:
                assert np.min(out[3].QMOYAN) >= 1 * (
                    1 - 10 / 100
                )  # Min of historical minus min of deltas
                assert np.max(out[3].QMOYAN) <= 7 * (
                    1 + 50 / 100
                )  # Max of historical plus max of deltas
                np.testing.assert_array_almost_equal(
                    out[0]["QMOYAN"].values, [1.9, 1.9, 4.06, 6.75, 7.34, 8.88, 10.338]
                )

        else:
            with pytest.raises(
                ValueError, match=f"Unknown operation '{delta_type}', expected one"
            ):
                xh.cc.sampled_indicators(ds, deltas, delta_type, n=10, seed=42)

    def test_sampled_indicators_return(self):
        ds = xr.DataArray(
            np.arange(1, 8), coords={"percentile": [1, 10, 20, 50, 80, 90, 99]}
        ).to_dataset(name="QMOYAN")
        deltas = xr.DataArray(
            np.arange(-10, 55, 5),
            coords={"realization": np.arange(13)},
        ).to_dataset(name="QMOYAN")

        out1 = xh.cc.sampled_indicators(
            ds, deltas, "absolute", n=10, seed=42, return_dist=False
        )
        assert isinstance(out1, xr.Dataset)

        out2 = xh.cc.sampled_indicators(
            ds, deltas, "absolute", n=10, seed=42, return_dist=True
        )
        assert isinstance(out2, tuple)
        assert len(out2) == 4
        assert all(isinstance(o, xr.Dataset) for o in out2)
        assert out2[0].equals(out1)
        for o in out2[1:]:
            assert list(o.dims) == ["sample"]
            assert len(o["sample"]) == 10

    @pytest.mark.parametrize("weights", [None, "ds", "deltas", "both"])
    def test_sampled_indicators_weights(self, weights):
        ds = xr.DataArray(
            np.array(
                [
                    [[1, 2, 3, 4, 5], [101, 102, 103, 104, 105]],
                    [[6, 7, 8, 9, 10], [106, 107, 108, 109, 110]],
                ]
            ),
            dims=("foo", "station", "percentile"),
            coords={
                "percentile": [1, 25, 50, 75, 99],
                "station": ["a", "b"],
                "foo": ["bar", "baz"],
            },
        ).to_dataset(name="QMOYAN")
        deltas = xr.DataArray(
            np.array(
                [
                    [[-10, -5, 0, 5, 10], [-25, -20, -15, -10, -5]],
                    [[40, 45, 50, 55, 60], [115, 120, 125, 130, 135]],
                ]
            ),
            dims=("platform", "station", "realization"),
            coords={
                "realization": [1, 25, 50, 75, 99],
                "station": ["a", "b"],
                "platform": ["x", "y"],
            },
        ).to_dataset(name="QMOYAN")
        ds_weights = xr.DataArray(
            np.array([0, 1]),
            dims="foo",
            coords={
                "foo": ["bar", "baz"],
            },
        )
        delta_weights = xr.DataArray(
            np.array([[0, 0, 0, 1, 0], [0, 0, 5, 0, 0]]),
            dims=("platform", "realization"),
            coords={
                "realization": [1, 25, 50, 75, 99],
                "platform": ["x", "y"],
            },
        )

        expected_out1 = (
            np.arange(1, 11)
            if (weights is None or weights == "deltas")
            else np.arange(6, 11)
        )
        expected_out2 = (
            [-10, -5, 0, 5, 10, 40, 45, 50, 55, 60]
            if (weights is None or weights == "ds")
            else [5, 50]
        )

        if weights is None:
            out = xh.cc.sampled_indicators(
                ds, deltas, "percentage", n=10, seed=42, return_dist=True
            )
            np.testing.assert_array_almost_equal(
                out[0].QMOYAN.isel(station=0).values, [1.809, 5.5, 11.8, 12.0, 15.8155]
            )
        elif weights == "ds":
            out = xh.cc.sampled_indicators(
                ds,
                deltas,
                "percentage",
                n=10,
                seed=42,
                return_dist=True,
                ds_weights=ds_weights,
            )
            np.testing.assert_array_almost_equal(
                out[0].QMOYAN.isel(station=0).values,
                [5.427, 8.8, 13.275, 13.5, 15.8155],
            )
        elif weights == "deltas":
            out = xh.cc.sampled_indicators(
                ds,
                deltas,
                "percentage",
                n=10,
                seed=42,
                return_dist=True,
                delta_weights=delta_weights,
            )
            np.testing.assert_array_almost_equal(
                out[0].QMOYAN.isel(station=0).values, [2.1, 7.5, 12.0, 12.0, 14.865]
            )
            assert sum(out[2].QMOYAN.isel(station=0).values == 5) < sum(
                out[2].QMOYAN.isel(station=0).values == 50
            )  # 50 should be sampled more often
        elif weights == "both":
            out = xh.cc.sampled_indicators(
                ds,
                deltas,
                "percentage",
                n=10,
                seed=42,
                return_dist=True,
                ds_weights=ds_weights,
                delta_weights=delta_weights,
            )
            np.testing.assert_array_almost_equal(
                out[0].QMOYAN.isel(station=0).values, [6.3, 12.0, 13.5, 13.5, 14.865]
            )

        assert all(
            chosen in expected_out1
            for chosen in np.unique(out[1].QMOYAN.isel(station=0).values)
        )
        assert all(
            chosen in expected_out2
            for chosen in np.unique(out[2].QMOYAN.isel(station=0).values)
        )
        assert all(
            "station" in o.dims for o in out
        )  # "station" is a shared dimension, so it should be in all outputs

    def test_sampled_indicators_weight_err(self):
        ds = xr.DataArray(
            np.array(
                [
                    [[1, 2, 3, 4, 5], [101, 102, 103, 104, 105]],
                    [[6, 7, 8, 9, 10], [106, 107, 108, 109, 110]],
                ]
            ),
            dims=("platform", "station", "percentile"),
            coords={
                "percentile": [1, 25, 50, 75, 99],
                "station": ["a", "b"],
                "platform": ["x", "y"],
            },
        ).to_dataset(name="QMOYAN")
        deltas = xr.DataArray(
            np.array(
                [
                    [[-10, -5, 0, 5, 10], [-25, -20, -15, -10, -5]],
                    [[40, 45, 50, 55, 60], [115, 120, 125, 130, 135]],
                ]
            ),
            dims=("platform", "station", "realization"),
            coords={
                "realization": [1, 25, 50, 75, 99],
                "station": ["a", "b"],
                "platform": ["x", "y"],
            },
        ).to_dataset(name="QMOYAN")
        delta_weights = xr.DataArray(
            np.array([[0, 0, 0, 1, 0], [0, 0, 5, 0, 0]]),
            dims=("platform", "realization"),
            coords={
                "realization": [1, 25, 50, 75, 99],
                "platform": ["x", "y"],
            },
        )

        with pytest.raises(
            ValueError,
            match="is shared between 'ds' and 'deltas', but not between 'ds_weights' and 'delta_weights'.",
        ):
            xh.cc.sampled_indicators(
                ds, deltas, "percentage", n=10, seed=42, delta_weights=delta_weights
            )

    def test_p_weights(self):
        def _make_da(arr):
            return xr.DataArray(arr, dims="percentile", coords={"percentile": arr})

        out = xh.cc._percentile_weights(_make_da(np.linspace(0, 100, 101)))
        ans = np.ones(101)
        ans[0] = 0.5
        ans[-1] = 0.5
        np.testing.assert_array_equal(out, ans)

        out = xh.cc._percentile_weights(
            _make_da(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]))
        )
        ans = np.array([5.5, 9.5, 10, 10, 10, 10, 10, 10, 10, 9.5, 5.5])
        np.testing.assert_array_equal(out, ans)

    @pytest.mark.parametrize("weights_ndim", [1, 2])
    def test_weighted_sampling(self, weights_ndim):
        data = np.array([[[1, 2, 3], [101, 102, 103]], [[2, 3, 4], [102, 103, 104]]])
        ds = xr.DataArray(
            data,
            dims=("platform", "station", "percentile"),
            coords={
                "percentile": [25, 50, 75],
                "station": ["a", "b"],
                "platform": ["x", "y"],
            },
        ).to_dataset(name="QMOYAN")
        weights = xr.DataArray(
            np.array([[15, 100, 33], [500, 0, 0]]).T,
            dims=("percentile", "platform"),
            coords={
                "platform": ["y", "x"],  # Invert order to test reordering
                "percentile": [25, 50, 75],
            },
        )
        if weights_ndim == 1:
            weights = weights.isel(platform=0)

        out = xh.cc._weighted_sampling(ds, weights, n=10, seed=42)
        assert set(out.dims) == set(
            ["station", "sample"] + (["platform"] if weights_ndim == 1 else [])
        )
        assert list(out["sample"].coords) == ["sample", "percentile"] + (
            ["platform"] if weights_ndim == 2 else []
        )
        assert len(out["sample"]) == 10

        np.testing.assert_array_equal(
            out["percentile"],
            {
                1: [50, 50, 75, 50, 25, 75, 50, 75, 50, 50],
                2: [25, 25, 50, 25, 25, 75, 25, 25, 25, 25],
            }[weights_ndim],
        )  # With seed=42
        if weights_ndim == 2:
            np.testing.assert_array_equal(
                out["platform"], ["y", "x", "y", "x", "x", "y", "x", "y", "x", "x"]
            )

    def test_weighted_sampling_errors(self):
        ds = xr.DataArray(
            np.arange(10),
        ).to_dataset(
            name="foo"
        )  # Doesn't matter what the data is here
        weights = xr.DataArray(
            [[1, 2, 3], [4, np.nan, 6]],
        )
        with pytest.raises(ValueError, match="The weights contain NaNs."):
            xh.cc._weighted_sampling(ds, weights, n=10, seed=42)

    @pytest.mark.parametrize("dim", ["percentile", "quantile"])
    def test_p_or_q(self, dim):
        if dim == "percentile":
            da = xr.DataArray(
                np.linspace(0, 100, 101),
                dims="percentile",
                attrs={"units": "%"},
            )
            da = da.assign_coords(percentile=da)
        else:
            da = xr.DataArray(np.linspace(0, 1, 101), dims="quantile")
            da = da.assign_coords(quantile=da)
        ds = da.to_dataset(name="foo").expand_dims({"time": np.arange(10)})
        out_da, pdim_da, mult_da = xh.cc._perc_or_quantile(da)
        out_ds, pdim_ds, mult_ds = xh.cc._perc_or_quantile(ds)

        assert out_da.equals(da)
        assert out_ds.equals(ds[dim])
        assert pdim_da == pdim_ds == dim
        assert mult_da == mult_ds == (100 if dim == "percentile" else 1)

    def test_p_or_q_errors(self):
        # Test 1: DataArray with >1 dimension
        da = xr.DataArray(
            np.linspace(0, 1, 101),
            dims="quantile",
        ).expand_dims({"time": np.arange(10)})
        with pytest.raises(ValueError, match="DataArray has more than one dimension"):
            xh.cc._perc_or_quantile(da)

        # Test 2: DataArray/Datset with no percentile or quantile dimension
        da = xr.DataArray(
            np.linspace(0, 1, 101),
            dims="foo",
        )
        with pytest.raises(
            ValueError, match="DataArray has no 'percentile' or 'quantile' dimension"
        ):
            xh.cc._perc_or_quantile(da)
        ds = da.to_dataset(name="foo")
        with pytest.raises(ValueError, match="The Dataset should contain one of "):
            xh.cc._perc_or_quantile(ds)

        # Wrong range
        da = xr.DataArray(
            np.linspace(0, 2, 101),
            dims="quantile",
        )
        with pytest.raises(ValueError, match="values do not seem to be in the "):
            xh.cc._perc_or_quantile(da)
        da = xr.DataArray(
            np.linspace(-1, 1, 101),
            dims="quantile",
        )
        with pytest.raises(ValueError, match="values do not seem to be in the "):
            xh.cc._perc_or_quantile(da)
        da = xr.DataArray(
            np.linspace(0, 1, 101),
            dims="percentile",
        )
        with pytest.raises(ValueError, match="values do not seem to be in the "):
            xh.cc._perc_or_quantile(da)
