import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xhydro.modelling._model_utils import aggregate_output


def make_unit_ds(with_nan: bool = False, chunked: bool = False) -> xr.Dataset:
    """
    Three subbasins in a linear chain: 1→2→3→0 (0 is the out-of-basin sentinel).
    Five units: 0,1 in sbid "1"; 2,3 in sbid "2"; 4 in sbid "3".
    Drainage areas: 10, 20, 5, 15, 8.
    Four daily time steps starting 2000-01-01.
    """
    unit_id = np.arange(5)
    subbasin_id = np.array(["1", "1", "2", "2", "3"])
    dowsub_id = np.array(["2", "2", "3", "3", "0"])
    area = np.array([10.0, 20.0, 5.0, 15.0, 8.0])
    time = pd.date_range("2000-01-01", periods=4, freq="D")

    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
            [3.0, 6.0, 9.0, 12.0, 15.0],
        ],
        dtype=np.float64,
    )
    if with_nan:
        data[1, 2] = np.nan  # unit 2 (in sbid "2") at time index 1

    ds = xr.Dataset(
        {"discharge": xr.DataArray(data, dims=["time", "unit_id"], attrs={"units": "m3 s-1", "long_name": "Discharge"})},
        coords={
            "time": time,
            "unit_id": unit_id,
            "subbasin_id": xr.DataArray(subbasin_id, dims=["unit_id"]),
            "dowsub_id": xr.DataArray(dowsub_id, dims=["unit_id"]),
            "unit_drainage_area": xr.DataArray(area, dims=["unit_id"]),
        },
    )
    if chunked:
        ds = ds.chunk({"time": 2, "unit_id": 5})
    return ds


def make_confluence_ds() -> xr.Dataset:
    """
    Two independent head subbasins both draining into a single outlet.
    Subbasins "1"→"3", "2"→"3", "3"→"0".
    Units: 0,1 in "1"; 2,3 in "2"; 4 in "3".
    """
    unit_id = np.arange(5)
    subbasin_id = np.array(["1", "1", "2", "2", "3"])
    dowsub_id = np.array(["3", "3", "3", "3", "0"])
    area = np.array([10.0, 20.0, 5.0, 15.0, 8.0])
    time = pd.date_range("2000-01-01", periods=3, freq="D")

    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
        ],
        dtype=np.float64,
    )
    return xr.Dataset(
        {"discharge": xr.DataArray(data, dims=["time", "unit_id"])},
        coords={
            "time": time,
            "unit_id": unit_id,
            "subbasin_id": xr.DataArray(subbasin_id, dims=["unit_id"]),
            "dowsub_id": xr.DataArray(dowsub_id, dims=["unit_id"]),
            "unit_drainage_area": xr.DataArray(area, dims=["unit_id"]),
        },
    )


def make_subbasin_ds() -> xr.Dataset:
    """
    Subbasin-level input (spatial dim = subbasin_id), linear chain "1"→"2"→"3"→"0".
    Drainage areas 30, 20, 8. Three daily time steps.
    """
    subbasin_id = np.array(["1", "2", "3"])
    dowsub_id = np.array(["2", "3", "0"])
    area = np.array([30.0, 20.0, 8.0])
    time = pd.date_range("2000-01-01", periods=3, freq="D")
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    return xr.Dataset(
        {"apport_lateral": xr.DataArray(data, dims=["time", "subbasin_id"])},
        coords={
            "time": time,
            "subbasin_id": xr.DataArray(subbasin_id, dims=["subbasin_id"]),
            "dowsub_id": xr.DataArray(dowsub_id, dims=["subbasin_id"]),
            "subbasin_drainage_area": xr.DataArray(area, dims=["subbasin_id"]),
        },
    )


# --- Error cases ---


def test_invalid_aggregation_same_level():
    ds = make_unit_ds()
    with pytest.raises(ValueError, match="Invalid aggregation levels"):
        aggregate_output(ds, by="subbasin", to="subbasin")


def test_missing_subbasin_id_coord():
    ds = make_unit_ds().drop_vars("subbasin_id")
    with pytest.raises(ValueError, match="standardize_outputs"):
        aggregate_output(ds, by="unit", to="subbasin")


# --- to="subbasin" ---


def test_subbasin_basic():
    ds, weights = aggregate_output(make_unit_ds(), by="unit", to="subbasin")
    t0 = ds["discharge"].isel(time=0)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="1")), 50 / 30, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="2")), 75 / 20, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="3")), 5.0, rtol=1e-9)
    assert ds.sizes["subbasin_id"] == 3
    assert ds.sizes["time"] == 4


def test_subbasin_nan_in_one_unit():
    # unit 2 (in sbid "2") is NaN at time=1; sbid "2" at time=1 uses only unit 3.
    # time=1 data: [2,4,NaN,8,10]. sbid "2": (8*15)/(15) = 8.0.
    ds, _ = aggregate_output(make_unit_ds(with_nan=True), by="unit", to="subbasin")
    np.testing.assert_allclose(float(ds["discharge"].sel(subbasin_id="2").isel(time=1)), 8.0, rtol=1e-9)
    np.testing.assert_allclose(
        float(ds["discharge"].sel(subbasin_id="1").isel(time=1)),
        (2 * 10 + 4 * 20) / 30,
        rtol=1e-9,
    )


def test_subbasin_all_nan_subbasin_gives_nan():
    raw = make_unit_ds()
    raw["discharge"].values[:, 4] = np.nan  # unit 4 is the only unit in sbid "3"
    ds, _ = aggregate_output(raw, by="unit", to="subbasin")
    assert np.all(np.isnan(ds["discharge"].sel(subbasin_id="3").values))


def test_subbasin_weight_reuse():
    raw = make_unit_ds()
    ds1, weights = aggregate_output(raw, by="unit", to="subbasin")
    ds2, _ = aggregate_output(raw, by="unit", to="subbasin", weights=weights)
    xr.testing.assert_allclose(ds1["discharge"], ds2["discharge"])


def test_subbasin_dask_matches_eager():
    ds_eager, _ = aggregate_output(make_unit_ds(), by="unit", to="subbasin")
    ds_lazy, _ = aggregate_output(make_unit_ds(chunked=True), by="unit", to="subbasin")
    xr.testing.assert_allclose(ds_eager["discharge"], ds_lazy["discharge"])


# --- by= aliases ---


@pytest.mark.parametrize("by", ["hru", "rhhu", "unit"])
def test_by_aliases_subbasin(by):
    ds, _ = aggregate_output(make_unit_ds(), by=by, to="subbasin")
    assert ds.sizes["subbasin_id"] == 3


# --- to="drainage_area" ---


def test_drainage_area_linear_chain():
    ds, _ = aggregate_output(make_unit_ds(), by="unit", to="drainage_area")
    t0 = ds["discharge"].isel(time=0)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="1")), 50 / 30, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="2")), 125 / 50, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="3")), 165 / 58, rtol=1e-9)


def test_drainage_area_confluence():
    # "1"→"3", "2"→"3". sbid "3" accumulates all 5 units.
    # time=0: data=[1,2,3,4,5], areas=[10,20,5,15,8], total_area=58
    ds, _ = aggregate_output(make_confluence_ds(), by="unit", to="drainage_area")
    t0 = ds["discharge"].isel(time=0)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="1")), 50 / 30, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="2")), 75 / 20, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="3")), 165 / 58, rtol=1e-9)


def test_drainage_area_nan_excludes_unit_from_denominator():
    # unit 2 (in sbid "2") is NaN at time=1.
    # sbid "2" at time=1 uses units {0,1,3}: (2*10+4*20+8*15)/(10+20+15) = 220/45
    # sbid "3" at time=1 uses units {0,1,3,4}: (2*10+4*20+8*15+10*8)/(10+20+15+8) = 300/53
    ds, _ = aggregate_output(make_unit_ds(with_nan=True), by="unit", to="drainage_area")
    t1 = ds["discharge"].isel(time=1)
    np.testing.assert_allclose(float(t1.sel(subbasin_id="2")), 220 / 45, rtol=1e-9)
    np.testing.assert_allclose(float(t1.sel(subbasin_id="3")), 300 / 53, rtol=1e-9)


def test_drainage_area_weight_reuse():
    raw = make_unit_ds()
    ds1, weights = aggregate_output(raw, by="unit", to="drainage_area")
    ds2, _ = aggregate_output(raw, by="unit", to="drainage_area", weights=weights)
    xr.testing.assert_allclose(ds1["discharge"], ds2["discharge"])


def test_drainage_area_dask_matches_eager():
    ds_eager, _ = aggregate_output(make_unit_ds(), by="unit", to="drainage_area")
    ds_lazy, _ = aggregate_output(make_unit_ds(chunked=True), by="unit", to="drainage_area")
    xr.testing.assert_allclose(ds_eager["discharge"], ds_lazy["discharge"])


# --- Output metadata ---


def test_output_aggregation_level_attr_subbasin():
    ds, _ = aggregate_output(make_unit_ds(), by="unit", to="subbasin")
    for v in ds.data_vars:
        assert ds[v].attrs["aggregation_level"] == "Subbasin"
        assert "Aggregated" in ds[v].attrs.get("history", "")


def test_output_aggregation_level_attr_drainage_area():
    ds, _ = aggregate_output(make_unit_ds(), by="unit", to="drainage_area")
    for v in ds.data_vars:
        assert ds[v].attrs["aggregation_level"] == "DrainageArea"


def test_output_sorted_and_transposed():
    ds, _ = aggregate_output(make_unit_ds(), by="unit", to="subbasin")
    ids = [int(x) for x in ds["subbasin_id"].values]
    assert ids == sorted(ids)
    assert ds["discharge"].dims[0] == "time"
    assert ds["discharge"].dims[1] == "subbasin_id"


# --- Weights shape and dtype ---


def test_weights_shape_subbasin():
    _, weights = aggregate_output(make_unit_ds(), by="unit", to="subbasin")
    assert weights.dims == ("sbid", "unit_id")
    assert weights.shape == (3, 5)
    assert weights.dtype == np.float64


def test_weights_shape_drainage_area():
    _, weights = aggregate_output(make_unit_ds(), by="unit", to="drainage_area")
    assert weights.dims == ("sbid", "unit_id")
    assert weights.shape == (3, 5)
    assert float(weights.sel(sbid="3").min()) >= 0


# --- Global attrs preservation ---


def test_global_attrs_preserved():
    ds = make_unit_ds()
    ds.attrs["HYDROTEL_version"] = "4.1"
    ds.attrs["title"] = "test run"
    out, _ = aggregate_output(ds, by="unit", to="subbasin")
    assert out.attrs.get("HYDROTEL_version") == "4.1"
    assert out.attrs.get("title") == "test run"
    assert out["discharge"].attrs.get("units") == ds["discharge"].attrs.get("units")


# --- by="subbasin" aggregation ---


def test_subbasin_input_to_drainage_area():
    # time=0 data [1,2,3], areas [30,20,8], chain 1→2→3:
    #   "1": 1*30/30 = 1.0
    #   "2": (1*30 + 2*20)/50 = 70/50 = 1.4
    #   "3": (1*30 + 2*20 + 3*8)/58 = 94/58
    ds, weights = aggregate_output(make_subbasin_ds(), by="subbasin", to="drainage_area")
    t0 = ds["apport_lateral"].isel(time=0)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="1")), 1.0, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="2")), 70 / 50, rtol=1e-9)
    np.testing.assert_allclose(float(t0.sel(subbasin_id="3")), 94 / 58, rtol=1e-9)
    assert weights.dims == ("sbid", "subbasin_id")
    assert weights.shape == (3, 3)
