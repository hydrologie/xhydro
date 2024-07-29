# noqa: N802,N806
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from lmoments3.distr import gev

from xhydro import pmp


def test_major_precipitation_events():
    precip = xr.DataArray(
        np.random.rand(10, 5, 5),
        dims=["time", "y", "x"],
        coords={
            "time": pd.date_range("2000-01-01", periods=10),
            "y": np.linspace(0, 1, 5),
            "x": np.linspace(0, 1, 5),
        },
    )

    result = pmp.major_precipitation_events(precip, [1], quantil=0.9, path=None)
    assert result.dims == ("acc_day", "time", "y", "x")


def test_major_precipitation_events2(tmp_path):
    precip = xr.DataArray(
        np.random.rand(10, 5),
        dims=["time", "conf"],
        coords={
            "time": pd.date_range("2000-01-01", periods=10),
            "conf": np.linspace(0, 1, 5),
        },
    )

    f = tmp_path / "1"
    f.mkdir()
    p = f / "Delete1.zarr"

    result = pmp.major_precipitation_events(precip, [1], quantil=0.9, path=p)
    assert "acc_day" in result.dims
    assert len(list(tmp_path.iterdir())) == 1


def test_precipitable_water(tmp_path):
    ds = xr.Dataset(
        data_vars=dict(
            hus=(["time", "plev", "y", "x"], np.random.rand(10, 5, 5, 5)),
            zg=(["time", "plev", "y", "x"], np.random.rand(10, 5, 5, 5)),
        ),
        coords={
            "time": pd.date_range("2000-01-01", periods=10),
            "plev": np.linspace(0, 1, 5),
            "y": np.linspace(0, 1, 5),
            "x": np.linspace(0, 1, 5),
        },
    )

    fx = xr.Dataset(
        data_vars=dict(
            orog=(["y", "x"], np.random.rand(5, 5)),
        ),
        coords={
            "y": np.linspace(0, 1, 5),
            "x": np.linspace(0, 1, 5),
        },
    )

    f = tmp_path / "2"
    f.mkdir()
    p = f / "Delete2.zarr"

    result = pmp.precipitable_water(ds, fx, acc_day=[1, 2], path=p)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("acc_day", "time", "y", "x")
    assert len(list(tmp_path.iterdir())) == 1


def test_rolling_max():
    array = np.random.rand(20, 20, 20)
    result = pmp.rolling_max(array, 3)
    assert result.size == array.size


def test_precipitable_water_100y(tmp_path):
    da = xr.DataArray(
        np.random.rand(2000, 20, 20),
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=2000, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 10, 20),
            "x": np.linspace(0, 10, 20),
        },
    )
    da = da.rename("pw")

    f = tmp_path / "3"
    f.mkdir()
    p = f / "Delete3.zarr"

    result = pmp.precipitable_water_100y(da, dist=gev, path=p)
    assert "quantile" in result.coords
    assert isinstance(result, xr.DataArray)
    assert len(list(tmp_path.iterdir())) == 1


def test_precipitable_water_100y2():
    da_pw = xr.DataArray(
        np.random.rand(2000, 2),
        dims=["time", "conf"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=2000, freq="1D", calendar="noleap"
            ),
            "conf": ["1_0", "4.1_0"],
        },
    )
    da_pw = da_pw.rename("pw")
    result = pmp.precipitable_water_100y(da_pw, dist=gev)
    assert "quantile" in result.coords
    assert isinstance(result, xr.DataArray)


def test_spatial_average_storm_configurations():
    da = xr.DataArray(
        np.random.rand(10, 4, 4),
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=10, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 10, 4),
            "x": np.linspace(0, 10, 4),
        },
    )
    da.attrs["units"] = "dd"

    result = pmp.spatial_average_storm_configurations(da, 2)
    assert result.shape[1] == da.shape[0]
    assert isinstance(result, xr.DataArray)


def test_spatial_average_storm_configurations2():
    da = xr.DataArray(
        np.random.rand(10, 2, 4),
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=10, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 10, 2),
            "x": np.linspace(0, 10, 4),
        },
    )
    da.attrs["units"] = "dd"

    result = pmp.spatial_average_storm_configurations(da, 10)
    assert result.shape[1] == da.shape[0]
    assert isinstance(result, xr.DataArray)


def test_spatial_average_storm_configurations3(tmp_path):
    da = xr.DataArray(
        np.random.rand(10, 2, 2),
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=10, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 100, 2),
            "x": np.linspace(0, 100, 2),
        },
    )

    f = tmp_path / "4"
    f.mkdir()
    p = f / "Delete4.zarr"

    result = pmp.spatial_average_storm_configurations(da, 3000, path=p)
    assert result.shape[1] == da.shape[0]
    assert isinstance(result, xr.DataArray)
    assert len(list(tmp_path.iterdir())) == 1


def test_compute_spring_and_summer_mask():
    da1 = xr.DataArray(
        np.ones((720, 2, 2)) * 1000,
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2010-01-01", periods=720, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 10, 2),
            "x": np.linspace(0, 10, 2),
        },
    )
    da2 = xr.DataArray(
        np.zeros((720, 2, 2)) * np.nan,
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2012-01-01", periods=720, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 10, 2),
            "x": np.linspace(0, 10, 2),
        },
    )
    da = xr.concat([da1, da2], dim="time")

    da.attrs["units"] = "kg m-2"

    results = pmp.compute_spring_and_summer_mask(da)

    assert list(np.unique(results.mask_spring))[0] == 1.0
    assert list(np.unique(results.mask_summer))[0] == 1.0
    assert np.isnan(list(np.unique(results.mask_summer))[1])
    assert np.isnan(list(np.unique(results.mask_spring))[1])
    assert np.unique(np.isnan(results.mask_spring[-720:, :, :].values))[0]
    assert np.unique(np.isnan(results.mask_summer[:720, :, :].values))[0]


def test_compute_spring_and_summer_mask2(capsys):
    da = xr.DataArray(
        np.random.rand(600, 2, 2),
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=600, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 10, 2),
            "x": np.linspace(0, 10, 2),
        },
    )
    da.attrs["units"] = "no"

    with pytest.raises(SystemExit) as exc_info:
        pmp.compute_spring_and_summer_mask(da)

    assert exc_info.value.args[0] == "snow units are not in kg m-2"
