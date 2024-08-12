import pytest  # noqa: E402

lmoments3 = pytest.importorskip("lmoments3")  # noqa: E402
# noqa: N802,N806
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from lmoments3.distr import gev  # noqa: E402

from xhydro import pmp  # noqa: E402


def test_major_precipitation_events():
    np.random.seed(42)
    precip = xr.DataArray(
        np.random.rand(5, 2, 2),
        dims=["time", "y", "x"],
        coords={
            "time": pd.date_range("2000-01-01", periods=5),
            "y": np.linspace(0, 2, 2),
            "x": np.linspace(0, 2, 2),
        },
    )

    result = pmp.major_precipitation_events(precip, windows=[1, 2], quantile=0.9)

    np.testing.assert_array_almost_equal(
        result.values[1],
        np.array(
            [
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, 1.10670883], [0.79007755, np.nan]],
                [[np.nan, np.nan], [np.nan, 1.836086]],
                [[1.43355765, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ]
        ),
    )
    assert "windows" in result.dims


def test_major_precipitation_events_agg():
    np.random.seed(42)
    precip = xr.DataArray(
        np.random.rand(5, 2, 2),
        dims=["time", "y", "x"],
        coords={
            "time": pd.date_range("2000-01-01", periods=5),
            "y": np.linspace(0, 2, 2),
            "x": np.linspace(0, 2, 2),
        },
    )

    precip_agg = pmp.spatial_average_storm_configurations(precip, 1).chunk(
        dict(time=-1)
    )
    result = pmp.major_precipitation_events(precip_agg, [1, 2], quantile=0.9)

    assert "windows" in result.dims
    np.testing.assert_array_equal(result.conf, ["1_0", "1_1", "1_2", "1_3"])
    np.testing.assert_array_almost_equal(
        result.sel(windows=2).values,
        np.array(
            [
                [np.nan, np.nan, np.nan, 1.43355765, np.nan],
                [np.nan, 1.10670883, np.nan, np.nan, np.nan],
                [np.nan, 0.79007755, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 1.836086, np.nan, np.nan],
            ]
        ),
    )


def test_precipitable_water():
    np.random.seed(42)
    ds = xr.Dataset(
        data_vars=dict(
            hus=(["time", "plev", "y", "x"], np.random.rand(5, 2, 2, 2)),
            zg=(["time", "plev", "y", "x"], np.random.rand(5, 2, 2, 2)),
        ),
        coords={
            "time": pd.date_range("2000-01-01", periods=5),
            "plev": np.linspace(0, 1, 2),
            "y": np.linspace(0, 1, 2),
            "x": np.linspace(0, 1, 2),
        },
    )

    fx = xr.Dataset(
        data_vars=dict(
            orog=(["y", "x"], np.random.rand(2, 2)),
        ),
        coords={
            "y": np.linspace(0, 1, 2),
            "x": np.linspace(0, 1, 2),
        },
    )

    result = pmp.precipitable_water(ds, fx, windows=[1, 2])

    np.testing.assert_array_almost_equal(
        result.sel(windows=2, x=1, y=1).values,
        np.array([np.nan, np.nan, 0.08461949, 0.08461949, 0.01033174]),
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("windows", "time", "y", "x")


def test_precipitable_water_100y():
    np.random.seed(42)
    da = xr.DataArray(
        np.random.rand(2000, 2, 2),
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=2000, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 10, 2),
            "x": np.linspace(0, 10, 2),
        },
    )
    da = da.rename("pw")

    result = pmp.precipitable_water_100y(da, dist=gev, method="PWM")
    np.testing.assert_array_almost_equal(
        [
            np.unique(result[1:30, 0, 0]),
            np.unique(result[157:188, 0, 0]),
            np.unique(result[346:378, 0, 0]),
        ],
        [np.array([1.00504545]), np.array([1.07299866]), np.array([1.00385665])],
    )
    assert isinstance(result, xr.DataArray)


def test_precipitable_water_100y2():
    np.random.seed(42)
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
    result = pmp.precipitable_water_100y(da_pw, dist=gev, method="PWM")
    np.testing.assert_array_almost_equal(
        [
            np.unique(result[0, 1:30]),
            np.unique(result[0, 157:188]),
            np.unique(result[0, 346:378]),
        ],
        [np.array([1.00473002]), np.array([0.99725196]), np.array([0.98628203])],
    )
    assert isinstance(result, xr.DataArray)


def test_spatial_average_storm_configurations():
    np.random.seed(42)
    da = xr.DataArray(
        np.random.rand(10, 2, 2),
        dims=["time", "y", "x"],
        coords={
            "time": xr.cftime_range(
                start="2000", periods=10, freq="1D", calendar="noleap"
            ),
            "y": np.linspace(0, 5, 2),
            "x": np.linspace(0, 5, 2),
        },
    )
    da.attrs["units"] = "dd"

    result = pmp.spatial_average_storm_configurations(da, 3)

    np.testing.assert_array_equal(
        result.conf,
        [
            "1_0",
            "1_1",
            "1_2",
            "1_3",
        ],
    )
    np.testing.assert_array_almost_equal(
        result[3, :],
        [
            0.59865848,
            0.86617615,
            0.96990985,
            0.18340451,
            0.29122914,
            0.36636184,
            0.51423444,
            0.17052412,
            0.80839735,
            0.44015249,
        ],
    )
    assert result.shape[1] == da.shape[0]
    assert isinstance(result, xr.DataArray)


def test_spatial_average_storm_configurations2():
    np.random.seed(42)
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

    np.testing.assert_array_almost_equal(
        result[10, :],
        [
            0.66532621,
            0.49524717,
            0.36158708,
            0.35695411,
            0.88701469,
            0.47185446,
            0.87235873,
            0.18527881,
            0.4715606,
            0.71793226,
        ],
    )
    assert result.shape[1] == da.shape[0]
    assert isinstance(result, xr.DataArray)


def test_spatial_average_storm_configurations3():
    np.random.seed(42)
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

    result = pmp.spatial_average_storm_configurations(da, 3000)
    np.testing.assert_array_almost_equal(
        result[10, :],
        [
            0.6413043,
            0.39272977,
            0.75969915,
            0.40939542,
            0.37340927,
            0.37256953,
            0.58516013,
            0.26979637,
            0.60744483,
            0.28081279,
        ],
    )
    assert result.shape[1] == da.shape[0]
    assert isinstance(result, xr.DataArray)


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
