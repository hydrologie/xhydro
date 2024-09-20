# noqa: N802,N806
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from pint.errors import DimensionalityError
from xclim.testing.helpers import test_timeseries as timeseries

from xhydro import pmp  # noqa: E402


class TestPMP:
    @staticmethod
    def prepare_era5(open_dataset):
        # Prepare a dataset with the required fields
        ds = open_dataset("ERA5/daily_surface_cancities_1990-1993.nc")[
            ["huss", "pr", "snw"]
        ]
        ds = ds.rename({"huss": "hus"})

        # Fake Geopotential field
        np.random.seed(42)
        zg = timeseries(
            np.random.rand(1461) * 1000,
            variable="geopotential",
            start="1990-01-01",
            freq="D",
        ).expand_dims(location=ds.location)
        zg.attrs = {
            "units": "m",
            "long_name": "Geopotential Height",
            "standard_name": "geopotential_height",
        }
        ds["zg"] = zg

        # Expand the dataset to have a 3D field
        ds = xr.concat(
            [
                ds.expand_dims(plev=[1000]),
                (ds * 1.1).expand_dims(plev=[1500]),
                (ds * 1.2).expand_dims(plev=[2000]),
            ],
            dim="plev",
        )
        ds["plev"].attrs = {
            "units": "Pa",
            "long_name": "pressure level",
            "standard_name": "air_pressure",
            "axis": "Z",
            "positive": "down",
        }

        # Fake orography field
        ds["orog"] = xr.DataArray(
            np.array([800, 100, 95, 25, 450]),
            dims=["location"],
            coords={"location": ds.location},
            attrs={
                "units": "m",
                "long_name": "Orography",
                "standard_name": "surface_altitude",
            },
        )

        return ds

    def test_major_precipitation_events(self, open_dataset):
        da = self.prepare_era5(open_dataset).pr.sel(location="Halifax").isel(plev=0)

        result = pmp.major_precipitation_events(da, windows=[1, 2], quantile=0.9)

        assert len(result.time) == len(da.time)
        assert "window" in result.dims
        np.testing.assert_array_almost_equal(result.sel(window=2).max(), 0.00079964)
        np.testing.assert_array_almost_equal(result.sel(window=2).min(), 0.00024292)
        np.testing.assert_array_equal(result.sel(window=2).isnull().sum(), 1313)

    def test_major_precipitation_events_agg(self):
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
        precip["x"].attrs = {"axis": "X"}
        precip["y"].attrs = {"axis": "Y"}

        precip_agg = pmp.spatial_average_storm_configurations(precip, 3).chunk(
            dict(time=-1)
        )
        result = pmp.major_precipitation_events(precip_agg, [1, 2], quantile=0.9)

        assert "window" in result.dims
        np.testing.assert_array_equal(
            result.conf, ["2.1", "2.2", "3.1", "3.2", "3.3", "3.4", "4.1"]
        )
        np.testing.assert_array_almost_equal(
            result.sel(window=2, conf=["2.2", "3.4"]).values,
            np.array(
                [
                    [
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[np.nan, 1.35007655], [np.nan, np.nan]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[0.87522743, np.nan], [np.nan, np.nan]],
                    ],
                    [
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[0.85212627, np.nan], [np.nan, np.nan]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                    ],
                ]
            ),
        )

    @pytest.mark.parametrize("beta_func", [True, False])
    def test_precipitable_water(self, open_dataset, beta_func):
        ds = self.prepare_era5(open_dataset)

        result = pmp.precipitable_water(
            ds.hus,
            ds.zg,
            ds.orog,
            windows=[1, 2],
            beta_func=beta_func,
            add_pre_lay=False,
        )

        if beta_func:
            print(
                result.sel(window=2, location="Halifax")
                .isel(time=slice(400, 410))
                .values
            )
            np.testing.assert_array_almost_equal(
                result.sel(window=2, location="Halifax").isel(time=slice(400, 410)),
                np.array(
                    [
                        364.72795369,
                        584.66359906,
                        584.66359906,
                        726.52627974,
                        726.52627974,
                        726.52627974,
                        683.28554801,
                        683.28554801,
                        211.11846614,
                        211.11846614,
                    ]
                ),
            )
            assert isinstance(result, xr.DataArray)
            assert result.dims == ("window", "location", "time")
            assert result.attrs["units"] == "mm"
        else:
            print(
                result.sel(window=2, location="Halifax")
                .isel(time=slice(400, 410))
                .values
            )
            np.testing.assert_array_almost_equal(
                result.sel(window=2, location="Halifax").isel(time=slice(400, 410)),
                np.array(
                    [
                        411.77127214,
                        584.66359906,
                        584.66359906,
                        726.52627974,
                        726.52627974,
                        726.52627974,
                        683.28554801,
                        683.28554801,
                        211.11846614,
                        211.11846614,
                    ]
                ),
            )

    @pytest.mark.parametrize("add_pre_lay", [True, False])
    def test_precipitable_water_2(self, open_dataset, add_pre_lay):
        ds = self.prepare_era5(open_dataset)

        result = pmp.precipitable_water(
            ds.hus,
            ds.zg,
            ds.orog,
            windows=[1, 2],
            beta_func=True,
            add_pre_lay=add_pre_lay,
        )

        if add_pre_lay:
            print(
                result.sel(window=2, location="Halifax")
                .isel(time=slice(400, 410))
                .values
            )
            np.testing.assert_array_almost_equal(
                result.sel(window=2, location="Halifax").isel(time=slice(400, 410)),
                np.array(
                    [
                        4283.78575011,
                        4411.55270279,
                        4411.55270279,
                        5481.97127172,
                        5481.97127172,
                        5481.97127172,
                        5155.70013505,
                        5155.70013505,
                        1592.98476697,
                        1592.98476697,
                    ]
                ),
            )
            assert isinstance(result, xr.DataArray)
            assert result.dims == ("window", "location", "time")
            assert result.attrs["units"] == "mm"
        else:
            print(
                result.sel(window=2, location="Halifax")
                .isel(time=slice(400, 410))
                .values
            )
            np.testing.assert_array_almost_equal(
                result.sel(window=2, location="Halifax").isel(time=slice(400, 410)),
                np.array(
                    [
                        364.72795369,
                        584.66359906,
                        584.66359906,
                        726.52627974,
                        726.52627974,
                        726.52627974,
                        683.28554801,
                        683.28554801,
                        211.11846614,
                        211.11846614,
                    ]
                ),
            )

    @pytest.mark.parametrize("rebuild_time", [True, False])
    def test_precipitable_water_100y(self, rebuild_time):
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

        result = pmp.precipitable_water_100y(
            da, dist="genextreme", method="ML", rebuild_time=rebuild_time
        )

        if rebuild_time:
            np.testing.assert_array_equal(result.time, da.time)
            np.testing.assert_array_almost_equal(
                [
                    np.unique(result.isel(time=slice(0, 31), y=0, x=0)),
                    np.unique(result.isel(time=slice(31, 59), y=0, x=0)),
                ],
                [np.array([0.99935599]), np.array([0.99393388])],
            )
        else:
            assert "time" not in result.dims
            assert "month" in result.dims
            np.testing.assert_array_almost_equal(
                result.isel(x=0, y=0),
                np.array(
                    [
                        0.99935599,
                        0.99393388,
                        0.96570211,
                        0.9966219,
                        0.99648921,
                        1.18093238,
                        0.97664543,
                        0.99280639,
                        0.99378353,
                        0.99474114,
                        0.99112894,
                        0.99933701,
                    ]
                ),
            )

        assert (result.isel(y=0, x=0).max() / da.isel(y=0, x=0).max()).values <= 1.2

    def test_precipitable_water_100_conf(self):
        np.random.seed(42)
        da_pw = xr.DataArray(
            np.random.rand(2000, 2),
            dims=["time", "conf"],
            coords={
                "time": xr.cftime_range(
                    start="2000", periods=2000, freq="1D", calendar="noleap"
                ),
                "conf": ["1.1", "4.1"],
            },
        )
        da_pw = da_pw.rename("pw")
        result = pmp.precipitable_water_100y(da_pw, dist="genextreme", method="ML")
        np.testing.assert_array_almost_equal(
            [
                np.unique(result[0, 1:30]),
                np.unique(result[0, 151:181]),
                np.unique(result[0, 334:365]),
            ],
            [np.array([1.00063745]), np.array([0.99776472]), np.array([0.98645788])],
        )
        assert isinstance(result, xr.DataArray)

    def test_spatial_average_storm_configurations(self):
        np.random.seed(42)
        da = xr.DataArray(
            np.random.rand(10, 2, 2),
            dims=["time", "some_y", "some_x"],
            coords={
                "time": xr.cftime_range(
                    start="2000", periods=10, freq="1D", calendar="noleap"
                ),
                "some_y": np.linspace(0, 5, 2),
                "some_x": np.linspace(0, 5, 2),
            },
        )
        da.attrs["units"] = "dd"
        da["some_y"].attrs = {"axis": "Y"}
        da["some_x"].attrs = {"axis": "X"}

        with pytest.raises(ValueError):
            pmp.spatial_average_storm_configurations(da, 3)

    def test_spatial_average_storm_configurations2(self):
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
        da["y"].attrs = {"axis": "Y"}
        da["x"].attrs = {"axis": "X"}

        result = pmp.spatial_average_storm_configurations(da, 10)

        np.testing.assert_array_equal(
            result.conf,
            [
                "2.1",
                "2.2",
                "3.1",
                "3.2",
                "3.3",
                "3.4",
                "4.1",
                "5.1",
                "5.2",
                "5.3",
                "5.4",
            ],
        )

        np.testing.assert_array_almost_equal(
            result[10, :, :],
            np.array(
                [
                    [
                        [np.nan, 0.33907024, 0.55259251, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.50715886, 0.26124513, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.37449802, 0.35094036, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.49753116, 0.36187383, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.42009121, 0.62731504, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.3700457, 0.40477336, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.6327582, 0.71380813, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.35464784, 0.33961, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.53146259, 0.52829354, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, 0.40495285, 0.41413964, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                ]
            ),
        )
        assert result.shape[1] == da.shape[0]
        assert isinstance(result, xr.DataArray)

    def test_spatial_average_storm_configurations3(self):
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
        da["y"].attrs = {"axis": "Y"}
        da["x"].attrs = {"axis": "X"}

        result = pmp.spatial_average_storm_configurations(da, 3000)
        np.testing.assert_array_almost_equal(
            result[:, 5, :, :].values,
            np.array(
                [
                    [[0.37567338, np.nan], [0.32925325, np.nan]],
                    [[0.45199877, 0.25292785], [np.nan, np.nan]],
                    [[0.42345313, np.nan], [np.nan, np.nan]],
                    [[0.26600012, np.nan], [np.nan, np.nan]],
                    [[0.37256953, np.nan], [np.nan, np.nan]],
                    [[0.34783047, np.nan], [np.nan, np.nan]],
                    [[0.35246331, np.nan], [np.nan, np.nan]],
                ]
            ),
        )
        assert result.shape[1] == da.shape[0]
        assert isinstance(result, xr.DataArray)

    def test_compute_spring_and_summer_mask(self):
        da = xr.DataArray(
            np.concatenate(
                (
                    np.ones(60) * 1000,
                    np.zeros(200),
                    np.ones(105 + 30) * 1000,
                    np.zeros(335 + 385),
                    np.ones(105) * 1000,
                    np.zeros(240),
                )
            ),
            dims=["time"],
            coords={
                "time": xr.cftime_range(
                    start="2010-01-01", periods=1460, freq="1D", calendar="noleap"
                ),
            },
        )
        da.attrs["units"] = "kg m-2"
        # Year 1: 60 days of snow, 200 days of no snow, 105 days of snow
        # Year 2: 30 days of snow, 335 days of no snow
        # Year 3: 365 days of no snow
        # Year 4: 20 days of no snow, 105 days of snow, 240 days of no snow

        results = pmp.compute_spring_and_summer_mask(
            da, window_wint_start=7, window_wint_end=45, spr_start=30, spr_end=25
        )

        # Checkups for spring
        np.testing.assert_array_equal(
            results.mask_spring.sel(
                time=(results.time.dt.year == 2010)
                & (results.time.dt.dayofyear >= 29)
                & (results.time.dt.dayofyear <= 30)
            ),
            np.array([np.nan, 1]),
        )
        np.testing.assert_array_equal(
            results.mask_spring.sel(
                time=(results.time.dt.year == 2010)
                & (results.time.dt.dayofyear >= 85)
                & (results.time.dt.dayofyear <= 86)
            ),
            np.array([1, np.nan]),
        )
        np.testing.assert_array_equal(
            results.mask_spring.sel(
                time=(results.time.dt.year == 2011) & (results.time.dt.dayofyear == 1)
            ),
            np.array([1]),
        )
        np.testing.assert_array_equal(
            results.mask_spring.sel(
                time=(results.time.dt.year == 2011)
                & (results.time.dt.dayofyear >= 55)
                & (results.time.dt.dayofyear <= 56)
            ),
            np.array([1, np.nan]),
        )
        np.testing.assert_array_equal(
            results.mask_spring.sel(time=(results.time.dt.year == 2012)).isnull().sum(),
            365,
        )
        np.testing.assert_array_equal(
            results.mask_spring.sel(
                time=(results.time.dt.year == 2013)
                & (results.time.dt.dayofyear >= 94)
                & (results.time.dt.dayofyear <= 95)
            ),
            np.array([np.nan, 1]),
        )
        np.testing.assert_array_equal(
            results.mask_spring.sel(
                time=(results.time.dt.year == 2013)
                & (results.time.dt.dayofyear >= 150)
                & (results.time.dt.dayofyear <= 151)
            ),
            np.array([1, np.nan]),
        )

        # Checkups for summer
        np.testing.assert_array_equal(
            results.mask_summer.sel(
                time=(results.time.dt.year == 2010)
                & (results.time.dt.dayofyear >= 60)
                & (results.time.dt.dayofyear <= 61)
            ),
            np.array([np.nan, 1]),
        )
        np.testing.assert_array_equal(
            results.mask_summer.sel(
                time=(results.time.dt.year == 2010)
                & (results.time.dt.dayofyear >= 260)
                & (results.time.dt.dayofyear <= 261)
            ),
            np.array([1, np.nan]),
        )
        np.testing.assert_array_equal(
            results.mask_summer.sel(
                time=(results.time.dt.year == 2011)
                & (results.time.dt.dayofyear >= 30)
                & (results.time.dt.dayofyear <= 31)
            ),
            np.array([np.nan, 1]),
        )
        np.testing.assert_array_equal(
            results.mask_summer.sel(
                time=(results.time.dt.year == 2011) & (results.time.dt.dayofyear == 365)
            ),
            np.array([1]),
        )
        np.testing.assert_array_equal(
            results.mask_summer.sel(time=(results.time.dt.year == 2012)).sum(), 365
        )
        np.testing.assert_array_equal(
            results.mask_summer.sel(
                time=(results.time.dt.year == 2013)
                & (results.time.dt.dayofyear >= 125)
                & (results.time.dt.dayofyear <= 126)
            ),
            np.array([np.nan, 1]),
        )
        np.testing.assert_array_equal(
            results.mask_summer.sel(
                time=(results.time.dt.year == 2013) & (results.time.dt.dayofyear == 365)
            ),
            np.array([1]),
        )

    def test_compute_spring_and_summer_mask2(self):
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
        da.attrs["units"] = "m3"

        with pytest.raises(DimensionalityError):
            pmp.compute_spring_and_summer_mask(da)
