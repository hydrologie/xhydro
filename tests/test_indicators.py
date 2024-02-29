import numpy as np
import pytest
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro as xh


# Smoke test for xscen functions that are imported into xhydro
def test_xscen_imported():
    assert callable(getattr(xh.indicators, "compute_indicators"))


class TestComputeVolume:
    @pytest.mark.parametrize("freq", ["D", "YS"])
    def test_compute_volume(self, freq):
        tile = 365 if freq == "D" else 1
        da = timeseries(
            np.tile(np.arange(1, tile + 1), 3),
            variable="streamflow",
            start="2001-01-01",
            freq=freq,
        )

        out = xh.indicators.compute_volume(da, attrs={"long_name": "Foo"})
        mult = 86400 if freq == "D" else 86400 * 365
        np.testing.assert_array_equal(out, da * mult)
        assert out.attrs["long_name"] == "Foo"
        assert out.attrs["cell_methods"] == "time: sum"
        assert out.attrs["description"] == "Volume of water"
        assert out.attrs["units"] == "m3"

    def test_units(self):
        da = timeseries(
            np.tile(np.arange(1, 366), 3),
            variable="streamflow",
            start="2001-01-01",
            freq="D",
        )

        out_m3 = xh.indicators.compute_volume(da)
        out_hm3 = xh.indicators.compute_volume(da, out_units="hm3")

        assert out_m3.attrs["units"] == "m3"
        assert out_hm3.attrs["units"] == "hm3"

        np.testing.assert_array_equal(out_m3 * 1e-6, out_hm3)


class TestGetYearlyOp:
    ds = timeseries(
        np.arange(1, 365 * 3 + 1),
        variable="streamflow",
        start="2001-01-01",
        freq="D",
        as_dataset=True,
    )

    @pytest.mark.parametrize("op", ["max", "min"])
    def test_get_yearly_op(self, op):
        timeargs = {
            "annual": {},
            "winterdate": {"date_bounds": ["12-01", "02-28"], "freq": "YS-DEC"},
            "winterdoy": {"doy_bounds": [335, 59], "freq": "YS-DEC"},
            "winterdjf": {"season": ["DJF"], "freq": "YS-DEC"},
            "summer": {"doy_bounds": [200, 300]},
        }

        out = xh.indicators.get_yearly_op(self.ds, op=op, timeargs=timeargs)
        assert all(["streamflow" in v for v in out.data_vars])
        assert len(out.data_vars) == len(timeargs)

        if op == "max":
            np.testing.assert_array_equal(
                out.streamflow_max_annual,
                np.add(np.tile(365, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.streamflow_max_summer,
                np.add(np.tile(300, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.streamflow_max_winterdate,
                np.add(
                    np.array([365 + 59, 365 + 59, 365]), np.array([0, 365, 365 * 2])
                ),
            )
            np.testing.assert_array_equal(
                out.streamflow_max_winterdoy, out.streamflow_max_winterdate
            )
            np.testing.assert_array_equal(
                out.streamflow_max_winterdjf, out.streamflow_max_winterdate
            )
        elif op == "min":
            np.testing.assert_array_equal(
                out.streamflow_min_annual,
                np.add(np.tile(1, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.streamflow_min_summer,
                np.add(np.tile(200, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.streamflow_min_winterdate,
                np.add(np.tile(335, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.streamflow_min_winterdoy, out.streamflow_min_winterdate
            )
            np.testing.assert_array_equal(
                out.streamflow_min_winterdjf, out.streamflow_min_winterdate
            )

    def test_missing(self):
        timeargs = {"winterdate": {"date_bounds": ["12-01", "02-28"], "freq": "YS-DEC"}}
        out = xh.indicators.get_yearly_op(
            self.ds,
            op="max",
            timeargs=timeargs,
            missing="pct",
            missing_options={"tolerance": 0.1},
        )

        np.testing.assert_array_equal(
            out.streamflow_max_winterdate,
            np.add(np.array([365 + 59, 365 + 59, np.nan]), np.array([0, 365, 365 * 2])),
        )

    def test_window(self):
        out = xh.indicators.get_yearly_op(self.ds, op="max", window=2)

        assert all(["streamflow2" in v for v in out.data_vars])
        np.testing.assert_array_equal(
            out.streamflow2_max_annual, np.array([364.5, 729.5, 1094.5])
        )

    def test_sum(self):
        ds = timeseries(
            np.arange(1, 365 * 3 + 1),
            variable="streamflow",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )
        ds["volume"] = xh.indicators.compute_volume(ds.streamflow)
        ds["volume"] = ds["volume"].where(
            ~((ds.time.dt.month == 1) & (ds.time.dt.day == 3))
        )

        timeargs = {
            "annual": {},
            "winterdate": {"date_bounds": ["12-01", "02-28"], "freq": "YS-DEC"},
            "summer": {"doy_bounds": [200, 300]},
        }
        out_sum = xh.indicators.get_yearly_op(
            ds, input_var="volume", op="sum", timeargs=timeargs
        )
        out_interp = xh.indicators.get_yearly_op(
            ds, input_var="volume", op="sum", timeargs=timeargs, interpolate_na=True
        )

        ans = {
            "annual": np.array(
                [
                    np.sum(np.arange(1, 365 + 1)) * 86400.0,
                    np.sum(np.arange(1 + 365, 365 + 365 + 1)) * 86400.0,
                    np.sum(np.arange(1 + 730, 365 + 730 + 1)) * 86400.0,
                ]
            ),
            "summer": np.array(
                [
                    np.sum(np.arange(200, 300 + 1)) * 86400.0,
                    np.sum(np.arange(200 + 365, 300 + 365 + 1)) * 86400.0,
                    np.sum(np.arange(200 + 730, 300 + 730 + 1)) * 86400.0,
                ]
            ),
            "winterdate": np.array(
                [
                    np.sum(
                        np.concatenate(
                            (np.arange(335, 365 + 1), np.arange(1 + 365, 59 + 365 + 1))
                        )
                    )
                    * 86400.0,
                    np.sum(
                        np.concatenate(
                            (
                                np.arange(335 + 365, 365 + 365 + 1),
                                np.arange(1 + 730, 59 + 730 + 1),
                            )
                        )
                    )
                    * 86400.0,
                    np.sum(np.arange(335 + 730, 365 + 730 + 1)) * 86400.0,
                ]
            ),
        }

        assert all(["volume" in v for v in out_interp.data_vars])
        np.testing.assert_array_equal(
            out_interp.volume_sum_summer, out_sum.volume_sum_summer
        )
        np.testing.assert_array_equal(out_interp.volume_sum_annual, ans["annual"])
        np.testing.assert_array_equal(out_interp.volume_sum_summer, ans["summer"])
        np.testing.assert_array_equal(
            out_interp.volume_sum_winterdate, ans["winterdate"]
        )

        np.testing.assert_array_equal(
            out_sum.volume_sum_annual,
            (ans["annual"] - np.array([3, 368, 733]) * 86400.0),
        )
        np.testing.assert_array_equal(out_sum.volume_sum_summer, ans["summer"])
        np.testing.assert_array_equal(
            out_sum.volume_sum_winterdate,
            ans["winterdate"] - np.array([368, 733, 0]) * 86400.0,
        )

    def test_errors(self):
        with pytest.raises(ValueError, match="Operation foo is not supported."):
            xh.indicators.get_yearly_op(self.ds, op="foo")
        with pytest.raises(ValueError, match="Cannot use a rolling window"):
            xh.indicators.get_yearly_op(self.ds, op="sum", window=2)
        with pytest.raises(ValueError, match="Frequency D is not supported"):
            xh.indicators.get_yearly_op(
                self.ds, op="max", timeargs={"annual": {"freq": "D"}}
            )
        with pytest.raises(ValueError, match="Only one indexer"):
            xh.indicators.get_yearly_op(
                self.ds,
                op="max",
                timeargs={"annual": {"season": ["DJF"], "doy_bounds": [200, 300]}},
            )
        with pytest.warns(UserWarning, match="The frequency is not YS-DEC"):
            xh.indicators.get_yearly_op(
                self.ds, op="max", timeargs={"annual": {"season": ["DJF"]}}
            )
        with pytest.warns(UserWarning, match="The bounds wrap around the year"):
            xh.indicators.get_yearly_op(
                self.ds,
                op="max",
                timeargs={"annual": {"date_bounds": ["06-15", "06-14"]}},
            )
        with pytest.warns(UserWarning, match="but the bounds"):
            xh.indicators.get_yearly_op(
                self.ds,
                op="max",
                timeargs={
                    "annual": {"date_bounds": ["06-01", "04-30"], "freq": "YS-DEC"}
                },
            )
