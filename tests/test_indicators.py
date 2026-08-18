from typing import cast

import numpy as np
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro as xh


# Smoke test for xscen functions that are imported into xhydro
def test_xscen_imported():
    assert callable(xh.indicators.compute_indicators)


class TestComputeVolume:
    @pytest.mark.parametrize("freq", ["D", "YS"])
    def test_compute_volume(self, freq):
        tile = 365 if freq == "D" else 1
        da = timeseries(
            np.tile(np.arange(1, tile + 1), 3),
            variable="q",
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
            variable="q",
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
        variable="q",
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
        assert all(["q" in v for v in out.data_vars])
        assert len(out.data_vars) == len(timeargs)

        if op == "max":
            np.testing.assert_array_equal(
                out.q_max_annual,
                np.add(np.tile(365, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.q_max_summer,
                np.add(np.tile(300, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.q_max_winterdate,
                np.add(np.array([365 + 59, 365 + 59, 365]), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(out.q_max_winterdoy, out.q_max_winterdate)
            np.testing.assert_array_equal(out.q_max_winterdjf, out.q_max_winterdate)
        elif op == "min":
            np.testing.assert_array_equal(
                out.q_min_annual,
                np.add(np.tile(1, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.q_min_summer,
                np.add(np.tile(200, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(
                out.q_min_winterdate,
                np.add(np.tile(335, 3), np.array([0, 365, 365 * 2])),
            )
            np.testing.assert_array_equal(out.q_min_winterdoy, out.q_min_winterdate)
            np.testing.assert_array_equal(out.q_min_winterdjf, out.q_min_winterdate)

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
            out.q_max_winterdate,
            np.add(np.array([365 + 59, 365 + 59, np.nan]), np.array([0, 365, 365 * 2])),
        )

    def test_window(self):
        out = xh.indicators.get_yearly_op(self.ds, op="max", window=2)

        assert all(["q2" in v for v in out.data_vars])
        np.testing.assert_array_equal(out.q2_max_annual, np.array([364.5, 729.5, 1094.5]))

    def test_sum(self):
        ds = timeseries(
            np.arange(1, 365 * 3 + 1),
            variable="q",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )
        ds["volume"] = xh.indicators.compute_volume(ds.q)
        ds["volume"] = ds["volume"].where(~((ds.time.dt.month == 1) & (ds.time.dt.day == 3)))

        timeargs = {
            "annual": {},
            "winterdate": {"date_bounds": ["12-01", "02-28"], "freq": "YS-DEC"},
            "summer": {"doy_bounds": [200, 300]},
        }
        out_sum = xh.indicators.get_yearly_op(ds, input_var="volume", op="sum", timeargs=timeargs)
        out_interp = xh.indicators.get_yearly_op(ds, input_var="volume", op="sum", timeargs=timeargs, interpolate_na=True)

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
                    np.sum(np.concatenate((np.arange(335, 365 + 1), np.arange(1 + 365, 59 + 365 + 1)))) * 86400.0,
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
        np.testing.assert_array_equal(out_interp.volume_sum_summer, out_sum.volume_sum_summer)
        np.testing.assert_array_equal(out_interp.volume_sum_annual, ans["annual"])
        np.testing.assert_array_equal(out_interp.volume_sum_summer, ans["summer"])
        np.testing.assert_array_equal(out_interp.volume_sum_winterdate, ans["winterdate"])

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
            xh.indicators.get_yearly_op(self.ds, op="max", timeargs={"annual": {"freq": "D"}})
        with pytest.raises(ValueError, match="Only one indexer"):
            xh.indicators.get_yearly_op(
                self.ds,
                op="max",
                timeargs={"annual": {"season": ["DJF"], "doy_bounds": [200, 300]}},
            )
        with pytest.warns(UserWarning, match="The frequency is not YS-DEC"):
            xh.indicators.get_yearly_op(self.ds, op="max", timeargs={"annual": {"season": ["DJF"]}})
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
                timeargs={"annual": {"date_bounds": ["06-01", "04-30"], "freq": "YS-DEC"}},
            )


class TestSplitStreamflow:
    k = 0.925

    # Sharp rise, slow recession: asymmetric, so a filter that mixes up the forward and backward
    # passes cannot reproduce these numbers. The expected baseflows come from the Lyne-Hollick
    # recursion qf[n] = k qf[n-1] + (1+k)/2 (x[n] - x[n-1]), b[n] = clip(x[n] - qf[n], 0, x[n]),
    # with k=0.925 and qf[0]=0, each pass reversing the direction of the previous one.
    FLOW = [2.0, 2.0, 12.0, 9.0, 7.0, 5.5, 4.5, 3.8, 3.2, 3.0]
    BASEFLOW = {
        1: [2.0, 2.0, 2.375, 2.984375, 3.360547, 3.577256, 3.683962, 3.718915, 3.2, 3.0],
        2: [2.0, 2.0, 2.375, 2.984375, 3.156692, 3.131378, 3.0909, 3.041397, 3.0075, 3.0],
        3: [2.0, 2.0, 2.014062, 2.063984, 2.139476, 2.214818, 2.282042, 2.34085, 2.39212, 2.437992],
    }

    @pytest.fixture
    def da(self):
        return timeseries(np.array(self.FLOW), variable="q", start="2001-01-01", freq="D")

    def test_values(self, da):
        # n_passes=2 has to be in the table: the backward pass is where the sign and
        # time-orientation bugs hid, and an odd number of passes never exercises it.
        previous = None
        for n_passes, expected in self.BASEFLOW.items():
            baseflow, runoff = xh.indicators.split_streamflow(da, k=self.k, n_passes=n_passes)

            np.testing.assert_allclose(baseflow, expected, rtol=1e-6)
            np.testing.assert_allclose(baseflow + runoff, da, rtol=1e-12)
            # The filter state is initialised so that quickflow[0] == 0, which leaves the first
            # timestep as pure baseflow. Getting the sign wrong zeroes it out instead.
            np.testing.assert_allclose(runoff[0], 0.0, atol=1e-12)
            assert (baseflow >= 0).all()
            assert (baseflow <= da).all()
            assert (runoff >= 0).all()
            if previous is not None:  # each pass can only remove flow
                assert (baseflow <= previous).all()
            previous = baseflow

    def test_steady_flow(self):
        da = cast("xr.DataArray", timeseries(np.full(10, 7.0), variable="q", start="2001-01-01", freq="D"))

        baseflow, runoff = xh.indicators.split_streamflow(da, k=self.k)

        np.testing.assert_allclose(baseflow, 7.0)
        np.testing.assert_allclose(runoff, 0.0, atol=1e-12)

    def test_preserves_structure(self, da):
        # Time first, so the internal move of "time" to the last axis has to be undone. float32
        # because lfilter silently promotes to float64 unless its coefficients are cast, which
        # would contradict the output_dtypes declared to dask.
        flows = np.stack([self.FLOW, self.FLOW[::-1], np.full(len(self.FLOW), 3.0)])
        da = da.expand_dims(station=["a", "b", "c"]).copy(data=flows).astype(np.float32).transpose("time", "station")

        baseflow, runoff = xh.indicators.split_streamflow(da, k=self.k)

        assert baseflow.dims == da.dims
        assert runoff.dims == da.dims
        assert (baseflow.time == da.time).all()  # a reversed pass must not reverse the coord
        assert baseflow.attrs["units"] == da.attrs["units"]
        assert baseflow.dtype == np.float32
        for station in da.station.values:  # stations are filtered independently of each other
            alone, _ = xh.indicators.split_streamflow(da.sel(station=station), k=self.k)
            np.testing.assert_allclose(baseflow.sel(station=station), alone, rtol=1e-6)

    # The naming branches on units, so both branches need a case. "m^3 s-1", "m3/s" and
    # "ft3 s-1" are discharges written differently, and the first two appear in this repo's own
    # tests: comparing the unit string instead of its dimensionality names a discharge "mrrob".
    # "kg m-2 s-1" and "kg m-2" only reach the depth branch through the hydro context.
    # A discharge gets no standard name at all: "q_base" and "q_runoff" are not CF names.
    _DISCHARGE = (("q_base", "q_runoff"), ("Baseflow", "Direct runoff"), None)
    _DEPTH = (
        ("mrrob", "mrros"),
        ("Subsurface runoff", "Surface runoff"),
        ("subsurface_runoff_flux", "surface_runoff_flux"),
    )

    @pytest.mark.parametrize(
        "units,expected",
        [
            ("m3/s", _DISCHARGE),
            ("ft3 s-1", _DISCHARGE),
            ("mm h-1", _DEPTH),
            ("kg m-2 s-1", _DEPTH),
            ("kg m-2", _DEPTH),
        ],
    )
    def test_attrs(self, da, units, expected):
        names, long_names, standard_names = expected
        # A bare input would let a broken implementation pass, so give it the full set that a
        # standardized `q` carries: the point is that standard_name is replaced or dropped, never copied.
        da = da.rename("q").assign_attrs(
            units=units,
            long_name="Simulated streamflow",
            standard_name="outgoing_water_volume_transport_along_river_channel",
        )

        baseflow, runoff = xh.indicators.split_streamflow(da, k=0.9, n_passes=5)

        assert (baseflow.name, runoff.name) == names
        assert (baseflow.attrs["long_name"], runoff.attrs["long_name"]) == long_names
        # Each output is a fraction of the streamflow, so keeping the input's own standard
        # name would assert that it still is the whole of it.
        if standard_names is None:
            assert "standard_name" not in baseflow.attrs
            assert "standard_name" not in runoff.attrs
        else:
            assert (baseflow.attrs["standard_name"], runoff.attrs["standard_name"]) == standard_names
        for out in (baseflow, runoff):
            assert out.attrs["units"] == units
            assert "k=0.9" in out.attrs["description"]
            assert "n_passes=5" in out.attrs["description"]

    def test_dask(self, da):
        flows = np.stack([self.FLOW, self.FLOW[::-1]])
        da = da.expand_dims(station=["a", "b"]).copy(data=flows)

        eager, _ = xh.indicators.split_streamflow(da, k=self.k)
        lazy, _ = xh.indicators.split_streamflow(da.chunk({"station": 1, "time": -1}), k=self.k)

        np.testing.assert_allclose(lazy.compute(), eager, rtol=1e-12)

    def test_errors(self, da):
        with pytest.raises(ValueError, match="`k` must be in"):
            xh.indicators.split_streamflow(da, k=0.0)
        with pytest.raises(ValueError, match="`k` must be in"):
            xh.indicators.split_streamflow(da, k=1.0)
        with pytest.raises(ValueError, match="`n_passes` must be >= 1"):
            xh.indicators.split_streamflow(da, n_passes=0)

    # The filter itself is unit-agnostic, so unusable units are a warning, not a refusal.
    # "" is dimensionless rather than missing, so it reaches the check like any other unit.
    @pytest.mark.parametrize("units", ["degC", ""])
    def test_unknown_units(self, da, units):
        with pytest.warns(UserWarning, match="should be a discharge"):
            baseflow, runoff = xh.indicators.split_streamflow(da.assign_attrs(units=units), k=self.k)

        # Not a discharge, so the naming falls through to the depth branch.
        assert (baseflow.name, runoff.name) == ("mrrob", "mrros")
        np.testing.assert_allclose(baseflow + runoff, da, rtol=1e-12)


def _flood_inputs(*, correlated, end="2006-12-31"):
    """
    Build a rainless baseline: streamflow 1 mm with one spike per reference year.

    With `correlated`, spike height and that year's soil water content rise together,
    which makes the threshold fit succeed; without it, SWI is constant and the
    Spearman gate returns an infinite threshold.
    """
    time = xr.date_range("1995-01-01", end, freq="D")
    starts = np.flatnonzero(time.dayofyear == 1)
    bounds = np.append(starts, time.size)
    rivo = np.ones(time.size)
    mrsol = np.full(time.size, 50.0)
    for i in range(10):
        rivo[starts[i] + 150] = 3.0 + i
        if correlated:
            mrsol[bounds[i] : bounds[i + 1]] = 100.0 * (0.15 + 0.07 * i)

    def da(values):
        return xr.DataArray(values, coords={"time": time}, dims="time", attrs={"units": "mm"})

    inputs = {
        "mrsosat": xr.DataArray(100.0, attrs={"units": "mm"}),
        "mrsol": da(mrsol),
        "prra": da(np.zeros(time.size)),
        "rivo": da(rivo),
        "snm": da(np.zeros(time.size)),
    }
    return inputs, starts


def _plant_flood_event(inputs, starts, *, rain, snow=None, peak=20.0):
    """Plant the 2006 annual maximum at day-of-year 201, with `rain` (and `snow`) ending on the peak day."""
    p = starts[11] + 200
    rain = np.asarray(rain, dtype=float)
    inputs["rivo"].values[p] = peak
    inputs["prra"].values[p - rain.size + 1 : p + 1] = rain
    if snow is not None:
        snow = np.asarray(snow, dtype=float)
        inputs["snm"].values[p - snow.size + 1 : p + 1] = snow
    return p


# the December-to-November period holding the engineered 2006 event, labelled by its first day
_EVENT_PERIOD = "2005-12-01"


class TestMajorFloodEvents:
    # 12 years of daily data; the last reference year leaves 2005 as a buffer
    # and 2006 as the year whose event each test engineers.

    def test_indicators(self):
        inputs, starts = _flood_inputs(correlated=False)
        # rain [0, 5, 5, 2]: the walk from the peak stops at the dry day, so the window is 3 days
        _plant_flood_event(inputs, starts, rain=[0, 5, 5, 2], snow=[0, 1, 1, 1])
        inputs["mrros"] = (0.25 * inputs["rivo"]).assign_attrs(units="mm")

        out = xh.indicators.flood_types.major_flood_events(**inputs).sel(time=_EVENT_PERIOD)
        assert out.rivo_peak.item() == 20.0
        assert out.rivo_peak_doy.item() == 201
        assert out.event_duration.item() == 3
        assert out.prra_sum.item() == 12.0
        assert out.prra_max.item() == 5.0
        assert out.snm_sum.item() == 3.0
        assert out.swi_antecedent.item() == 0.5  # mrsol 50 / mrsosat 100 on the day before the window
        np.testing.assert_allclose(out.direct_streamflow_fraction.item(), 0.25)

    def test_max_days_caps_the_window(self):
        inputs, starts = _flood_inputs(correlated=False)
        _plant_flood_event(inputs, starts, rain=[5] * 10)

        out = xh.indicators.flood_types.major_flood_events(**inputs).sel(time=_EVENT_PERIOD)
        assert out.event_duration.item() == 7

    def test_mrros_default(self):
        inputs, starts = _flood_inputs(correlated=False)
        _plant_flood_event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])

        from_none = xh.indicators.flood_types.major_flood_events(**inputs)
        from_explicit = xh.indicators.flood_types.major_flood_events(**inputs, mrros=xh.indicators.split_streamflow(inputs["rivo"])[1])

        xr.testing.assert_identical(from_none, from_explicit)

    @pytest.mark.parametrize("units,scale", [("1", 100.0), ("%", 1.0)])
    def test_normalized_mrsol(self, units, scale):
        """Without `mrsosat`, `mrsol` is taken as the wetness index, and percent is rescaled to 0-1."""
        inputs, starts = _flood_inputs(correlated=False)
        _plant_flood_event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])
        normalized = {**inputs, "mrsol": (inputs["mrsol"] / scale).assign_attrs(units=units)}
        del normalized["mrsosat"]

        out = xh.indicators.flood_types.major_flood_events(**normalized)
        reference = xh.indicators.flood_types.major_flood_events(**inputs)

        np.testing.assert_allclose(out.swi_antecedent.values, reference.swi_antecedent.values)

    def test_structure(self):
        inputs, starts = _flood_inputs(correlated=False)
        _plant_flood_event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])
        for name in ("mrsol", "prra", "rivo", "snm"):
            inputs[name] = inputs[name].expand_dims(station=["a", "b"]).transpose("time", "station")

        out = xh.indicators.flood_types.major_flood_events(**inputs)
        assert out.rivo_peak.dims == ("time", "station")
        # December 2006 spills into a 13th period, truncated like the December-less 1995
        assert out.time.size == 13
        # resample labels each period by its first day, which is a 1 December
        assert out.time.dt.month.values.tolist() == [12] * 13
        assert "swi_threshold" not in out
        np.testing.assert_array_equal(out.rivo_peak.sel(station="a").values, out.rivo_peak.sel(station="b").values)

    @pytest.mark.parametrize(
        "freq,n_periods,months,event_period",
        [
            # calendar years split the record into 12 periods instead of 13 December-anchored ones
            ("YS", 12, [1], "2006-01-01"),
            # seasons: the 20 July 2006 event lands in the June-August quarter
            ("QS-DEC", 49, [12, 3, 6, 9], "2006-06-01"),
        ],
    )
    def test_freq(self, freq, n_periods, months, event_period):
        """`freq` takes any resampling frequency, not just a yearly one."""
        inputs, starts = _flood_inputs(correlated=False)
        _plant_flood_event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])

        out = xh.indicators.flood_types.major_flood_events(**inputs, freq=freq)

        assert out.time.size == n_periods
        assert sorted(set(out.time.dt.month.values.tolist())) == sorted(months)
        assert out.rivo_peak.sel(time=event_period).item() == 20.0

    def test_dask(self):
        inputs, starts = _flood_inputs(correlated=False)
        _plant_flood_event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])
        eager = xh.indicators.flood_types.major_flood_events(**inputs)

        for name in ("mrsol", "prra", "rivo", "snm"):
            inputs[name] = inputs[name].chunk({"time": -1})
        lazy = xh.indicators.flood_types.major_flood_events(**inputs)

        xr.testing.assert_identical(eager, lazy.compute())

    def test_truncated_period_is_kept(self):
        """A period cut short by the end of the record still reports its peak, from a partial window."""
        inputs, _ = _flood_inputs(correlated=False, end="2006-06-30")

        out = xh.indicators.flood_types.major_flood_events(**inputs)

        assert not out.rivo_peak.isnull().any()
        assert out.rivo_peak.isel(time=-1).item() == 1.0  # the rainless baseline, no spike planted in 2006

    def test_missing_period(self):
        """Only an entirely missing streamflow period gives NaN, which `classify_flood_events` turns into -1."""
        inputs, _ = _flood_inputs(correlated=False)
        time = inputs["rivo"].time
        gap = (time >= np.datetime64("1999-12-01")) & (time <= np.datetime64("2000-11-30"))
        inputs["rivo"] = inputs["rivo"].where(~gap)

        out = xh.indicators.flood_types.major_flood_events(**inputs)

        assert out.rivo_peak.isnull().sum().item() == 1
        assert np.isnan(out.rivo_peak.sel(time="1999-12-01").item())
        flood_type = xh.indicators.flood_types.classify_flood_events(out, threshold=np.inf)
        assert flood_type.sel(time="1999-12-01").item() == -1

    def test_errors(self):
        inputs, _ = _flood_inputs(correlated=False)
        with pytest.raises(ValueError, match="missing"):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "prra": inputs["prra"].drop_attrs()})
        with pytest.raises(ValueError, match='convertible to "mm"'):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "prra": inputs["prra"].assign_attrs(units="K")})
        with pytest.raises(ValueError, match="time coordinate"):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "snm": inputs["snm"].isel(time=slice(0, 100))})
        with pytest.raises(ValueError, match="daily"):
            weekly = {k: v.isel(time=slice(None, None, 7)) if "time" in v.dims else v for k, v in inputs.items()}
            xh.indicators.flood_types.major_flood_events(**weekly)
        with pytest.raises(ValueError, match="`max_days` must be >= 1"):
            xh.indicators.flood_types.major_flood_events(**inputs, max_days=0)
        with pytest.raises(ValueError, match="dimensionless"):
            # a water depth needs `mrsosat` to be normalized
            xh.indicators.flood_types.major_flood_events(**{k: v for k, v in inputs.items() if k != "mrsosat"})
        with pytest.raises(ValueError, match="must be a 0-1 index"):
            # dimensionless units alone do not make `mrsol` an index: the values must be in range
            out_of_range = {k: v for k, v in inputs.items() if k != "mrsosat"}
            out_of_range["mrsol"] = inputs["mrsol"].assign_attrs(units="1")
            xh.indicators.flood_types.major_flood_events(**out_of_range)


class TestSoilMoistureThreshold:
    REF = ["1995", "2004"]

    @staticmethod
    def _threshold(inputs, **kwargs):
        return xh.indicators.flood_types.soil_moisture_threshold(
            rivo=inputs["rivo"],
            prra=inputs["prra"],
            mrsol=inputs["mrsol"],
            mrsosat=inputs["mrsosat"],
            drainage_area=xr.DataArray(100.0, attrs={"units": "km2"}),
            **kwargs,
        )

    def test_threshold(self):
        """A significant SWI-to-peak correlation yields a finite threshold, its absence an infinite one."""
        correlated = self._threshold(_flood_inputs(correlated=True)[0], period=self.REF)
        uncorrelated = self._threshold(_flood_inputs(correlated=False)[0], period=self.REF)

        assert np.isfinite(correlated.item())
        assert np.isinf(uncorrelated.item())

    def test_threshold_few_events(self):
        # only 5 reference years, so fewer than the 10 events the Spearman gate needs
        threshold = self._threshold(_flood_inputs(correlated=True)[0], period=["1995", "1999"])

        assert np.isinf(threshold.item())

    def test_default_period(self):
        """`period=None` fits over the whole record."""
        inputs, _ = _flood_inputs(correlated=True)

        assert np.isfinite(self._threshold(inputs).item())

    def test_structure(self):
        inputs, _ = _flood_inputs(correlated=True)
        for name in ("mrsol", "prra", "rivo"):
            inputs[name] = inputs[name].expand_dims(station=["a", "b"]).transpose("time", "station")
        drainage_area = xr.DataArray([100.0, 100.0], coords={"station": ["a", "b"]}, dims="station", attrs={"units": "km2"})

        threshold = xh.indicators.flood_types.soil_moisture_threshold(
            rivo=inputs["rivo"], prra=inputs["prra"], mrsol=inputs["mrsol"], mrsosat=inputs["mrsosat"], drainage_area=drainage_area, period=self.REF
        )

        assert threshold.dims == ("station",)
        assert threshold.attrs["units"] == "1"

    def test_errors(self):
        inputs, _ = _flood_inputs(correlated=False)
        with pytest.raises(ValueError, match="km2"):
            xh.indicators.flood_types.soil_moisture_threshold(
                rivo=inputs["rivo"],
                prra=inputs["prra"],
                mrsol=inputs["mrsol"],
                mrsosat=inputs["mrsosat"],
                drainage_area=xr.DataArray(100.0, attrs={"units": "m"}),
            )
        with pytest.raises(ValueError, match="intersect"):
            self._threshold(inputs, period=["2050", "2060"])
        with pytest.raises(ValueError, match="chronological order"):
            self._threshold(inputs, period=["2004", "1995"])
        with pytest.raises(ValueError, match="`max_days` must be >= 1"):
            self._threshold(inputs, max_days=0)

    def test_decluster(self):
        # The merge semantics are unreachable deterministically from the public API,
        # so pin them directly, as TestSplitStreamflow does with its reference recursion.
        from xhydro.indicators.flood_types import _decluster

        q = np.ones(50)
        q[10], q[13] = 10.0, 8.0
        # too close: merged, larger peak wins regardless of order
        assert _decluster(q, np.array([10, 13]), min_days=5, discharge_threshold=2 / 3).tolist() == [10]
        q = np.ones(50)
        q[10], q[13] = 8.0, 10.0
        assert _decluster(q, np.array([10, 13]), min_days=5, discharge_threshold=2 / 3).tolist() == [13]
        # far apart with a deep trough: independent
        q = np.ones(50)
        q[10], q[30] = 10.0, 8.0
        assert _decluster(q, np.array([10, 30]), min_days=5, discharge_threshold=2 / 3).tolist() == [10, 30]
        # far apart but the flow never recedes below 2/3 of the smaller peak: merged
        q = np.full(50, 7.0)
        q[10], q[30] = 10.0, 8.0
        assert _decluster(q, np.array([10, 30]), min_days=5, discharge_threshold=2 / 3).tolist() == [10]


class TestClassifyFloodEvents:
    @staticmethod
    def _events(*, rain_sum, rain_max, melt_sum, swi, peak=20.0):
        """Build a one-event Dataset holding only the fields the decision tree reads."""
        values = {"prra_sum": rain_sum, "prra_max": rain_max, "snm_sum": melt_sum, "swi_antecedent": swi, "rivo_peak": peak}
        return xr.Dataset({name: xr.DataArray([value], dims="time") for name, value in values.items()})

    @pytest.mark.parametrize(
        "expected,rain_sum,rain_max,melt_sum,swi",
        [
            (0, 3.0, 1.0, 15.0, 0.0),  # snowmelt: melt > 4x rain
            (1, 3.0, 1.0, 9.0, 0.0),  # mostly snowmelt: melt > 2x rain
            (2, 15.0, 5.0, 6.0, 0.0),  # rain-on-snow: melt > 0.25x rain
            (3, 20.0, 18.0, 0.0, 0.9),  # soil water excess and short rain
            (4, 20.0, 5.0, 0.0, 0.9),  # soil water excess and long rain
            (5, 20.0, 18.0, 0.0, 0.1),  # short rain: one day holds > 75% of the rain
            (6, 20.0, 5.0, 0.0, 0.1),  # long rain
        ],
    )
    def test_types(self, expected, rain_sum, rain_max, melt_sum, swi):
        events = self._events(rain_sum=rain_sum, rain_max=rain_max, melt_sum=melt_sum, swi=swi)

        out = xh.indicators.flood_types.classify_flood_events(events, threshold=0.5)

        assert out.item() == expected

    def test_no_event(self):
        """A period without a peak is the -1 sentinel, whatever the other fields hold."""
        events = self._events(rain_sum=20.0, rain_max=5.0, melt_sum=0.0, swi=0.1, peak=np.nan)

        assert xh.indicators.flood_types.classify_flood_events(events, threshold=0.5).item() == -1

    def test_nan_swi_is_dry(self):
        """A missing antecedent SWI never exceeds the threshold, so it behaves like 0."""
        events = self._events(rain_sum=20.0, rain_max=18.0, melt_sum=0.0, swi=np.nan)

        assert xh.indicators.flood_types.classify_flood_events(events, threshold=0.5).item() == 5

    def test_infinite_threshold_disables_soil_water_excess(self):
        events = self._events(rain_sum=20.0, rain_max=5.0, melt_sum=0.0, swi=0.99)

        assert xh.indicators.flood_types.classify_flood_events(events, threshold=np.inf).item() == 6

    def test_attrs(self):
        events = self._events(rain_sum=20.0, rain_max=5.0, melt_sum=0.0, swi=0.1)

        out = xh.indicators.flood_types.classify_flood_events(events, threshold=0.5)

        assert out.dtype == np.int16
        assert out.attrs["flag_values"] == [0, 1, 2, 3, 4, 5, 6]
        assert len(out.attrs["flag_meanings"].split()) == 7

    def test_errors(self):
        events = self._events(rain_sum=20.0, rain_max=5.0, melt_sum=0.0, swi=0.1)
        with pytest.raises(ValueError, match="missing"):
            xh.indicators.flood_types.classify_flood_events(events[["rivo_peak"]], threshold=0.5)

    def test_end_to_end(self):
        """The three functions chain on real series, and a wet-soil event picks up a soil-water-excess type."""
        inputs, starts = _flood_inputs(correlated=True)
        _plant_flood_event(inputs, starts, rain=[0, 5, 5, 2])
        inputs["mrsol"].values[starts[11] :] = 95.0  # a wet soil going into the 2006 event

        threshold = xh.indicators.flood_types.soil_moisture_threshold(
            rivo=inputs["rivo"],
            prra=inputs["prra"],
            mrsol=inputs["mrsol"],
            mrsosat=inputs["mrsosat"],
            drainage_area=xr.DataArray(100.0, attrs={"units": "km2"}),
            period=["1995", "2004"],
        )
        events = xh.indicators.flood_types.major_flood_events(**inputs)
        flood_type = xh.indicators.flood_types.classify_flood_events(events, threshold=threshold)

        assert flood_type.sizes["time"] == events.sizes["time"]
        assert flood_type.sel(time=_EVENT_PERIOD).item() == 4  # soil water excess and long rain
