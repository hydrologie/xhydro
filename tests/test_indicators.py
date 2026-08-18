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
        da = timeseries(np.full(10, 7.0), variable="q", start="2001-01-01", freq="D")

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


class TestMajorFloodEvents:
    # 12 years of daily data; the last reference year leaves 2005 as a buffer
    # and 2006 as the year whose event each test engineers.
    REF = (1995, 2004)

    @staticmethod
    def _inputs(*, correlated, end="2006-12-31"):
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
            "drainage_area": xr.DataArray(100.0, attrs={"units": "km2"}),
            "reference_period": TestMajorFloodEvents.REF,
        }
        return inputs, starts

    @staticmethod
    def _event(inputs, starts, *, rain, snow=None, peak=20.0):
        """Plant the 2006 annual maximum at day-of-year 201, with `rain` (and `snow`) ending on the peak day."""
        p = starts[11] + 200
        rain = np.asarray(rain, dtype=float)
        inputs["rivo"].values[p] = peak
        inputs["prra"].values[p - rain.size + 1 : p + 1] = rain
        if snow is not None:
            snow = np.asarray(snow, dtype=float)
            inputs["snm"].values[p - snow.size + 1 : p + 1] = snow
        return p

    def _flood_type(self, inputs):
        return xh.indicators.flood_types.major_flood_events(**inputs).flood_type.sel(time="2006-01-01").item()

    def test_snowmelt(self):
        inputs, starts = self._inputs(correlated=False)
        self._event(inputs, starts, rain=[1, 1, 1], snow=[5, 5, 5])

        assert self._flood_type(inputs) == 0

    def test_mostly_snowmelt_with_some_rainfall(self):
        inputs, starts = self._inputs(correlated=False)
        self._event(inputs, starts, rain=[2, 2, 2], snow=[5, 5, 5])

        assert self._flood_type(inputs) == 1

    def test_rain_on_snow(self):
        inputs, starts = self._inputs(correlated=False)
        self._event(inputs, starts, rain=[4, 4, 4], snow=[2, 2, 2])

        assert self._flood_type(inputs) == 2

    def test_soil_excess_and_short_rain(self):
        inputs, starts = self._inputs(correlated=True)
        inputs["mrsol"].loc[{"time": slice("2006", None)}] = 99.0
        self._event(inputs, starts, rain=[20])

        assert self._flood_type(inputs) == 3

    def test_soil_excess_and_long_rain(self):
        inputs, starts = self._inputs(correlated=True)
        inputs["mrsol"].loc[{"time": slice("2006", None)}] = 99.0
        self._event(inputs, starts, rain=[5] * 7)

        assert self._flood_type(inputs) == 4

    def test_short_rain(self):
        inputs, starts = self._inputs(correlated=False)
        self._event(inputs, starts, rain=[20])

        assert self._flood_type(inputs) == 5

    def test_long_rain(self):
        inputs, starts = self._inputs(correlated=False)
        # 9 rainy days, but max_days=7 caps the window
        self._event(inputs, starts, rain=[5] * 9)

        out = xh.indicators.flood_types.major_flood_events(**inputs).sel(time="2006-01-01")
        assert out.flood_type.item() == 6
        assert out.event_duration.item() == 7

    def test_threshold(self):
        correlated = xh.indicators.flood_types.major_flood_events(**self._inputs(correlated=True)[0])
        uncorrelated = xh.indicators.flood_types.major_flood_events(**self._inputs(correlated=False)[0])

        # the threshold is one of the events' antecedent SWI values
        assert 0.15 <= correlated.swi_threshold.item() <= 0.78
        # constant SWI: the Spearman gate opts out, so no soil-water-excess type can trigger
        assert np.isinf(uncorrelated.swi_threshold.item())
        assert not np.isin(uncorrelated.flood_type, [3, 4]).any()

    def test_threshold_few_events(self):
        inputs, _ = self._inputs(correlated=True)
        inputs["reference_period"] = (1995, 1999)  # 5 spikes < 10 events

        out = xh.indicators.flood_types.major_flood_events(**inputs)
        assert np.isinf(out.swi_threshold.item())

    def test_indicators(self):
        inputs, starts = self._inputs(correlated=False)
        # rain [0, 5, 5, 2]: the walk from the peak stops at the dry day, so the window is 3 days
        self._event(inputs, starts, rain=[0, 5, 5, 2], snow=[0, 1, 1, 1])
        inputs["mrros"] = (0.25 * inputs["rivo"]).assign_attrs(units="mm")

        out = xh.indicators.flood_types.major_flood_events(**inputs).sel(time="2006-01-01")
        assert out.rivo_peak.item() == 20.0
        assert out.rivo_peak_doy.item() == 201
        assert out.event_duration.item() == 3
        assert out.prra_sum.item() == 12.0
        assert out.prra_max.item() == 5.0
        assert out.snm_sum.item() == 3.0
        assert out.swi_antecedent.item() == 0.5  # mrsol 50 / mrsosat 100 on the day before the window
        np.testing.assert_allclose(out.direct_streamflow_fraction.item(), 0.25)

    def test_mrros_default(self):
        inputs, starts = self._inputs(correlated=False)
        self._event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])

        from_none = xh.indicators.flood_types.major_flood_events(**inputs)
        from_explicit = xh.indicators.flood_types.major_flood_events(**inputs, mrros=xh.indicators.split_streamflow(inputs["rivo"])[1])

        xr.testing.assert_identical(from_none, from_explicit)

    def test_structure(self):
        inputs, starts = self._inputs(correlated=False)
        self._event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])
        for name in ("mrsol", "prra", "rivo", "snm"):
            inputs[name] = inputs[name].expand_dims(station=["a", "b"]).transpose("time", "station")
        inputs["drainage_area"] = xr.DataArray([100.0, 100.0], coords={"station": ["a", "b"]}, dims="station", attrs={"units": "km2"})

        out = xh.indicators.flood_types.major_flood_events(**inputs)
        assert out.flood_type.dims == ("time", "station")
        assert out.flood_type.dtype == np.int16
        assert out.swi_threshold.dims == ("station",)
        # December 2006 spills into a 13th year, incomplete like the December-less 1995
        assert out.time.size == 13
        assert out.time.dt.month.values.tolist() == [1] * 13
        assert out.flood_type.attrs["flag_values"] == [0, 1, 2, 3, 4, 5, 6]
        assert len(out.flood_type.attrs["flag_meanings"].split()) == 7
        assert (out.flood_type.sel(station="a") == out.flood_type.sel(station="b")).all()

    def test_dask(self):
        inputs, starts = self._inputs(correlated=False)
        self._event(inputs, starts, rain=[2, 2, 2], snow=[3, 3, 3])
        eager = xh.indicators.flood_types.major_flood_events(**inputs)

        for name in ("mrsol", "prra", "rivo", "snm"):
            inputs[name] = inputs[name].chunk({"time": -1})
        lazy = xh.indicators.flood_types.major_flood_events(**inputs)

        xr.testing.assert_identical(eager, lazy.compute())

    def test_incomplete_year(self):
        inputs, _ = self._inputs(correlated=False, end="2006-06-30")

        out = xh.indicators.flood_types.major_flood_events(**inputs).isel(time=-1)
        assert out.flood_type.item() == -1
        assert np.isnan(out.rivo_peak.item())
        assert np.isnan(out.prra_sum.item())

    def test_errors(self):
        inputs, _ = self._inputs(correlated=False)
        with pytest.raises(ValueError, match="missing"):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "prra": inputs["prra"].drop_attrs()})
        with pytest.raises(ValueError, match='convertible to "mm"'):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "prra": inputs["prra"].assign_attrs(units="K")})
        with pytest.raises(ValueError, match="km2"):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "drainage_area": xr.DataArray(100.0, attrs={"units": "m"})})
        with pytest.raises(ValueError, match="time coordinate"):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "snm": inputs["snm"].isel(time=slice(0, 100))})
        with pytest.raises(ValueError, match="daily"):
            weekly = {k: v.isel(time=slice(None, None, 7)) if isinstance(v, xr.DataArray) and "time" in v.dims else v for k, v in inputs.items()}
            xh.indicators.flood_types.major_flood_events(**weekly)
        with pytest.raises(ValueError, match="<start>"):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "reference_period": (2004, 1995)})
        with pytest.raises(ValueError, match="intersect"):
            xh.indicators.flood_types.major_flood_events(**{**inputs, "reference_period": (2050, 2060)})
        with pytest.raises(ValueError, match="`max_days` must be >= 1"):
            xh.indicators.flood_types.major_flood_events(**inputs, max_days=0)

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
