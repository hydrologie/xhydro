"""Module to classify flood events."""

import warnings
from collections.abc import Hashable
from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.stats
import xarray as xr
import xscen
from xclim.core.units import convert_units_to, units2pint

from .generic import split_streamflow


__all__ = ["classify_flood_events", "major_flood_events", "soil_moisture_threshold"]


def soil_moisture_threshold(
    *,
    rivo: xr.DataArray,
    prra: xr.DataArray,
    mrsol: xr.DataArray,
    drainage_area: xr.DataArray,
    mrsosat: xr.DataArray | None = None,
    period: list[str] | None = None,
    max_days: int = 7,
    min_prec: float = 1,
) -> xr.DataArray:
    """
    Fit the soil moisture threshold separating wet from dry antecedent conditions.

    The threshold is the soil wetness index (SWI) above which reference-period
    floods stop growing with antecedent wetness. Feed it to
    :py:func:`xhydro.indicators.flood_types.classify_flood_events`, which uses it
    to split rain-driven events into soil-water-excess types and plain rain types.

    Parameters
    ----------
    rivo : xr.DataArray
        Streamflow, expressed as a water depth per day (mm d-1).
    prra : xr.DataArray
        Rainfall (mm d-1).
    mrsol : xr.DataArray
        Soil water content (mm) if ``mrsosat`` is given, otherwise the soil
        wetness index in [0, 1] directly, which must then be dimensionless.
    drainage_area : xr.DataArray
        Drainage area, convertible to km2. Only used to scale the independence
        criterion when declustering the reference-period events.
    mrsosat : xr.DataArray | None, default: None
        Water the soil holds at saturation (mm), i.e. its porosity times its
        depth. Static (no time dependence needed). If given, ``mrsol`` is
        normalized on the fly as ``mrsol / mrsosat``; if None, ``mrsol`` is
        taken to be normalized already.
    period : list of str | None, default: None
        First and last year (inclusive) of the period used to fit the threshold,
        standardized with :py:func:`xscen.utils.standardize_periods`. Must be a
        list, not a tuple. If None, the whole record is used.
    max_days : int, default: 7
        Maximum number of days for an event.
    min_prec : float, default: 1
        Minimum daily rainfall (mm d-1) to extend an event backward from its peak.

    Returns
    -------
    xr.DataArray
        The fitted threshold, with no time axis. ``inf`` where fewer than 10
        reference events exist or antecedent SWI and peak flow are uncorrelated,
        in which case the soil-water-excess types never trigger.

    Notes
    -----
    Pass the same ``max_days`` and ``min_prec`` you pass to
    :py:func:`xhydro.indicators.flood_types.classify_flood_events`. Both functions
    build event windows the same way, and the threshold is only comparable to the
    antecedent SWI of events delimited identically.

    The threshold follows Tramblay et al. (2022) [1]_: reference-period events are
    the declustered days with streamflow above its 10th percentile; if at least 10
    events show a significant Spearman correlation (p < 0.05) between antecedent
    SWI and peak flow, an exponential curve is fitted and the threshold is the SWI
    at the split minimizing the two-segment squared error; otherwise the threshold
    is ``inf``. The split is the exact single-changepoint solution of the PELT
    method [2]_ used in [1]_; deviations from [1]_ are flagged in the inline
    comments.

    The fit is not grouped by year: it thresholds the whole ``period`` slice at
    once, declusters it and fits. ``period`` therefore covers calendar years, and
    does not follow the December-to-November year that
    :py:func:`xhydro.indicators.flood_types.major_flood_events` uses by default.
    The difference amounts to one month of data at each edge of the slice.

    The SWI (``mrsol / mrsosat``) follows Tramblay et al. (2025) [3]_, which
    scales soil moisture by the model store's capacity, so it stays within
    [0, 1] as long as both inputs cover the same soil column. Tramblay et al.
    (2022) [1]_ instead scales by soil moisture's long-term range; the per-cell
    threshold absorbs that difference, since a positive rescaling of the index
    carries the fitted threshold with it.

    References
    ----------
    .. [1] Tramblay, Y., Villarini, G., Saidi, M. E., Massari, C., & Stein, L. (2022). Classification of flood-generating processes in Africa.
       Scientific Reports, 12, 18920. https://doi.org/10.1038/s41598-022-23725-5
    .. [2] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost.
       Journal of the American Statistical Association, 107(500), 1590-1598. https://doi.org/10.1080/01621459.2012.737745
    .. [3] Tramblay, Y., Thirel, G., Strohmenger, L., Evin, G., Corre, L., Heraut, L., & Sauquet, E. (2025). Evolution of flood generating
       processes under climate change in France. Hydrology and Earth System Sciences, 29, 7023-7039. https://doi.org/10.5194/hess-29-7023-2025
    """
    if max_days < 1:
        raise ValueError(f"`max_days` must be >= 1: is {max_days}.")
    variables = {"rivo": rivo, "prra": prra, "mrsol": mrsol}
    if mrsosat is not None:
        variables["mrsosat"] = mrsosat
    _validate_inputs(variables, drainage_area=drainage_area, normalized_mrsol=mrsosat is None)
    # `min_prec` is compared to raw `prra` values, so the fluxes must all be on
    # the same scale before any of them is read
    rivo = cast(xr.DataArray, convert_units_to(rivo, "mm d-1"))
    prra = cast(xr.DataArray, convert_units_to(prra, "mm d-1"))

    years = (
        cast(list[str], xscen.utils.standardize_periods(cast(list, period), multiple=False))
        if period is not None
        else [str(int(rivo.time.dt.year.min())), str(int(rivo.time.dt.year.max()))]
    )
    reference = slice(years[0], years[1])
    if rivo.sel(time=reference).time.size == 0:
        raise ValueError(f"`period` {years} does not intersect the data's time range.")

    antecedent_swi = _antecedent_swi(mrsol, mrsosat)
    # Tramblay et al. (2022) decluster with "5 + log(catchment area)" days
    # between events: larger basins integrate flow over longer times, so their
    # events must be further apart. The paper gives neither the log base nor
    # the area unit; we read it as log10 of km2.
    min_days = 5 + xr.ufuncs.log10(convert_units_to(drainage_area, "km2"))

    threshold = xr.apply_ufunc(
        _threshold_kernel,
        rivo.sel(time=reference),
        prra.sel(time=reference),
        antecedent_swi.sel(time=reference),
        min_days,
        input_core_dims=[["time"], ["time"], ["time"], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        kwargs={"max_days": max_days, "min_prec": min_prec},
    )
    threshold.attrs = {
        "units": "1",
        "long_name": "Soil moisture threshold",
        "description": "Changepoint of the exponential fit of reference-period peak flows against antecedent SWI; "
        "inf where fewer than 10 events exist or the correlation is not significant.",
    }
    return threshold.rename("swi_threshold")


def major_flood_events(*, rivo: xr.DataArray, freq: str = "YS-DEC") -> xr.DataArray:
    """
    Extract the date of the major flood event of each period.

    The event is the day of maximum streamflow of the period. The default
    ``freq`` runs years from 1 December to 30 November, so a winter belongs to a
    single year.

    Feed the result to
    :py:func:`xhydro.indicators.flood_types.classify_flood_events` to describe
    and classify those events.

    Parameters
    ----------
    rivo : xr.DataArray
        Streamflow. Only the ordering of its values matters here, so any unit will do.
    freq : str, default: "YS-DEC"
        Resampling frequency delimiting the periods. The default runs years from
        1 December to 30 November; "YS" gives calendar years, "YS-OCT" an
        October-to-September hydrological year, and "QS-DEC" seasons.

    Returns
    -------
    xr.DataArray
        Date of each period's maximum streamflow, on a "time" axis labelled by
        each period's first day. A period whose streamflow is all missing gets
        NaT. Periods truncated by the start or end of the record are kept, so
        their peak is drawn from a partial window.
    """
    if "time" not in rivo.dims:
        raise ValueError('`rivo` must have a "time" dimension.')

    # `idxmax` gives NaT for an all-missing period, which
    # `classify_flood_events` reads back as "this period holds no event"
    dates = rivo.resample(time=freq).map(lambda period: period.idxmax("time"))
    dates.attrs = {
        "long_name": "Date of the major flood event",
        "description": "Date of the period's maximum streamflow; NaT where the period holds no streamflow.",
    }
    return dates.rename("event_date").transpose(*rivo.dims)


def classify_flood_events(
    dates: xr.DataArray | str | np.datetime64,
    *,
    rivo: xr.DataArray,
    prra: xr.DataArray,
    snm: xr.DataArray,
    mrsol: xr.DataArray,
    threshold: xr.DataArray | float,
    mrsosat: xr.DataArray | None = None,
    mrros: xr.DataArray | None = None,
    max_days: int = 7,
    min_prec: float = 1,
) -> xr.Dataset:
    """
    Describe flood events and classify them into seven types.

    The event is the given day plus the contiguous run of rainy days
    (``prra >= min_prec``) directly before it, up to ``max_days`` days in total.
    Rainfall, snowmelt and antecedent wetness are summarized over that window,
    then fed to the decision tree.

    Parameters
    ----------
    dates : xr.DataArray | str | np.datetime64
        Either the event dates returned by
        :py:func:`xhydro.indicators.flood_types.major_flood_events`, which must
        carry a "time" dimension and its coordinate, or a single date given as
        anything ``sel`` accepts: a string, a numpy or python datetime, a cftime
        object, or a scalar DataArray. A NaT date means there is no event.
    rivo : xr.DataArray
        Streamflow, expressed as a water depth per day (mm d-1).
    prra : xr.DataArray
        Rainfall (mm d-1).
    snm : xr.DataArray
        Snowmelt (mm d-1).
    mrsol : xr.DataArray
        Soil water content (mm) if ``mrsosat`` is given, otherwise the soil
        wetness index in [0, 1] directly, which must then be dimensionless.
    threshold : xr.DataArray | float
        Soil moisture threshold above which antecedent conditions count as wet,
        typically from
        :py:func:`xhydro.indicators.flood_types.soil_moisture_threshold`. Pass
        ``inf`` to disable the two soil-water-excess types.
    mrsosat : xr.DataArray | None, default: None
        Water the soil holds at saturation (mm), i.e. its porosity times its
        depth. Static (no time dependence needed). If given, ``mrsol`` is
        normalized on the fly as ``mrsol / mrsosat``; if None, ``mrsol`` is
        taken to be normalized already.
    mrros : xr.DataArray | None, default: None
        Surface runoff (mm d-1). If None, it is derived from ``rivo`` with
        :py:func:`xhydro.indicators.split_streamflow`.
    max_days : int, default: 7
        Maximum number of days for an event.
    min_prec : float, default: 1
        Minimum daily rainfall (mm d-1) to extend an event backward from its peak.

    Returns
    -------
    xr.Dataset
        One value per date, on the "time" axis of ``dates``; a single date gives
        a Dataset with no time dimension. An event whose date is missing gets
        NaN everywhere and a flood type of -1.

        - ``rivo_peak``, ``rivo_peak_doy``: streamflow of the event day and its
          calendar day of year.
        - ``event_duration``: length of the event window (days).
        - ``prra_sum``, ``prra_max``: total and maximum daily rainfall over the
          event window.
        - ``snm_sum``: total snowmelt over the event window.
        - ``swi_antecedent``: SWI on the day before the event window starts. An
          event whose window starts on the record's first day has none, and gets
          NaN.
        - ``direct_streamflow_fraction``: the ratio of surface runoff to
          streamflow on the event day, using ``mrros`` or the Lyne-Hollick runoff
          if ``mrros`` is None.
        - ``flood_type``: integer code of the event's type (see ``flag_values`` /
          ``flag_meanings``).

    Notes
    -----
    Pass the same ``max_days`` and ``min_prec`` you pass to
    :py:func:`xhydro.indicators.flood_types.soil_moisture_threshold`, so that the
    threshold and the events it classifies rest on the same event definition.

    The classification is a modification of the decision tree of Tramblay et
    al. (2025) [1]_, applied in order: "snowmelt" if total snowmelt exceeds
    four times the total rainfall; "mostly snowmelt with some rainfall" if it
    exceeds twice the rainfall; "rain-on-snow" if it exceeds a quarter of it;
    then "short rain" if the largest rain day holds more than 75% of the
    event's rainfall, "long rain" otherwise, with a "soil water excess and"
    prefix when the antecedent SWI exceeds ``threshold``.
    Modifications from [1]_: the paper's two snow classes ("snowmelt" above
    1x the rainfall, "rain and snowmelt" above 0.25x) become three with
    cutoffs at 4x, 2x and 0.25x, and its single "soil water excess" class is
    split by the short/long rain criterion.

    A NaN antecedent SWI never exceeds the threshold, so it is treated as 0.

    References
    ----------
    .. [1] Tramblay, Y., Thirel, G., Strohmenger, L., Evin, G., Corre, L., Heraut, L., & Sauquet, E. (2025). Evolution of flood generating
       processes under climate change in France. Hydrology and Earth System Sciences, 29, 7023-7039. https://doi.org/10.5194/hess-29-7023-2025
    """
    if max_days < 1:
        raise ValueError(f"`max_days` must be >= 1: is {max_days}.")
    variables = {"rivo": rivo, "prra": prra, "snm": snm, "mrsol": mrsol}
    if mrsosat is not None:
        variables["mrsosat"] = mrsosat
    if mrros is not None:
        variables["mrros"] = mrros
    _validate_inputs(variables, normalized_mrsol=mrsosat is None)
    # `min_prec` is compared to raw `prra` values and `direct_streamflow_fraction`
    # divides `mrros` by `rivo`, so the fluxes must all be on the same scale
    rivo = cast(xr.DataArray, convert_units_to(rivo, "mm d-1"))
    prra = cast(xr.DataArray, convert_units_to(prra, "mm d-1"))
    snm = cast(xr.DataArray, convert_units_to(snm, "mm d-1"))

    # `rivo` is already converted, so the derived runoff inherits "mm d-1"
    mrros = cast(xr.DataArray, convert_units_to(mrros, "mm d-1")) if mrros is not None else split_streamflow(rivo)[1]

    antecedent_swi = _antecedent_swi(mrsol, mrsosat)

    scalar = not isinstance(dates, xr.DataArray) or "time" not in dates.dims
    if scalar:
        # `sel` resolves strings, datetimes and cftime objects against the
        # record's own calendar
        resolved = np.atleast_1d(rivo.time.sel(time=dates).values)
        if resolved.size != 1:
            raise ValueError(f"A single date must match a single day: `{dates}` matches {resolved.size} of them.")
        dates = xr.DataArray(resolved, coords={"time": resolved}, dims="time")
    elif "time" not in dates.coords:
        raise ValueError('`dates` must carry its "time" coordinate, as returned by `major_flood_events`.')

    outputs = xr.apply_ufunc(
        _events_kernel,
        rivo,
        prra,
        snm,
        mrros,
        antecedent_swi,
        rivo.time.dt.dayofyear,
        _peak_index(rivo.time, dates),
        input_core_dims=[["time"]] * 6 + [["event"]],
        output_core_dims=[["event"]] * 8,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64] * 8,
        dask_gufunc_kwargs={"output_sizes": {"event": dates.sizes["time"]}},
        kwargs={"max_days": max_days, "min_prec": min_prec},
    )
    out = _build_output(outputs, dims=rivo.dims, time=dates.time.values)
    out["flood_type"] = _flood_type(out, threshold)
    # a single date is a single event: the time axis carries no information the
    # caller does not already have
    return out.isel(time=0) if scalar else out


def _peak_index(time: xr.DataArray, dates: xr.DataArray) -> xr.DataArray:
    """
    Return the position of each date within `time`, on an "event" dimension.

    A missing date maps to -1, which `_events_kernel` reads as "no event"; a date
    that is absent from the record is a caller error and raises.
    """
    index = time.get_index("time").get_indexer(np.ravel(dates.values))
    if (index[np.ravel(dates.notnull().values)] < 0).any():
        raise ValueError("`dates` holds dates that are absent from `rivo`'s time coordinate.")
    return xr.DataArray(index.reshape(dates.shape), coords=dates.coords, dims=dates.dims).rename(time="event")


def _flood_type(events: xr.Dataset, threshold: xr.DataArray | float) -> xr.DataArray:
    """Apply the decision tree of Tramblay et al. (2025) to the event indicators, returning -1 where there is no event."""
    flood_types = (
        "snowmelt",
        "mostly_snowmelt_with_some_rainfall",
        "rain_on_snow",
        "soil_water_excess_and_short_rain",
        "soil_water_excess_and_long_rain",
        "short_rain",
        "long_rain",
    )

    snowmelt, rain_sum = events["snm_sum"], events["prra_sum"]
    # a NaN comparison is False, which is what makes a NaN antecedent SWI
    # behave like 0
    short_rain = events["prra_max"] > 0.75 * rain_sum
    wet = events["swi_antecedent"] > threshold

    rain_type = xr.where(wet, xr.where(short_rain, 3, 4), xr.where(short_rain, 5, 6))
    flood_type = xr.where(
        snowmelt > 4 * rain_sum,
        0,
        xr.where(snowmelt > 2 * rain_sum, 1, xr.where(snowmelt > 0.25 * rain_sum, 2, rain_type)),
    )
    # `rivo_peak` is NaN exactly where the date held no usable event; the
    # other variables are NaN in legitimate cases too, so keying the sentinel
    # on them would mislabel events
    flood_type = flood_type.where(events["rivo_peak"].notnull(), -1).astype(np.int16)

    flood_type.attrs = {
        "long_name": "Flood type",
        "flag_values": list(range(len(flood_types))),
        "flag_meanings": " ".join(flood_types),
        "description": "Type of the flood event; -1 where there is no event.",
    }
    return flood_type.rename("flood_type").transpose(*events["rivo_peak"].dims)


def _antecedent_swi(mrsol: xr.DataArray, mrsosat: xr.DataArray | None) -> xr.DataArray:
    """Return the soil wetness index of the day preceding each time step, normalizing `mrsol` if `mrsosat` is given."""
    if mrsosat is None:
        # converting rather than passing through rescales a `mrsol` given in
        # percent to the 0-1 index
        swi = cast(xr.DataArray, convert_units_to(mrsol, ""))
        min_ = float(swi.min())
        max_ = float(swi.max())
        if min_ < 0 or max_ > 1:
            raise ValueError(f"Without `mrsosat`, `mrsol` must be a 0-1 index: ranges from {min_} to {max_}.")
    else:
        # the two stores may be reported in different depth units, so align
        # them before dividing
        swi = mrsol / convert_units_to(mrsosat, mrsol)
    return swi.shift(time=1)


def _validate_inputs(
    variables: dict[str, xr.DataArray],
    *,
    drainage_area: xr.DataArray | None = None,
    normalized_mrsol: bool = False,
) -> None:
    """Validate units and time alignment of the daily variables; raises ValueError on the first violation."""
    named = {**variables} if drainage_area is None else {**variables, "drainage_area": drainage_area}
    if missing := [name for name, variable in named.items() if "units" not in variable.attrs]:
        raise ValueError("All variables must have units. These variables are missing them: {}.".format(", ".join(missing)))

    # `mrsol` and `mrsosat` are stocks (a water depth), every other variable is a
    # flux (a water depth per unit of time); without `mrsosat`, `mrsol` is already
    # a 0-1 index, so it is dimensionless rather than a water depth
    stocks = {name: variable for name, variable in variables.items() if name in ("mrsol", "mrsosat") and not (normalized_mrsol and name == "mrsol")}
    fluxes = {name: variable for name, variable in variables.items() if name not in ("mrsol", "mrsosat")}
    for group, reference in ((fluxes, "mm d-1"), (stocks, "mm")):
        pint_reference = units2pint(reference)
        if bad := {name: v.attrs["units"] for name, v in group.items() if not units2pint(v).is_compatible_with(pint_reference, "hydro")}:
            raise ValueError(
                f'These variable units must be convertible to "{reference}": {", ".join(f"{name}: {units}" for name, units in bad.items())}.'
            )
    if normalized_mrsol and "mrsol" in variables and not units2pint(variables["mrsol"]).dimensionless:
        raise ValueError(f'Without `mrsosat`, `mrsol` must be a dimensionless 0-1 index: is "{variables["mrsol"].attrs["units"]}".')

    if drainage_area is not None:
        if not units2pint(drainage_area).is_compatible_with(units2pint("km2")):
            raise ValueError(f'`drainage_area` units must be convertible to "km2": is "{drainage_area.attrs["units"]}".')
        if "time" in drainage_area.dims:
            raise ValueError("`drainage_area` must not have a time dimension.")

    if "time" not in variables["rivo"].dims:
        raise ValueError('`rivo` must have a "time" dimension.')
    time = variables["rivo"].time
    for name, variable in variables.items():
        if name in ("mrsosat", "rivo"):
            continue
        if "time" not in variable.dims or not variable.time.equals(time):
            raise ValueError(f"`{name}` must share `rivo`'s time coordinate.")
    # declustering and event windows count time steps, so a step must be a day
    if xr.infer_freq(time) != "D":
        raise ValueError("All variables must have a daily time step.")


def _threshold_kernel(
    q: npt.NDArray[np.floating],
    prra: npt.NDArray[np.floating],
    antecedent_swi: npt.NDArray[np.floating],
    min_days: float,
    *,
    max_days: int,
    min_prec: float,
) -> float:
    """
    Reduce one reference-period series to its soil moisture threshold (Tramblay et al., 2022).

    Returns inf when fewer than 10 events have a finite antecedent SWI or when
    antecedent SWI and peak flow show no significant Spearman correlation.
    """
    if np.isnan(q).all():
        return np.inf
    candidates = np.flatnonzero(q >= np.nanquantile(q, 0.1))
    if candidates.size == 0:
        return np.inf
    kept = _decluster(q, candidates, min_days=float(min_days), discharge_threshold=2 / 3)

    swi = np.empty(kept.size)
    peaks = np.empty(kept.size)
    for i, peak in enumerate(kept):
        # deviations from Tramblay et al. (2022): they discard events with no
        # rainfall the days before the peak, which we keep because
        # snowmelt-dominated basins could lose most of their events; their
        # event window is capped only by a dry day, while we also reuse the
        # classification cap (`max_days`) so both event definitions stay
        # coherent
        start = _window_start(prra, peak, max_days=max_days, min_prec=min_prec)
        swi[i] = antecedent_swi[start]
        peaks[i] = np.max(q[start : peak + 1])
    finite = np.isfinite(swi)
    swi, peaks = swi[finite], peaks[finite]

    # deviation from Tramblay et al. (2022), which only gates on significance:
    # below 10 events the Spearman p-value is too coarse to mean anything
    if swi.size < 10:
        return np.inf
    with warnings.catch_warnings():
        # a constant series is a legitimate opt-out, handled through its NaN
        # p-value below
        warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)
        p_value = scipy.stats.spearmanr(swi, peaks).pvalue
    if not p_value < 0.05:  # "not <" so a NaN p-value (constant series) also opts out
        return np.inf
    # the form a * exp(b * x) comes from Tarasova et al. (2018) via Tramblay et
    # al. (2022), but no paper in the citation chain names the estimator; least
    # squares on the log is the simplest
    b, a = np.polyfit(swi, np.log(peaks + 1e-16), 1)
    return _changepoint(swi, a=np.exp(a), b=b)


def _decluster(q: npt.NDArray[np.floating], candidates: npt.NDArray[np.intp], *, min_days: float, discharge_threshold: float) -> npt.NDArray[np.intp]:
    """
    Merge dependent candidate peaks, returning the indices of the independent ones.

    Declustering rules of Tramblay et al. (2022): two peaks are independent when they
    are at least `min_days` apart AND the trough between them drops below
    `discharge_threshold` times the smaller peak; otherwise they merge and the larger
    peak survives.
    """
    kept = []
    current_idx, current_q, current_qmin = candidates[0], q[candidates[0]], np.inf
    previous_idx = candidates[0]
    for idx in candidates[1:]:
        # the trough since the previous candidate, including the candidate day
        # itself
        qmin = min(np.nanmin(q[previous_idx + 1 : idx + 1]), current_qmin)
        independent = (idx - current_idx >= min_days) and (qmin < discharge_threshold * min(q[idx], current_q))
        if independent:
            kept.append(current_idx)
            current_idx, current_q, current_qmin = idx, q[idx], np.inf
        elif q[idx] > current_q:
            current_idx, current_q, current_qmin = idx, q[idx], np.inf
        else:
            current_qmin = min(qmin, q[idx])
        previous_idx = idx
    kept.append(current_idx)
    return np.asarray(kept, dtype=np.intp)


def _window_start(prra: npt.NDArray[np.floating], peak: int, *, max_days: int, min_prec: float) -> int:
    """
    Return the index of the first day of the event window ending at `peak`.

    The window holds the peak day plus the contiguous rainy days (`prra >= min_prec`)
    before it, up to `max_days` days in total; a NaN rain day stops the walk.
    """
    start = peak
    stop = max(peak - max_days + 1, 0)
    while start > stop and prra[start - 1] >= min_prec:
        start -= 1
    return start


def _changepoint(swi: npt.NDArray[np.floating], *, a: float, b: float) -> float:
    """
    Return the SWI splitting the fitted exponential curve into the two segments with the least total squared error.

    Tramblay et al. (2022) use PELT (Killick et al., 2012) here; for a single changepoint with the same
    squared-error cost, this exhaustive search is the exact solution, and the Spearman gate upstream
    plays the role of PELT's penalty by rejecting series with no changepoint worth finding.
    """
    x = np.sort(swi)
    y = a * np.exp(b * x)
    s1, s2 = np.cumsum(y), np.cumsum(y**2)
    n = len(y)
    i = np.arange(1, n)
    sse_left = s2[i - 1] - s1[i - 1] ** 2 / i
    sse_right = (s2[-1] - s2[i - 1]) - (s1[-1] - s1[i - 1]) ** 2 / (n - i)
    best = i[np.argmin(sse_left + sse_right)]
    return float(x[best])


def _events_kernel(
    q: npt.NDArray[np.floating],
    prra: npt.NDArray[np.floating],
    snm: npt.NDArray[np.floating],
    mrros: npt.NDArray[np.floating],
    antecedent_swi: npt.NDArray[np.floating],
    doy: npt.NDArray[np.integer],
    peaks: npt.NDArray[np.intp],
    *,
    max_days: int,
    min_prec: float,
) -> tuple[npt.NDArray[np.floating], ...]:
    """
    Compute the flood event indicators of one series, one value per event.

    Returns an 8-tuple of arrays holding one entry per index in `peaks`; an event
    whose index is the -1 sentinel, or whose day has no streamflow, gets NaN.
    """
    indicators = np.full((8, peaks.size), np.nan)
    peak_q, peak_doy, duration, rain_sum, rain_max, melt_sum, swi, fraction = indicators

    for i, peak in enumerate(peaks):
        peak = int(peak)
        if peak < 0 or np.isnan(q[peak]):
            continue
        start = _window_start(prra, peak, max_days=max_days, min_prec=min_prec)
        rain = prra[start : peak + 1]

        peak_q[i] = q[peak]
        peak_doy[i] = doy[peak]
        duration[i] = peak - start + 1
        rain_sum[i] = np.nansum(rain)
        rain_max[i] = np.nan if np.isnan(rain).all() else np.nanmax(rain)
        melt_sum[i] = np.nansum(snm[start : peak + 1])
        swi[i] = antecedent_swi[start]
        fraction[i] = mrros[peak] / q[peak]
    return (peak_q, peak_doy, duration, rain_sum, rain_max, melt_sum, swi, fraction)


def _build_output(outputs: tuple[xr.DataArray, ...], *, dims: tuple[Hashable, ...], time: npt.NDArray) -> xr.Dataset:
    """Assemble the per-event output Dataset: time axis and attributes."""
    names = (
        "rivo_peak",
        "rivo_peak_doy",
        "event_duration",
        "prra_sum",
        "prra_max",
        "snm_sum",
        "swi_antecedent",
        "direct_streamflow_fraction",
    )
    out = xr.Dataset(dict(zip(names, outputs, strict=True)))
    out = out.rename(event="time").assign_coords(time=time)

    attrs: dict[str, dict] = {
        "rivo_peak": {"units": "mm d-1", "long_name": "Streamflow of the flood event"},
        "rivo_peak_doy": {"units": "1", "long_name": "Day of year of the flood event"},
        "event_duration": {
            "units": "d",
            "long_name": "Flood event duration",
            "description": "Length of the contiguous rain window ending on the event's day.",
        },
        "prra_sum": {"units": "mm", "long_name": "Total rainfall during the flood event", "cell_methods": "time: sum"},
        "prra_max": {"units": "mm d-1", "long_name": "Maximum daily rainfall during the flood event", "cell_methods": "time: maximum"},
        "snm_sum": {"units": "mm", "long_name": "Total snowmelt during the flood event", "cell_methods": "time: sum"},
        "swi_antecedent": {
            "units": "1",
            "long_name": "Antecedent soil wetness index",
            "description": "Soil water content divided by field capacity on the day before the event window starts.",
        },
        "direct_streamflow_fraction": {
            "units": "1",
            "long_name": "Direct streamflow fraction",
            "description": "Ratio of surface runoff to streamflow on the event's day, "
            "from `mrros` or the Lyne-Hollick runoff if `mrros` was not given.",
        },
    }
    for name, variable_attrs in attrs.items():
        out[name].attrs = variable_attrs
    return out.transpose(*dims)
