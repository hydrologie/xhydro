"""Module to classify flood events."""

import warnings

import numpy as np
import numpy.typing as npt
import scipy.stats
import xarray as xr
from pint.errors import DimensionalityError
from xclim.core.units import convert_units_to, units2pint

from .generic import split_streamflow


__all__ = ["major_flood_events"]

FLOOD_TYPES = (
    "snowmelt",
    "mostly_snowmelt_with_some_rainfall",
    "rain_on_snow",
    "soil_water_excess_and_short_rain",
    "soil_water_excess_and_long_rain",
    "short_rain",
    "long_rain",
)


def major_flood_events(
    *,
    fldcapacity: xr.DataArray,
    mrsol: xr.DataArray,
    prra: xr.DataArray,
    rivo: xr.DataArray,
    snm: xr.DataArray,
    drainage_area: xr.DataArray,
    reference_period: tuple[int, int],
    mrros: xr.DataArray | None = None,
    max_days: int = 7,
    min_prec: float = 1,
) -> xr.Dataset:
    """
    Extract major flood events (annual maximum streamflow) and classify them.

    Years run from 1 December to 30 November: a December day counts toward
    the following year, so a winter belongs to a single year. For each year,
    the flood event is the day of maximum streamflow plus the contiguous run
    of rainy days (``prra >= min_prec``) directly before it, up to
    ``max_days`` days in total. Each event is classified by
    comparing its total snowmelt, rainfall distribution and antecedent soil
    wetness index (SWI, ``mrsol / fldcapacity``) against a per-basin soil
    moisture threshold fitted over ``reference_period``.

    Parameters
    ----------
    fldcapacity : xr.DataArray
        Soil field capacity (mm). Static (no time dependence needed).
    mrsol : xr.DataArray
        Soil water content (mm).
    prra : xr.DataArray
        Rainfall (mm).
    rivo : xr.DataArray
        Streamflow, expressed as a water depth (mm).
    snm : xr.DataArray
        Snowmelt (mm).
    drainage_area : xr.DataArray
        Drainage area, convertible to km2. Only used to scale the independence
        criterion when declustering the reference-period events feeding the
        soil moisture threshold.
    reference_period : tuple[int, int]
        First and last year (inclusive) of the period used to fit the soil
        moisture threshold. Years run December through November, so the
        period covers 1 December of ``reference_period[0] - 1`` to
        30 November of ``reference_period[1]``.
    mrros : xr.DataArray | None, default: None
        Surface runoff (mm). If None, it is derived from ``rivo`` with
        :py:func:`xhydro.indicators.split_streamflow`.
    max_days : int, default: 7
        Maximum number of days for an event.
    min_prec : float, default: 1
        Minimum daily rainfall (mm) to extend an event backward from its peak.

    Returns
    -------
    xr.Dataset
        One value per year (annual "time" axis) for each variable:

        - ``flood_type``: integer code of the event type (see ``flag_values`` /
          ``flag_meanings``); -1 where the year is incomplete or streamflow is
          all missing. A record starting in January leaves its first year
          without a December, hence incomplete.
        - ``rivo_peak``, ``rivo_peak_doy``: annual maximum streamflow and its
          calendar day of year.
        - ``event_duration``: length of the event window (days).
        - ``prra_sum``, ``prra_max``: total and maximum daily rainfall over the
          event window.
        - ``snm_sum``: total snowmelt over the event window.
        - ``swi_antecedent``: SWI on the day before the event window starts.
        - ``direct_streamflow_fraction``:
          the ratio of surface runoff to streamflow on the peak day,
          using ``mrros`` or the Lyne-Hollick runoff if ``mrros`` is None.
        - ``swi_threshold``: the fitted soil moisture threshold (no time axis);
          ``inf`` where fewer than 10 reference events exist or antecedent SWI
          and peak flow are uncorrelated, in which case the soil-water-excess
          types never trigger.

    Notes
    -----
    The classification is a modification of the decision tree of Tramblay et
    al. (2025) [1]_, applied in order: "snowmelt" if total snowmelt exceeds
    four times the total rainfall; "mostly snowmelt with some rainfall" if it
    exceeds twice the rainfall; "rain-on-snow" if it exceeds a quarter of it;
    then "short rain" if the largest rain day holds more than 75% of the
    event's rainfall, "long rain" otherwise, with a "soil water excess and"
    prefix when the antecedent SWI exceeds the soil moisture threshold.
    Modifications from [1]_: the paper's two snow classes ("snowmelt" above
    1x the rainfall, "rain and snowmelt" above 0.25x) become three with
    cutoffs at 4x, 2x and 0.25x, and its single "soil water excess" class is
    split by the short/long rain criterion. An event whose window starts on
    the record's first day has no antecedent SWI, which is treated as 0.

    The soil moisture threshold follows Tramblay et al. (2022) [2]_:
    reference-period events are the declustered days with streamflow above its
    10th percentile; if at least 10 events show a significant Spearman
    correlation (p < 0.05) between antecedent SWI and peak flow, an exponential
    curve is fitted and the threshold is the SWI at the split minimizing the
    two-segment squared error; otherwise the threshold is ``inf``. The split
    is the exact single-changepoint solution of the PELT method [3]_ used in
    [2]_; deviations from [2]_ are flagged in the inline comments.

    The SWI (``mrsol / fldcapacity``) deviates from both papers, which scale
    soil moisture into [0, 1] (by its long-term range in [2]_, by the model
    store's capacity in [1]_): with land-surface-model inputs it can exceed 1
    where soil water tops field capacity, which is harmless when comparing
    against a threshold fitted on the same index.

    References
    ----------
    .. [1] Tramblay, Y., Thirel, G., Strohmenger, L., Evin, G., Corre, L., Heraut, L., & Sauquet, E. (2025). Evolution of flood generating
       processes under climate change in France. Hydrology and Earth System Sciences, 29, 7023-7039. https://doi.org/10.5194/hess-29-7023-2025
    .. [2] Tramblay, Y., Villarini, G., Saidi, M. E., Massari, C., & Stein, L. (2022). Classification of flood-generating processes in Africa.
       Scientific Reports, 12, 18920. https://doi.org/10.1038/s41598-022-23725-5
    .. [3] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost.
       Journal of the American Statistical Association, 107(500), 1590-1598. https://doi.org/10.1080/01621459.2012.737745
    """
    variables = {name: variable for name, variable in locals().items() if isinstance(variable, xr.DataArray) and name != "drainage_area"}
    if max_days < 1:
        raise ValueError(f"`max_days` must be >= 1: is {max_days}.")
    _validate_inputs(variables, drainage_area, reference_period)

    if mrros is None:
        _, mrros = split_streamflow(rivo)

    antecedent_swi = (mrsol / fldcapacity).shift(time=1)
    # Tramblay et al. (2022) decluster with "5 + log(catchment area)" days between events: larger basins
    # integrate flow over longer times, so their events must be further apart. The paper gives neither
    # the log base nor the area unit; we read it as log10 of km2.
    min_days = 5 + xr.ufuncs.log10(convert_units_to(drainage_area, "km2"))

    threshold = _soil_moisture_threshold(
        rivo=rivo,
        prra=prra,
        antecedent_swi=antecedent_swi,
        min_days=min_days,
        reference_period=reference_period,
        max_days=max_days,
        min_prec=min_prec,
    )

    # a December flood belongs to the winter that follows it, so December counts toward the following year
    years = rivo.time.dt.year + (rivo.time.dt.month == 12)
    n_years = len(np.unique(years.values))
    outputs = xr.apply_ufunc(
        _events_kernel,
        rivo,
        prra,
        snm,
        mrros,
        antecedent_swi,
        rivo.time.dt.dayofyear,
        years,
        threshold,
        input_core_dims=[["time"]] * 7 + [[]],
        output_core_dims=[["year"]] * 9,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int16] + [np.float64] * 8,
        dask_gufunc_kwargs={"output_sizes": {"year": n_years}},
        kwargs={"max_days": max_days, "min_prec": min_prec, "n_years": n_years},
    )
    return _build_output(outputs, threshold, rivo, prra, snm, first_year=int(years[0]))


def _validate_inputs(variables: dict[str, xr.DataArray], drainage_area: xr.DataArray, reference_period: tuple[int, int]) -> None:
    """Validate units, time alignment and reference period; raises ValueError on the first violation."""
    named = {**variables, "drainage_area": drainage_area}
    if missing := [name for name, variable in named.items() if "units" not in variable.attrs]:
        raise ValueError("All variables must have units. These variables are missing them: {}.".format(", ".join(missing)))
    mm = units2pint("mm")
    if bad := {name: variable.attrs["units"] for name, variable in variables.items() if not units2pint(variable).is_compatible_with(mm, "hydro")}:
        raise ValueError('All variable units must be convertible to "mm": {}.'.format(", ".join(f"{name}: {units}" for name, units in bad.items())))
    try:
        convert_units_to(drainage_area, "km2")
    except DimensionalityError as err:
        raise ValueError(f'`drainage_area` units must be convertible to "km2": is "{drainage_area.attrs["units"]}".') from err
    if "time" in drainage_area.dims:
        raise ValueError("`drainage_area` must not have a time dimension.")

    if "time" not in variables["rivo"].dims:
        raise ValueError('`rivo` must have a "time" dimension.')
    time = variables["rivo"].time
    for name, variable in variables.items():
        if name in ("fldcapacity", "rivo"):
            continue
        if "time" not in variable.dims or not variable.time.equals(time):
            raise ValueError(f"`{name}` must share `rivo`'s time coordinate.")
    # declustering and event windows count time steps, so a step must be a day
    if xr.infer_freq(time) != "D":
        raise ValueError("All variables must have a daily time step.")

    if reference_period[0] > reference_period[1]:
        raise ValueError(f"`reference_period` must be (<start>, <end>): is {reference_period}.")
    if variables["rivo"].sel(time=slice(f"{reference_period[0] - 1}-12", f"{reference_period[1]}-11")).time.size == 0:
        raise ValueError(f"`reference_period` {reference_period} does not intersect the data's time range.")


def _soil_moisture_threshold(
    *,
    rivo: xr.DataArray,
    prra: xr.DataArray,
    antecedent_swi: xr.DataArray,
    min_days: xr.DataArray,
    reference_period: tuple[int, int],
    max_days: int,
    min_prec: float,
) -> xr.DataArray:
    """Fit the per-cell soil moisture threshold from the reference-period events, following Tramblay et al. (2022)."""
    # years run December through November, so year Y starts on 1 December of Y - 1
    reference = slice(f"{reference_period[0] - 1}-12", f"{reference_period[1]}-11")
    return xr.apply_ufunc(
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
        # deviations from Tramblay et al. (2022): they discard events with no rainfall the days before
        # the peak, which we keep because snowmelt-dominated basins would lose most of their events;
        # their event window is capped only by a dry day, while we also reuse the classification cap
        # (`max_days`) so both event definitions stay coherent
        start = _window_start(prra, peak, max_days=max_days, min_prec=min_prec)
        swi[i] = antecedent_swi[start]
        peaks[i] = np.max(q[start : peak + 1])
    finite = np.isfinite(swi)
    swi, peaks = swi[finite], peaks[finite]

    # deviation from Tramblay et al. (2022), which only gates on significance: below 10 events the
    # Spearman p-value is too coarse to mean anything
    if swi.size < 10:
        return np.inf
    with warnings.catch_warnings():
        # a constant series is a legitimate opt-out, handled through its NaN p-value below
        warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)
        p_value = scipy.stats.spearmanr(swi, peaks).pvalue
    if not p_value < 0.05:  # "not <" so a NaN p-value (constant series) also opts out
        return np.inf
    # the form a * exp(b * x) comes from Tarasova et al. (2018) via Tramblay et al. (2022), but no paper in
    # the citation chain names the estimator; least squares on the log is the simplest
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
        # the trough since the previous candidate, including the candidate day itself
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
    years: npt.NDArray[np.integer],
    threshold: float,
    *,
    max_days: int,
    min_prec: float,
    n_years: int,
) -> tuple[npt.NDArray[np.number], ...]:
    """
    Compute the annual-max flood event indicators and type for one series.

    Returns a 9-tuple of length-`n_years` arrays; a year with fewer than 360 time
    steps or all-missing streamflow gets NaN indicators and flood type -1.
    """
    flood_type = np.full(n_years, -1, dtype=np.int16)
    indicators = np.full((8, n_years), np.nan)
    peak_q, peak_doy, duration, rain_sum, rain_max, melt_sum, swi, fraction = indicators

    bounds = np.append(np.unique(years, return_index=True)[1], years.size)
    for i in range(n_years):
        year_q = q[bounds[i] : bounds[i + 1]]
        if year_q.size < 360 or np.isnan(year_q).all():
            continue
        peak = bounds[i] + int(np.nanargmax(year_q))
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
        flood_type[i] = _classify(melt_sum=melt_sum[i], rain_sum=rain_sum[i], rain_max=rain_max[i], swi=swi[i], threshold=float(threshold))
    return (flood_type, peak_q, peak_doy, duration, rain_sum, rain_max, melt_sum, swi, fraction)


def _classify(*, melt_sum: float, rain_sum: float, rain_max: float, swi: float, threshold: float) -> int:
    """
    Return the flood type code of one event; a NaN antecedent SWI is treated as 0.

    Decision tree modified from Tramblay et al. (2025); see the Notes of :py:func:`major_flood_events` for the modifications.
    """
    if melt_sum > 4 * rain_sum:
        return 0
    if melt_sum > 2 * rain_sum:
        return 1
    if melt_sum > 0.25 * rain_sum:
        return 2
    short_rain = rain_max > 0.75 * rain_sum
    if swi > threshold:
        return 3 if short_rain else 4
    return 5 if short_rain else 6


def _build_output(
    outputs: tuple[xr.DataArray, ...], threshold: xr.DataArray, rivo: xr.DataArray, prra: xr.DataArray, snm: xr.DataArray, *, first_year: int
) -> xr.Dataset:
    """Assemble the yearly output Dataset: annual time axis, threshold diagnostic and attributes."""
    names = (
        "flood_type",
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
    out["swi_threshold"] = threshold
    out = out.rename(year="time").assign_coords(
        time=xr.date_range(
            f"{first_year}-01-01",
            periods=out.sizes["year"],
            freq="YS",
            calendar=rivo.time.dt.calendar,
        )
    )

    attrs: dict[str, dict] = {
        "flood_type": {
            "long_name": "Flood type",
            "flag_values": list(range(len(FLOOD_TYPES))),
            "flag_meanings": " ".join(FLOOD_TYPES),
            "description": "Type of the annual maximum streamflow event; -1 where the year is incomplete or streamflow is all missing.",
        },
        "rivo_peak": {"units": rivo.attrs["units"], "long_name": "Annual maximum streamflow", "cell_methods": "time: maximum within years"},
        "rivo_peak_doy": {"units": "1", "long_name": "Day of year of the annual maximum streamflow"},
        "event_duration": {
            "units": "d",
            "long_name": "Flood event duration",
            "description": "Length of the contiguous rain window ending at the annual streamflow peak.",
        },
        "prra_sum": {"units": prra.attrs["units"], "long_name": "Total rainfall during the flood event", "cell_methods": "time: sum"},
        "prra_max": {"units": prra.attrs["units"], "long_name": "Maximum daily rainfall during the flood event", "cell_methods": "time: maximum"},
        "snm_sum": {"units": snm.attrs["units"], "long_name": "Total snowmelt during the flood event", "cell_methods": "time: sum"},
        "swi_antecedent": {
            "units": "1",
            "long_name": "Antecedent soil wetness index",
            "description": "Soil water content divided by field capacity on the day before the event window starts.",
        },
        "direct_streamflow_fraction": {
            "units": "1",
            "long_name": "Direct streamflow fraction",
            "description": "Ratio of surface runoff to streamflow on the peak day, from `mrros` or the Lyne-Hollick runoff if `mrros` was not given.",
        },
        "swi_threshold": {
            "units": "1",
            "long_name": "Soil moisture threshold",
            "description": "Changepoint of the exponential fit of reference-period peak flows against antecedent SWI; "
            "inf where fewer than 10 events exist or the correlation is not significant.",
        },
    }
    for name, variable_attrs in attrs.items():
        out[name].attrs = variable_attrs
    return out.transpose(*rivo.dims)
