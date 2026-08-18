"""Module to compute indicators using xclim's build_indicator_module_from_yaml."""

import warnings

import numpy as np
import numpy.typing as npt
import scipy.signal
import xarray as xr
import xclim as xc
from xclim.core.units import rate2amount, units2pint

# Special imports from xscen
from xscen import compute_indicators
from xscen.utils import clean_up


__all__ = [
    "compute_indicators",
    "compute_volume",
    "get_yearly_op",
    "split_streamflow",
]


def compute_volume(da: xr.DataArray, *, out_units: str = "m3", attrs: dict | None = None) -> xr.DataArray:
    """
    Compute the volume of water from a streamflow variable, keeping the same frequency.

    Parameters
    ----------
    da : xr.DataArray
        Streamflow variable.
    out_units : str
        Output units. Defaults to "m3".
    attrs : dict, optional
        Attributes to add to the output variable.
        Default attributes for "long_name", "units", "cell_methods" and "description" will be added if not provided.

    Returns
    -------
    xr.DataArray
        Volume of water.
    """
    default_attrs = {
        "long_name": "Volume of water",
        "cell_methods": "time: sum",
        "description": "Volume of water",
    }
    attrs = attrs or {}
    # Add default attributes
    for k, v in default_attrs.items():
        attrs.setdefault(k, v)

    out = rate2amount(da, out_units=out_units)
    out.attrs.update(attrs)

    return out


def get_yearly_op(  # noqa: C901
    ds,
    op,
    *,
    input_var: str = "q",
    window: int = 1,
    timeargs: dict | None = None,
    missing: str = "skip",
    missing_options: dict | None = None,
    interpolate_na: bool = False,
) -> xr.Dataset:
    """
    Compute yearly operations on a variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable to compute the operation on.
    op : str
        Operation to compute. One of ["max", "min", "mean", "sum"].
    input_var : str
        Name of the input variable. Defaults to "q".
    window : int
        Size of the rolling window. A "mean" operation is performed on the rolling window before the call to xclim.
        This parameter cannot be used with the "sum" operation.
    timeargs : dict, optional
        Dictionary of time arguments for the operation.
        Keys are the name of the period that will be added to the results (e.g. "winter", "summer", "annual").
        Values are up to two dictionaries, with both being optional.
        The first is {'freq': str}, where str is a frequency supported by xarray (e.g. "YS", "YS-JAN", "YS-DEC").
        It needs to be a yearly frequency. Defaults to "YS-JAN".
        The second is an indexer as supported by :py:func:`xclim.core.calendar.select_time`.
        Defaults to {}, which means the whole year.
        See :py:func:`xclim.core.calendar.select_time` for more information.
        Examples: {"winter": {"freq": "YS-DEC", "date_bounds": ["12-01", "02-28"]}}, {"jan": {"freq": "YS", "month": 1}}, {"annual": {}}.
    missing : str
        How to handle missing values. One of "skip", "any", "at_least_n", "pct", "wmo".
        See :py:func:`xclim.core.missing` for more information.
    missing_options : dict, optional
        Dictionary of options for the missing values' method. See :py:func:`xclim.core.missing` for more information.
    interpolate_na : bool
        Whether to interpolate missing values before computing the operation. Only used with the "sum" operation.
        Defaults to False.

    Returns
    -------
    xr.Dataset
        Dataset containing the computed operations, with one variable per indexer.
        The name of the variable follows the pattern `{input_var}{window}_{op}_{indexer}`.

    Notes
    -----
    If you want to perform a frequency analysis on a frequency that is finer than annual, simply use multiple timeargs
    (e.g. 1 per month) to create multiple distinct variables.
    """
    missing_options = missing_options or {}
    timeargs = timeargs or {"annual": {}}

    if op not in ["max", "min", "mean", "sum"]:
        raise ValueError(f"Operation {op} is not supported. Please use one of ['max', 'min', 'mean', 'sum'].")
    if op == "sum":
        if window > 1:
            raise ValueError("Cannot use a rolling window with a sum operation.")
        if interpolate_na:
            ds[input_var] = ds[input_var].interpolate_na(dim="time", method="linear")

    # Add the variable to xclim to avoid raising an error
    if input_var not in xc.core.VARIABLES:
        attrs = {
            "long_name": None,
            "units": None,
            "cell_methods": None,
            "description": None,
        }
        attrs.update(ds[input_var].attrs)
        attrs["canonical_units"] = attrs["units"]
        attrs.pop("units")
        xc.core.VARIABLES[input_var] = attrs

    # FIXME: This should be handled by xclim once it supports rolling stats (Issue #1480)
    # rolling window
    if window > 1:
        ds[input_var] = ds[input_var].rolling(dim={"time": window}, min_periods=window, center=False).mean()

    indicators = []
    month_labels = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]
    for i in timeargs:
        freq = timeargs[i].get("freq", "YS-JAN")
        if not xc.core.calendar.compare_offsets(freq, "==", "YS"):
            raise ValueError(f"Frequency {freq} is not supported. Please use a yearly frequency.")
        indexer = {k: v for k, v in timeargs[i].items() if k != "freq"}
        if len(indexer) > 1:
            raise ValueError("Only one indexer is supported per operation.")

        # Manage the frequency
        if "season" in indexer.keys() and "DJF" in indexer["season"] and freq != "YS-DEC":
            warnings.warn("The frequency is not YS-DEC, but the season indexer includes DJF. This will lead to misleading results.", stacklevel=2)
        elif ("doy_bounds" in indexer.keys() and indexer["doy_bounds"][0] >= indexer["doy_bounds"][1]) or (
            "date_bounds" in indexer.keys() and int(indexer["date_bounds"][0].split("-")[0]) >= int(indexer["date_bounds"][1].split("-")[0])
        ):
            if "doy_bounds" in indexer.keys():
                # transform doy to a date to find the month
                ts = xr.date_range(
                    start="2000-01-01",
                    periods=366,
                    freq="D",
                    calendar=ds.time.dt.calendar,
                    use_cftime=True,
                )
                month_start = ts[indexer["doy_bounds"][0] - 1].month
                month_end = ts[indexer["doy_bounds"][1] - 1].month
            else:
                month_start = int(indexer["date_bounds"][0].split("-")[0])
                month_end = int(indexer["date_bounds"][1].split("-")[0])
            if month_end == month_start:
                warnings.warn(
                    "The bounds wrap around the year, but the month is the same between the both of them. "
                    "This is not supported and will lead to wrong results.",
                    stacklevel=2,
                )
            if freq == "YS" or (month_start != month_labels.index(freq.split("-")[1])):
                warnings.warn(
                    f"The frequency is {freq}, but the bounds are between months {month_start} and {month_end}. "
                    f"You should use 'YS-{month_labels[month_start - 1]}' as the frequency.",
                    stacklevel=2,
                )

        identifier = f"{input_var}{window if window > 1 else ''}_{op}_{i.lower()}"
        ind = xc.core.indicator.Indicator.from_dict(
            data={
                "base": "stats",
                "input": {"da": input_var},
                "parameters": {
                    "op": op if op != "sum" else "integral",
                    "indexer": indexer,
                    "freq": freq,
                },
                "missing": missing,
                "missing_options": missing_options,
            },
            identifier=identifier,
            module="fa",
        )
        indicators.append((identifier, ind))

    # Compute the indicators
    ind_dict = compute_indicators(ds, indicators=indicators)

    # Combine all the indicators into one dataset
    out = xr.merge(
        [
            da.assign_coords(
                time=xr.date_range(
                    da.time[0].dt.strftime("%Y-01-01").item(),
                    periods=da.time.size,
                    calendar=da.time.dt.calendar,
                    freq="YS",
                )
            )
            for da in ind_dict.values()
        ]
    )
    out = clean_up(out, common_attrs_only=ind_dict)

    return out


def split_streamflow(
    da: xr.DataArray,
    *,
    k: float = 0.925,
    n_passes: int = 3,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Split the streamflow into baseflow and runoff using the Lyne-Hollick algorithm.

    Parameters
    ----------
    da : xr.DataArray
        Streamflow variable.
    k : float, default: 0.925
        Recursive filter parameter. Must be between 0 and 1.
    n_passes : int, default: 3
        Number of passes of the algorithm to do, alternating forward and
        backward. Must be greater than 1.

    Returns
    -------
    xr.DataArray
        Baseflow variable, named "q_base" if the streamflow is a discharge and
        "mrrob" if it is a water depth.
    xr.DataArray
        Runoff variable, named "q_runoff" if the streamflow is a discharge and
        "mrros" if it is a water depth.
        
    References
    ----------
    Lyne, V. D. and Hollick, M.: Stochastic time-variable rainfall
    runoff modelling, Hydrology and Water Resources Symposium, in: Institute of
    engineers Australia national conference, Barton, Australia, 79, 89–93,
    1979.
    """
    if not 0.0 < k < 1.0:
        raise ValueError(f"`k` must be in (0, 1): is {k}.")
    if n_passes < 1:
        raise ValueError(f"`n_passes` must be >= 1: is {n_passes}.")
    if "time" not in da.dims:
        raise ValueError(f'`da` must have a "time" dim: has {da.dims}.')
    if "units" not in da.attrs:
        raise ValueError("`da` must have a `units` attribute, which decides how the outputs are named.")

    units = units2pint(da)
    is_discharge = units.is_compatible_with(units2pint("m3 s-1"))
    if not is_discharge and not any(units.is_compatible_with(units2pint(u), "hydro") for u in ("mm", "mm h-1")):
        raise ValueError(f'`da` must be a discharge ("m3 s-1") or a water depth ("mm", "mm h-1"): is "{da.attrs["units"]}".')

    c = (1 + k) / 2

    def _filter(total_flow: npt.NDArray) -> npt.NDArray:
        # q[t] = k q[t-1] + c [Q[t] - Q[t-1]] which is a first order IIR filter
        # zi is set so q[0] = 0
        b = np.array([c, -c], dtype=total_flow.dtype)
        a = np.array([1.0, -k], dtype=total_flow.dtype)
        baseflow = total_flow
        for i in range(n_passes):
            if i % 2 == 1:  # backward pass
                baseflow = baseflow[..., ::-1]
            runoff, _ = scipy.signal.lfilter(b, a, baseflow, axis=-1, zi=-b[0] * baseflow[..., :1])
            baseflow = np.clip(baseflow - runoff, 0.0, baseflow)
            if i % 2 == 1:  # backward pass
                baseflow = baseflow[..., ::-1]
        return baseflow

    baseflow = xr.apply_ufunc(
        _filter,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[da.dtype],
        keep_attrs=True,
    ).transpose(*da.dims)

    runoff = da - baseflow

    # The input's own identity is overwritten below, not propagated: each output is a fraction
    # of the streamflow, so the inherited name and standard name would be false on both.
    inherited = dict(da.attrs)

    filtered_with = f"the Lyne-Hollick filter (k={k}, n_passes={n_passes})"

    if is_discharge:
        # not standard names
        base_name, runoff_name = "q_base", "q_runoff"
        base_attrs = {
            "long_name": "Baseflow",
            "standard_name": "outgoing_water_volume_transport_along_river_channel_due_to_baseflow",
        }
        runoff_attrs = {
            "long_name": "Direct runoff",
            "standard_name": "outgoing_water_volume_transport_along_river_channel_due_to_surface_runoff",
        }
    else:
        base_name, runoff_name = "mrrob", "mrros"
        base_attrs = {"long_name": "Subsurface runoff", "standard_name": "subsurface_runoff_flux"}
        runoff_attrs = {"long_name": "Surface runoff", "standard_name": "surface_runoff_flux"}

    baseflow = baseflow.rename(base_name)
    baseflow.attrs = {
        **inherited,
        **base_attrs,
        "description": f"Baseflow component of the streamflow, separated with {filtered_with}.",
    }
    runoff = runoff.rename(runoff_name)
    runoff.attrs = {
        **inherited,
        **runoff_attrs,
        "description": f"Direct runoff component of the streamflow, obtained by subtracting the baseflow separated with {filtered_with}.",
    }

    return baseflow, runoff
