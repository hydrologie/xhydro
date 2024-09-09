"""Module to compute indicators using xclim's build_indicator_module_from_yaml."""

import warnings
from typing import Optional

import xarray as xr
import xclim as xc
import xscen as xs
from xclim.core.units import rate2amount

# Special imports from xscen
from xscen import compute_indicators

__all__ = [
    "compute_indicators",
    "compute_volume",
    "get_yearly_op",
]


def compute_volume(
    da: xr.DataArray, *, out_units: str = "m3", attrs: dict | None = None
) -> xr.DataArray:
    """Compute the volume of water from a streamflow variable, keeping the same frequency.

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


def get_yearly_op(
    ds,
    op,
    *,
    input_var: str = "streamflow",
    window: int = 1,
    timeargs: dict | None = None,
    missing: str = "skip",
    missing_options: dict | None = None,
    interpolate_na: bool = False,
) -> xr.Dataset:
    """Compute yearly operations on a variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable to compute the operation on.
    op : str
        Operation to compute. One of ["max", "min", "mean", "sum"].
    input_var : str
        Name of the input variable. Defaults to "streamflow".
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
        raise ValueError(
            f"Operation {op} is not supported. Please use one of ['max', 'min', 'mean', 'sum']."
        )
    if op == "sum":
        if window > 1:
            raise ValueError("Cannot use a rolling window with a sum operation.")
        if interpolate_na:
            ds[input_var] = ds[input_var].interpolate_na(dim="time", method="linear")

    # Add the variable to xclim to avoid raising an error
    if input_var not in xc.core.utils.VARIABLES:
        attrs = {
            "long_name": None,
            "units": None,
            "cell_methods": None,
            "description": None,
        }
        attrs.update(ds[input_var].attrs)
        attrs["canonical_units"] = attrs["units"]
        attrs.pop("units")
        xc.core.utils.VARIABLES[input_var] = attrs

    # FIXME: This should be handled by xclim once it supports rolling stats (Issue #1480)
    # rolling window
    if window > 1:
        ds[input_var] = (
            ds[input_var]
            .rolling(dim={"time": window}, min_periods=window, center=False)
            .mean()
        )

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
            raise ValueError(
                f"Frequency {freq} is not supported. Please use a yearly frequency."
            )
        indexer = {k: v for k, v in timeargs[i].items() if k != "freq"}
        if len(indexer) > 1:
            raise ValueError("Only one indexer is supported per operation.")

        # Manage the frequency
        if (
            "season" in indexer.keys()
            and "DJF" in indexer["season"]
            and freq != "YS-DEC"
        ):
            warnings.warn(
                "The frequency is not YS-DEC, but the season indexer includes DJF. "
                "This will lead to misleading results."
            )
        elif (
            "doy_bounds" in indexer.keys()
            and indexer["doy_bounds"][0] >= indexer["doy_bounds"][1]
        ) or (
            "date_bounds" in indexer.keys()
            and int(indexer["date_bounds"][0].split("-")[0])
            >= int(indexer["date_bounds"][1].split("-")[0])
        ):
            if "doy_bounds" in indexer.keys():
                # transform doy to a date to find the month
                ts = xr.cftime_range(
                    start="2000-01-01",
                    periods=366,
                    freq="D",
                    calendar=ds.time.dt.calendar,
                )
                month_start = ts[indexer["doy_bounds"][0] - 1].month
                month_end = ts[indexer["doy_bounds"][1] - 1].month
            else:
                month_start = int(indexer["date_bounds"][0].split("-")[0])
                month_end = int(indexer["date_bounds"][1].split("-")[0])
            if month_end == month_start:
                warnings.warn(
                    "The bounds wrap around the year, but the month is the same between the both of them. "
                    "This is not supported and will lead to wrong results."
                )
            if freq == "YS" or (month_start != month_labels.index(freq.split("-")[1])):
                warnings.warn(
                    f"The frequency is {freq}, but the bounds are between months {month_start} and {month_end}. "
                    f"You should use 'YS-{month_labels[month_start - 1]}' as the frequency."
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
    out = xs.clean_up(out, common_attrs_only=ind_dict)

    return out
