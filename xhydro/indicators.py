"""Module to compute indicators using xclim's build_indicator_module_from_yaml."""
import xarray as xr
import xclim as xc
from xclim.core.units import rate2amount

# Special imports from xscen
from xscen import compute_indicators

__all__ = [
    "compute_indicators",
    "compute_volume",
    "get_yearly_op",
]


def compute_volume(
    da: xr.DataArray, *, out_units: str = "m3", attrs: dict = None
) -> xr.DataArray:
    """Compute the volume of water from a streamflow variable, keeping the same frequency.

    Parameters
    ----------
    da : xr.DataArray
        Streamflow variable.
    out_units : str
        Output units. Defaults to "m3".
    attrs : dict
        Attributes to add to the output variable.
        Default attributes for "long_name", "units", "cell_methods" and "description" will be added if not provided.

    Returns
    -------
    xr.DataArray
        Volume of water.
    """
    default_attrs = {
        "long_name": "Volume of water",
        "units": "m3",
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
    indexers: dict = None,
    missing: str = "skip",
    missing_options: dict = None,
    sum_method: str = "mean",
) -> xr.Dataset:
    """
    Compute yearly operations on a variable.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset containing the variable to compute the operation on.
    op: str
        Operation to compute. One of ["max", "min", "mean", "sum"].
    input_var: str
        Name of the input variable. Defaults to "streamflow".
    window: int
        Size of the rolling window. A "mean" operation is performed on the rolling window before the call to xclim.
        This parameter cannot be used with the "sum" operation.
    indexers: dict
        Dictionary of indexers for the operation. Keys are the name of the indexers (e.g. "winter", "summer", "annual") and the values are the indexers.
        Examples are {"winter": {"doy_bounds": [30, 150]}}, {"jan": {"month": [1]}} or {"annual": {}}.
        See :py:func:`xclim.indices.generic.select_time` for more information.
        If op=="sum" and sum_method=="mean", `indexers` must be either None, "doy_bounds" or "date_bounds".
    missing: str
        How to handle missing values. One of "skip", "any", "at_least_n", "pct", "wmo".
        See :py:func:`xclim.core.missing` for more information.
    missing_options: dict
        Dictionary of options for the missing values' method. See :py:func:`xclim.core.missing` for more information.
    sum_method: str
        Method to compute the sum. One of ["mean", "sum"]. Defaults to "mean", which multiplies the mean streamflow by the number of days in the indexer.
        This is done to avoid a potential issue with missing values in the variable, which would lead to an underestimation of the sum.

    Returns
    -------
    xr.Dataset
        Dataset containing the computed operations, with one variable per indexer. The name of the variable follows the pattern `{input_var}{window}_{op}_{indexer}`.

    Notes
    -----
    `freq` is currently hard-coded to "YS", since other frequencies would lead to misleading results with the frequency analysis.
    If you want to perform a frequency analysis on a different frequency, simply use multiple indexers (e.g. 1 per month) to create multiple distinct variables.

    """
    missing_options = missing_options or {}
    indexers = indexers or {"annual": {}}

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
        if op != "sum":
            ds[input_var] = (
                ds[input_var]
                .rolling(dim={"time": window}, min_periods=window, center=False)
                .mean()
            )
        else:
            raise ValueError("Cannot use a rolling window with a sum operation.")

    indicators = []
    mult = []
    for i in indexers:
        if len(indexers[i]) > 1:
            raise ValueError("Only one indexer is supported per operation.")

        identifier = f"{input_var}{window if window > 1 else ''}_{op}_{i.lower()}"
        ind = xc.core.indicator.Indicator.from_dict(
            data={
                "base": "stats",
                "input": {"da": input_var},
                "parameters": {
                    "op": op if op != "sum" else sum_method,
                    "freq": "YS",
                    "indexer": indexers[i],
                },
                "missing": missing,
                "missing_options": missing_options,
            },
            identifier=identifier,
            module="fa",
        )
        indicators.append((identifier, ind))

        # Get the multiplier for the sum
        if op == "sum" and sum_method == "mean":
            if "doy_bounds" in indexers[i]:
                mult.append(
                    indexers[i]["doy_bounds"][1] - indexers[i]["doy_bounds"][0] + 1
                )
            elif "date_bounds" in indexers[i]:
                dt_start = ds.time.where(
                    (
                        ds.time.dt.month
                        == int(indexers[i]["date_bounds"][0].split("-")[0])
                    )
                    & (
                        ds.time.dt.day
                        == int(indexers[i]["date_bounds"][0].split("-")[1])
                    ),
                    drop=True,
                ).dt.dayofyear
                dt_end = ds.time.where(
                    (
                        ds.time.dt.month
                        == int(indexers[i]["date_bounds"][1].split("-")[0])
                    )
                    & (
                        ds.time.dt.day
                        == int(indexers[i]["date_bounds"][1].split("-")[1])
                    ),
                    drop=True,
                ).dt.dayofyear
                dt = xr.align(dt_start, dt_end, join="override")[1] - dt_start + 1
                mult.append(dt)
            elif indexers[i] == {}:
                dt_start = ds.time.groupby("time.year").min().dt.dayofyear
                dt_end = ds.time.groupby("time.year").max().dt.dayofyear
                dt = xr.align(dt_start, dt_end, join="override")[1] - dt_start + 1
                dt = dt.rename({"year": "time"})
                mult.append(dt)
            else:
                raise NotImplementedError(
                    "Only doy_bounds, date_bounds, or no indexers are currently supported for a sum operation with sum_method='mean'."
                )

    # Compute the indicators
    out = compute_indicators(ds, indicators=indicators).popitem()[1]

    # Multiply by the number of days for the sum method
    if op == "sum" and sum_method == "mean":
        for i, m in zip(indexers, mult):
            identifier = f"{input_var}{window if window > 1 else ''}_{op}_{i.lower()}"
            if isinstance(m, xr.DataArray):
                out[identifier] = out[identifier] * xr.align(out, m, join="override")[1]
            else:
                out[identifier] = out[identifier] * m

    return out
