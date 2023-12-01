"""Module to compute indicators using xclim's build_indicator_module_from_yaml."""
import warnings
from typing import Optional

import numpy as np
import pymannkendall
import scipy.stats as stats
import xarray as xr
import xclim as xc
import xscen as xs
from scipy.stats import mannwhitneyu, norm
from xclim.core.units import rate2amount

# Special imports from xscen
from xscen import compute_indicators

__all__ = [
    "compute_indicators",
    "compute_volume",
    "get_yearly_op",
    "mannkendall",
    "pval_mannwhitneyu",
    "pval_wald_wolfowitz",
]


def compute_volume(
    da: xr.DataArray, *, out_units: str = "m3", attrs: Optional[dict] = None
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
    timeargs: Optional[dict] = None,
    missing: str = "skip",
    missing_options: Optional[dict] = None,
    interpolate_na: bool = False,
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
    timeargs: dict, optional
        Dictionary of time arguments for the operation.
        Keys are the name of the period that will be added to the results (e.g. "winter", "summer", "annual").
        Values are up to two dictionaries, with both being optional.
        The first is {'freq': str}, where str is a frequency supported by xarray (e.g. "YS", "AS-JAN", "AS-DEC").
        It needs to be a yearly frequency. Defaults to "AS-JAN".
        The second is an indexer as supported by :py:func:`xclim.core.calendar.select_time`. Defaults to {}, which means the whole year.
        See :py:func:`xclim.core.calendar.select_time` for more information.
        Examples: {"winter": {"freq": "AS-DEC", "date_bounds": ['12-01', '02-28']}}, {"jan": {"freq": "YS", "month": 1}}, {"annual": {}}.
    missing: str
        How to handle missing values. One of "skip", "any", "at_least_n", "pct", "wmo".
        See :py:func:`xclim.core.missing` for more information.
    missing_options: dict, optional
        Dictionary of options for the missing values' method. See :py:func:`xclim.core.missing` for more information.
    interpolate_na: bool
        Whether to interpolate missing values before computing the operation. Only used with the "sum" operation. Defaults to False.

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
        freq = timeargs[i].get("freq", "AS-JAN")
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
            and freq != "AS-DEC"
        ):
            warnings.warn(
                "The frequency is not AS-DEC, but the season indexer includes DJF. This will lead to misleading results."
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
                    f"You should use 'AS-{month_labels[month_start - 1]}' as the frequency."
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


def mannkendall(da, outputs: list = ["h", "p"], test: str = "original_test"):
    """Calculate Mann-Kendall test statistics and associated values for a given dataarray.

    Parameters
    ----------
    da : xarray DataArray
        Input time series
    outputs : list of str, optional
        Names of output variables to compute, by default ["h", "p"]
        Possible values include 'trend', 'h', 'p', 'z', 'S', 'var_s', 'trend_type', 'seasonal', and 'seasonal_slope'.
        Refer to the documentation of the specific Mann-Kendall test variant for available outputs.
    test : str, optional
        The Mann-Kendall test variant to be applied, by default "original_test"

    Returns
    -------
    xarray Dataset
        Dataset containing output variables specified in outputs
    """

    def _pval_mannkendall_ufunc(da, test, outputs):
        da = da[~np.isnan(da)]
        pmk = getattr(pymannkendall, test)
        mk = pmk(da)
        return tuple([getattr(mk, out) for out in outputs])

    try:
        return xr.concat(
            xr.apply_ufunc(
                _pval_mannkendall_ufunc,
                da,
                test,
                outputs,
                input_core_dims=[["time"], [], ["dim0"]],
                output_core_dims=np.empty((len(outputs), 0)).tolist(),
                vectorize=True,
            ),
            dim="mk",
        ).assign_coords(mk=outputs)
    except TypeError:
        return xr.apply_ufunc(
            _pval_mannkendall_ufunc,
            da,
            test,
            outputs,
            input_core_dims=[["time"], [], ["dim0"]],
            output_core_dims=np.empty((len(outputs), 0)).tolist(),
            output_dtypes=[tuple],
            vectorize=True,
        ).assign_coords(mk=outputs)


def pval_mannwhitneyu(da_in, buffer=5):
    """
    Apply Mann-Whitney U test for change detection.

    Computes mean before and after each timestep, as well as p-value from
    Mann-Whitney U test. Buffer is used to avoid comparing only a few values.

    Parameters
    ----------
    da_in : xarray.DataArray
        Input data array with time dimension.

    buffer : int
        Number of timesteps to exclude at beginning/end.

    Returns
    -------
    xr.DataArray
        Data array with mannwhitneyu dimension containing mean_before, mean_after
        and p_value variables.

    """

    def pval_mannwhitneyu_ufunc(da_in, buffer=5, test="two-sided"):
        """
        Calculate Mann-Whitney U test p-values over a moving window.

        Parameters
        ----------
        da_in : xarray DataArray
            Input data array
        buffer : int, optional
            Size of moving window, by default 5

        Returns
        -------
        mean_before : xarray DataArray
            Mean values before each index
        mean_after : xarray DataArray
            Mean values after each index
        p_values : xarray DataArray
            p-values for difference in distribution before and
            after each index

        Handles NaNs by ignoring them in calculations.

        Performs a two-sided test by default. Can use 'greater' or
        'less' for one-sided test.
        """
        da_no_nan = da_in[~np.isnan(da_in)]
        mean_before = np.empty(len(da_in)) * np.nan
        mean_after = np.empty(len(da_in)) * np.nan
        p_values = np.empty(len(da_in)) * np.nan

        temp_pval = np.empty(len(da_no_nan)) * np.nan
        temp_mean_before = np.empty(len(da_no_nan)) * np.nan
        temp_mean_after = np.empty(len(da_no_nan)) * np.nan

        for i in range(buffer, len(da_no_nan) - buffer):
            part_1 = da_no_nan[:i]
            part_2 = da_no_nan[i:]
            mwu = mannwhitneyu(x=part_1, y=part_2, alternative=test)
            temp_pval[i] = mwu.pvalue
            temp_mean_before[i] = np.nanmean(da_no_nan[:i])
            temp_mean_after[i] = np.nanmean(da_no_nan[i:])

        mean_before[~np.isnan(da_in)] = temp_mean_before
        mean_after[~np.isnan(da_in)] = temp_mean_after
        p_values[~np.isnan(da_in)] = temp_pval

        return (mean_before, mean_after, p_values)

    return xr.concat(
        xr.apply_ufunc(
            pval_mannwhitneyu_ufunc,
            da_in,
            buffer,
            input_core_dims=[["time"], []],
            output_core_dims=[["time"], ["time"], ["time"]],
            output_dtypes=[tuple, tuple, tuple],
            vectorize=True,
        ),
        dim="mannwhitneyu",
    ).assign_coords(mannwhitneyu=["mean_before", "mean_after", "p_values"])


def pval_wald_wolfowitz(da: xr.DataArray):
    """Calculate the p-value for the Wald-Wolfowitz runs test on a timeseries.

    Applies pval_wald_wolfowitz_ufunc over the 'time' dimension of
    the input DataArray.

    Parameters
    ----------
    da : xarray DataArray
        Streamflow variable.

    Returns
    -------
    xr.DataArray
        DataArray containing p-values for Wald-Wolfowitz test applied to each
        timeseries in da.
    """

    def pval_wald_wolfowitz_ufunc(q):
        """
        Perform a two-sided Wald-Wolfowitz independence test.

        Parameters
        ----------
        q: series (array-like):
            Consecutive numeric observation series (n-dimensional vector).

        Returns
        -------
            float: The observed test threshold (p-value).

        Example
        -------
            >>> x = normrnd(0, 1, 50, 1)
            >>> p = pval_wald_wolfowitz_ufunc(x)

        Notes
        -----
        Data must be consecutive as this test relies on serial correlation (first-order autocorrelation).
        Therefore, if there are gaps in the series (e.g., if some data points have been
        disabled), the results may be inaccurate. In such a case, we recommend using the longest
        continuous subseries.
        The HYFRAN software does not consider this issue and performs the test even if the sample has gaps.

        References
        ----------
        Bobée and Ashkar (1991). The Gamma Family and Derived Distributions Applied in Hydrology
        Wald and Wolfowitz (1943). An exact test for randomness in the non-parametric case based on serial correlation. Ann. Math. Statist., 14, 378-388
        Distributions Applied in Hydrology
        """
        # Remove NaN values
        q = np.array(q)
        qo = q[~np.isnan(q)]
        qo = np.array(qo)

        """ Length of the series """
        lseries = len(qo)

        """ Calculate the Wald-Wolfowitz runs statistic R """
        r = np.nansum(qo[: len(qo) - 1] * qo[1 : len(qo)]) + qo[0] * qo[-1]

        """ Calculate the non-central moments of the sample of orders 1 to 4 """
        m = []
        s = []
        for i in range(1, 5):
            m.append(np.nanmean(qo**i))
            s.append(lseries * np.nanmean(qo**i))

        """ Calculate the mean and variance of the R statistic """
        rmoy = (s[0] ** 2 - s[1]) / (lseries - 1)
        term1 = (s[1] ** 2 - s[3]) / (lseries - 1)
        term2 = rmoy**2
        term3 = (
            s[0] ** 4 - 4 * (s[0] ** 2) * s[1] + 4 * s[0] * s[2] + s[1] ** 2 - 2 * s[3]
        ) / ((lseries - 1) * (lseries - 2))
        rvar = term1 - term2 + term3

        """ Calculate the centered Wald-Wolfowitz statistic """
        rcenter = abs((r - rmoy) / (np.sqrt(rvar)))

        """ Calculate the observed threshold (p-value) """
        pvalue = 2 * (1 - norm.cdf(abs(rcenter)))

        return pvalue

    return xr.apply_ufunc(
        pval_wald_wolfowitz_ufunc, da, input_core_dims=[["time"]], vectorize=True
    )
