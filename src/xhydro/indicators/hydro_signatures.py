# Created on Mon Sep 22 2025
# @author: Ève Larose (user e-larose)
"""
Aggregated hydrological signature package for xhydro, for model evaluation.

This signature package is useful for watershed comparisons.
For temporal analysis see xclim.indices._hydrology library
"""
# Import packages

import numpy as np
import xarray
from scipy import signal, stats
from xclim.core.units import convert_units_to
from xscen.utils import standardize_period
from xhydro.indicators import generic
from xhydro.utils import health_checks


__all__ = [
    "elasticity_index",
    "flow_duration_curve_slope",
    "total_runoff_ratio",
]


def elasticity_index(
    q: xarray.DataArray,
    pr: xarray.DataArray,
    periods: list[str] | list[list[str]] | None = None,
    missing: str = "skip",
    missing_options: dict | None = None,
) -> xarray.DataArray:
    """
    Elasticity index.

    Compute the median of yearly streamflow elasticity index for given catchments,
    where elasticity (εₚ) is defined as the relative change in streamflow (ΔQ/Q)
    divided by the relative change in precipitation (ΔP/P).

    Parameters
    ----------
    q : xarray.DataArray
        Daily discharge data.
    pr : xarray.DataArray
        Daily precipitation data.
    periods : list of str or list of list of str, optional
        Either [start, end] or list of [start, end] of periods to be considered.
        If multiple periods are given, the output will have a `horizon` dimension.
        If None, all data is used.
    missing : str
        How to handle missing values. One of "skip", "any", "at_least_n", "pct", "wmo".
        See :py:func:`xclim.core.missing` for more information.
    missing_options : dict, optional
        Dictionary of options for the missing values' method. See :py:func:`xclim.core.missing` for more information.

    Returns
    -------
    xarray.DataArray
        Nonparametric estimator for streamflow elasticity index (dimensionless).

    Notes
    -----
    A value of εp greater than 1 indicates that streamflow is highly sensitive to precipitation changes,
    meaning a 1% change in precipitation will lead to a greater than 1% change in streamflow.
    A value less than 1 suggests a less sensitive relationship.
    It is recommended to use yearly frequency in order to have more robust elasticity_index

    References
    ----------
    Sankarasubramanian, A., Vogel, R. M., & Limbrunner, J. F. (2001). Climate elasticity of streamflow
    in the United States. Water Resources Research, 37(6), 1771–1781. https://doi.org/10.1029/2000WR900330
    """
    # if missing is None:
    #     missing = {"missing_pct": {"freq": "YE", "tolerance": 0.3}} #checks for missing values on a yearly basis

    ds_q = q.to_dataset(name="q")
    health_checks(ds_q, freq="D")  # all other health checks on a daily basis

    ds_pr = pr.to_dataset(name="pr")
    ds_pr["pr"].attrs["units"] = ds_pr["pr"].attrs.get("units", "undefined")

    periods = (
        standardize_periods(periods, multiple=True)
        if periods is not None
        else [[str(int(ds_q.time.dt.year.min())), str(int(ds_q.time.dt.year.max()))]]
    )
    out = []
    for period in periods:
        ds_subset_q = ds_q.sel(time=slice(period[0], period[1]))
        ds_subset_p = ds_pr.sel(time=slice(period[0], period[1]))

        # Annual mean from raw data
        q_annual = generic.get_yearly_op(ds_subset_q, op="mean", timeargs={"annual": {}}, missing=missing, missing_options=missing_options)
        p_annual = generic.get_yearly_op(
            ds_subset_p, op="mean", timeargs={"annual": {}}, missing=missing, missing_options=missing_options, input_var="pr"
        )

        # Year-to-year changes
        delta_q = q_annual.diff(dim="time")
        delta_p = p_annual.diff(dim="time")

        # Avoid division by zero
        # TODO replace with jitter_under_thresh
        epsilon = 1e-10

        # Relative changes
        rel_delta_p = delta_p / (p_annual + epsilon)
        rel_delta_q = delta_q / (q_annual + epsilon)

        # Compute yearly streamflow elasticity (not mathematically robust values)
        yearly_elasticity = rel_delta_q["q_mean_annual"] / (rel_delta_p["pr_mean_annual"] + epsilon)

        out.append(yearly_elasticity)

    out_combined = xarray.concat(out, dim="time")

    # Median over selected periods
    ei = out_combined.median(dim="time")

    return xarray.DataArray(ei.values, attrs={"units": "", "long_name": "Elasticity index"})


def flow_duration_curve_slope(
    q: xarray.DataArray,
    periods: list[str] | list[list[str]] | None = None,
    missing=None,
) -> xarray.DataArray:
    """
    Calculate the slope of the flow duration curve mid-section between the 33% and 66% exceedance probabilities.

    Aggregated analysis : Single value as a long-term benchmark.

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data, expected to have a discharge unit.
    periods : list of str or list of list of str, optional
        Either [start, end] or list of [start, end] of periods to be considered.
        If multiple periods are given, the output will have a `horizon` dimension.
        If None, all data is used.
    missing : str
        Checks for xclim.core.missing to perform. Default is a tolerance of 30% of missing values.

    Returns
    -------
    xarray.DataArray
        Slope of the FDC between the 33% and 66% exceedance probabilities, unitless.

    Notes
    -----
    Single Returned numeric values are positive.

    High slope value :
    Steep slopes of the midsegment FDC are typically a characteristic behavior for watersheds having ‘flashy’ responses
    (for example due to small soil storage capacity and hence larger percentage of overland flow).

    Lower slope value :
    Flatter slopes of the midsegment FDC are associated with watersheds having slower and
    more sustained groundwater low response.

    References
    ----------
    Vogel, R. M., & Fennessey, N. M. (1994). Flow-duration curves. I: New interpretation and confidence intervals.
    Journal of Water Resources Planning and Management, 120(4), 485-504.
    DOI:10.1061/(ASCE)0733-9496(1994)120:4(485)

    Yilmaz, K. K., Gupta, H. V., & Wagener, T. (2008). A process‐based diagnostic approach to model evaluation:
    Application to the NWS distributed hydrologic model. Water resources research, 44(9).
    DOI:10.1029/2007WR006716
    """
    if missing is None:
        missing = {"missing_pct": {"freq": "YE", "tolerance": 0.1}}

    ds_q = q.to_dataset(name="q")
    health_checks(ds_q, missing=missing, raise_on="all")
    periods = (
        standardize_periods(periods, multiple=True)
        if periods is not None
        else [[str(int(ds_q.time.dt.year.min())), str(int(ds_q.time.dt.year.max()))]]
    )
    out = []
    for period in periods:
        subset_q = q.sel(time=slice(period[0], period[1]))
        # Calculate the 33rd and 66th percentiles directly across the 'time' dimension
        q33 = subset_q.quantile(0.33, dim="time", skipna=True)
        q66 = subset_q.quantile(0.66, dim="time", skipna=True)

        # Calculate the natural logarithm of the quantiles
        ln_q33 = np.log(q33)
        ln_q66 = np.log(q66)

        # Calculate the slope (unitless)
        slope = (ln_q33 - ln_q66) / (0.33 - 0.66)
        slope.attrs["units"] = " "
        slope.attrs["long_name"] = "Slope of FDC between 33% and 66% exceedance probabilities"
        out.append(slope)

    out_combined = xarray.concat(out, dim="time")
    return out_combined


def total_runoff_ratio(
    q: xarray.DataArray,
    drainage_area: xarray.DataArray,
    pr: xarray.DataArray,
    missing=None,
) -> xarray.DataArray:
    """
    Total runoff ratio.

    Compute the ratio of streamflow measured at a stream gauge station to the total precipitation over the watershed.
    Also known as the runoff coefficient, it is higher in watersheds with steep slopes,
    impervious surfaces or high evapotranspiration.

    Parameters
    ----------
    q : xarray.DataArray
        Streamflow in [discharge] units, will be converted to [m3/s].
    drainage_area : xarray.DataArray
        Watershed area [area] units, will be converted to in [km²].
    pr : xarray.DataArray
        Mean daily Precipitation [precipitation] units, will be converted to [mm/hr].
    missing : str
        Checks for xclim.core.missing to perform. Default is a tolerance of 30% of missing values.

    Returns
    -------
    xarray.DataArray
        Single value rainfall-runoff ratio (RRR) as long-term benchmark.

    Notes
    -----
    - Total Runoff ratio values are comparable to Runoff coefficients,
    - Values near 0 mean most precipitation infiltrates watershed soil or is lost to evapotranspiration.
    - Values near 1 mean most precipitation leaves the watershed as runoff;
      possible causes are impervious surfaces from urban sprawl, thin soils, steep slopes, etc.
    - Long-term averages are typically ≤ 1.

    References
    ----------
    HydroBM https://hydrobm.readthedocs.io/en/latest/usage.html#benchmarks
    """
    # TODO compute only for valid years
    if missing is None:
        missing = {"missing_pct": {"freq": "YE", "tolerance": 0.1}}

    ds_q = q.to_dataset(name="q")
    health_checks(ds_q, missing=missing, raise_on="all")

    q = convert_units_to(q, "mm3/hr")
    drainage_area = convert_units_to(drainage_area, "mm2")
    if xc.units.units2pint(pr).dimensionality.get("[time]") == 0:
        pr = xc.units.amount2rate(pr)
    pr = convert_units_to(pr, "mm h-1", context="hydro")

    runoff = q / drainage_area
    total_rr = runoff.sum() / pr.sum()
    total_rr.attrs["units"] = ""
    total_rr.attrs["long_name"] = "Total Rainfall-Runoff Ratio"

    return total_rr


def hurst_exp(
    q: xarray.DataArray,
    selected_low_frequecy: int = 0.01,
    missing=None,
) -> xarray.DataArray:
    """
    Hurst Exponent.

    Compute the Hurst Exponent (H) of time-series obtained from the slope of
    the power spectral density (estimated by periodogram) of the streamflow time series at near-zero frequency.

    Parameters
    ----------
    q : xarray.DataArray
        Streamflow in [discharge] units.
    missing : str
        Checks for xclim.core.missing to perform. Default is a tolerance of 30% of missing values.

    Returns
    -------
    xarray.DataArray
        Single value Hurst Exponent (H).

    Notes
    -----
    - Hurst Exponent serves as a statistical health check for observed and simulated streamflows;
    - H>0.5: persistence (long-term memory, common in hydrology) acceptable range: 0,5 to 1.
    - H<0.5: noise and anti-persistence.

    References
    ----------
    Gupta, A., Hantush, M. M., Govindaraju, R. S., & Beven, K. (2024). Evaluation of hydrological models
    at gauged and ungauged basins using machine learning-based limits-of-acceptability and hydrological signatures.
    Journal of Hydrology, 641, 131774. https://doi.org/10.1016/j.jhydrol.2024.131774.

    Koutsoyiannis, D., and A. Montanari (2007), Statistical analysis of hydroclimatic time series:
    Uncertainty and insights, Water Resour. Res., 43, W05429, doi:10.1029/2006WR005592.
    """
    if missing is None:
        missing = {"missing_pct": {"freq": "YE", "tolerance": 0.1}}

    ds_q = q.to_dataset(name="q")
    health_checks(ds_q, missing=missing, raise_on="all")

    q_interp = q.interpolate_na("time")

    # Fill leading/trailing NaNs using nearest valid neighbor
    q_filled = q_interp.ffill("time").bfill("time")
    q_filled = q_filled.where(np.isfinite(q_filled), np.nan)

    # Final fill for remaining single points (if any)
    q_filled = q_filled.interpolate_na("time")
    q_filled = q_filled.astype(float)

    # Detrend to remove strong low-freq bias
    q_detrended = xarray.DataArray(
        signal_detrended := signal.detrend(q_filled.values),
        coords=q_filled.coords,
        dims=q_filled.dims,
    )
    # Compute periodogram using correct fs
    f, pxx = signal.periodogram(signal_detrended, fs=1, scaling="density")  # freq default fs = 1day

    mask = (f > 0) & (f < f.max() * selected_low_frequecy)  # select near zero freq

    if mask.sum() < 20:
        mask = (f > 0) & (f < f.max() * 0.02)  # check for at least 20 frequency bins

    freqs_low = f[mask]
    pxx_low = pxx[mask]

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(freqs_low), np.log10(pxx_low))
    beta = -slope  # slope is negative, so Beta = -slope
    h = (beta + 1) / 2
    return h
