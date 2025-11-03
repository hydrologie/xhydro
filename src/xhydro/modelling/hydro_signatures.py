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
from xclim.core.units import convert_units_to


__all__ = [
    "elasticity_index",
    "flow_duration_curve_slope",
    "total_runoff_ratio",
]


def elasticity_index(q: xarray.DataArray, pr: xarray.DataArray, freq: str = "YS") -> xarray.DataArray:
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
    freq : str
        Resampling frequency (e.g., 'YS' for year starting in Jan).

    Returns
    -------
    xarray.DataArray
        Nonparametric estimator for streamflow elasticity index (dimensionless).

    Notes
    -----
    A value of εp greater than 1 indicates that streamflow is highly sensitive to precipitation changes,
    meaning a 1% change in precipitation will lead to a greater than 1% change in streamflow.
    A value less than 1 suggests a less sensitive relationship.

    References
    ----------
    Sankarasubramanian, A., Vogel, R. M., & Limbrunner, J. F. (2001). Climate elasticity of streamflow
    in the United States. Water Resources Research, 37(6), 1771–1781. https://doi.org/10.1029/2000WR900330
    """
    p_annual = pr.resample(time=freq).mean()
    q_annual = q.resample(time=freq).mean()

    p_mean = p_annual.mean(dim="time")
    q_mean = q_annual.mean(dim="time")

    # Year-to-year changes
    delta_p = p_annual.diff(dim="time")
    delta_q = q_annual.diff(dim="time")

    # Avoid division by zero
    epsilon = 1e-6

    # Relative changes
    rel_delta_p = delta_p / (p_mean + epsilon)
    rel_delta_q = delta_q / (q_mean + epsilon)

    # Compute yearly streamflow elasticity (not mathematically robust values)
    yearly_elasticity = rel_delta_q / (rel_delta_p + epsilon)

    # Compute single value using median (more robust value)
    elasticity_index = yearly_elasticity.median(dim="time")
    elasticity_index.attrs["units"] = ""
    return elasticity_index


# @declare_units(q="[discharge]")
def flow_duration_curve_slope(q: xarray.DataArray) -> xarray.DataArray:
    """
    Calculate the slope of the flow duration curve mid-section between the 33% and 66% exceedance probabilities.

    Aggregated analysis : Single value as a long-term benchmark.

    Parameters
    ----------
    q : xarray.DataArray
        Daily streamflow data, expected to have a discharge unit.

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
    # Calculate the 33rd and 66th percentiles directly across the 'time' dimension
    q33 = q.quantile(0.33, dim="time", skipna=True)
    q66 = q.quantile(0.66, dim="time", skipna=True)

    # Calculate the natural logarithm of the quantiles
    ln_q33 = np.log(q33)
    ln_q66 = np.log(q66)

    # Calculate the slope (unitless)
    slope = (ln_q33 - ln_q66) / (0.33 - 0.66)
    slope.attrs["units"] = " "
    slope.attrs["long_name"] = "Slope of FDC between 33% and 66% exceedance probabilities"
    return slope


def total_runoff_ratio(q: xarray.DataArray, a: xarray.DataArray, pr: xarray.DataArray) -> xarray.DataArray:
    """
    Total runoff ratio.

    Compute the ratio of streamflow measured at a stream gauge station to the total precipitation over the watershed.
    Also known as the runoff coefficient, it is higher in watersheds with steep slopes,
    impervious surfaces or high evapotranspiration.

    Parameters
    ----------
    q : xarray.DataArray
        Streamflow in [discharge] units, will be converted to [m3/s].
    a : xarray.DataArray
        Watershed area [area] units, will be converted to in [km²].
    pr : xarray.DataArray
        Mean daily Precipitation [precipitation] units, will be converted to [mm/hr].

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
    q = convert_units_to(q, "m3/s")
    a = convert_units_to(a, "km2")
    pr = convert_units_to(pr, "mm/hr")

    runoff = q * 3.6 / a  # unit conversion for runoff in mm/h : 3.6 [s/h * km2/m2]
    total_rr = runoff.sum() / pr.sum()
    total_rr.attrs["units"] = ""
    total_rr.attrs["long_name"] = "Total Rainfall-Runoff Ratio"

    return total_rr
