"""Perform the cross-validation for the optimal interpolation."""

from typing import Optional

import xarray as xr

from xhydro.optimal_interpolation import optimal_interpolation_fun as opt

__all__ = ["execute"]


def execute(
    flow_obs: xr.Dataset,
    flow_sim: xr.Dataset,
    station_correspondence: xr.Dataset,
    crossvalidation_stations: list,
    write_file: str,
    ratio_var_bg: float = 0.15,
    percentiles: Optional[list[float]] = None,
    iterations: int = 10,
    parallelize: bool = False,
):
    """Run the interpolation algorithm for cross-validation.

    Parameters
    ----------
    flow_obs : xr.Dataset
        Streamflow and catchment properties dataset for observed data.
    flow_sim : xr.Dataset
        Streamflow and catchment properties dataset for simulated data.
    station_correspondence: xr.Dataset
        Matching between the tag in the HYDROTEL simulated files and the observed station number for the obs dataset.
    crossvalidation_stations: list
        Observed hydrometric dataset stations to be used in the cross-validation step.
    write_file : str
        Name of the NetCDF file to be created.
    ratio_var_bg : float
        Ratio for background variance (default is 0.15).
    percentiles : list(float), optional
        List of percentiles to analyze (default is [0.25, 0.50, 0.75, 1.00]).
    iterations : int
        Number of iterations for the interpolation (default is 10).
    parallelize : bool
        Execute the profiler in parallel or in series (default is False).

    Returns
    -------
    list
        The results of the interpolated percentiles flow.
    """
    if percentiles is None:
        percentiles = [0.25, 0.50, 0.75, 1.00]

    results = opt.execute_interpolation(
        flow_obs=flow_obs,
        flow_sim=flow_sim,
        station_correspondence=station_correspondence,
        crossvalidation_stations=crossvalidation_stations,
        ratio_var_bg=ratio_var_bg,
        percentiles=percentiles,
        iterations=iterations,
        parallelize=parallelize,
        write_file=write_file,
    )

    return results
