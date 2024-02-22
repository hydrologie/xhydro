"""Perform the cross-validation for the optimal interpolation."""

import datetime as dt
from typing import Optional

from .functions import optimal_interpolation as opt


def execute(
    start_date: dt.datetime,
    end_date: dt.datetime,
    files: list[str],
    write_file,
    ratio_var_bg: float = 0.15,
    percentiles: Optional[list[float]] = None,
    iterations: int = 10,
    parallelize: bool = False,
):
    """Run the interpolation algorithm for cross-validation.

    Parameters
    ----------
    start_date : datetime.datetime
        Start date of the analysis.
    end_date : datetime.datetime
        End date of the analysis.
    files : list(str)
        List of files path for getting flows and watersheds info.
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
        start_date,
        end_date,
        time_range=(end_date - start_date).days + 1,
        files=files,
        ratio_var_bg=ratio_var_bg,
        percentiles=percentiles,
        iterations=iterations,
        parallelize=parallelize,
        write_file=write_file,
    )

    return results
