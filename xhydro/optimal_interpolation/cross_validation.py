"""Perform the cross-validation for the optimal interpolation."""

from .functions import optimal_interpolation as opt


def execute(
    start_date,
    end_date,
    files,
    write_file,
    ratio_var_bg=0.15,
    percentiles=[0.25, 0.50, 0.75, 1.00],
    iterations=10,
    parallelize=False,
):
    """
    Run the interpolation algorithm for cross-validation.

    Parameters
    ----------
    start_date : datetime
        Start date of the analysis.
    end_date : datetime
        End date of the analysis.
    files : list(str)
        List of files path for getting flows and watersheds info.
    ratio_var_bg : float, optional
        Ratio for background variance (default is 0.15).
    percentiles : list(float), optional
        List of percentiles to analyze (default is [0.25, 0.50, 0.75, 1.00]).
    iterations: int, optional
        Number of iterations for the interpolation (default is 10).
    parallelize : bool, optional
        Execute the profiler in parallel or in series (default is False).

    Returns
    -------
    list
        A list containing the results of the interpolated percentiles flow
    """
    # Run the code
    # TODO: Replace inputs to file with args dict constructed upstream and pass along.
    args = {
        "start_date": start_date,
        "end_date": end_date,
        "files": files,
        "ratio": ratio_var_bg,
        "percentiles": percentiles,
    }

    time_range = (end_date - start_date).days + 1

    results = opt.execute_interpolation(
        start_date,
        end_date,
        time_range,
        files,
        ratio_var_bg,
        percentiles,
        iterations,
        parallelize=parallelize,
        write_file=write_file,
    )

    return results
