"""Utility functions for parameter estimation."""

from juliacall import convert as jl_convert

from xhydro.extreme_value_analysis.julia_import import jl
from xhydro.extreme_value_analysis.structures.conversions import (
    py_variable_to_jl_variable,
)
from xhydro.extreme_value_analysis.structures.dataitem import Variable


def jl_variable_fit_parameters(params: list[list[Variable]]) -> tuple:
    r"""
    Transform a list of lists of Variables into a tuple of julia.Vectors of julia.Extremes.Variables.

    Parameters
    ----------
    params : list[list[Variable]]
        List of lists of Variables to be transformed into a tuple of julia.Vectors of julia.Extremes.Variables.

    Returns
    -------
    tuple(julia.Vector(julia.Extremes.Variables))
        The sequence of julia Variables to be used in non-stationary parameter estimation.

    Notes
    -----
    This function is necessary for non-stationary parameter estimation: see example at extreme_value_analysis/parameterestimation.gevfit().
    """
    # python list of lists of julia.Extremes Variables
    variables = [
        [py_variable_to_jl_variable(variable) for variable in params[i]]
        for i in range(len(params))
    ]

    # python tuple of julia vectors of julia.Extremes Variables
    jl_params = tuple(
        jl_convert(jl.Vector[jl.Extremes.Variable], variables[i])
        for i in range(len(variables))
    )
    return jl_params


# FIXME: not really used right now, delete soon?
def values_above_threshold(values: list, threshold: float) -> list:
    r"""
    Return a list containing the values above the specified threshold.

    The threshold is a proportion (between 0 and 1) of the total number of values.
    This function calculates the number of values corresponding to this threshold and
    returns the top values based on sorting in descending order.

    Parameters
    ----------
    values : list
        A list of numerical values to be filtered.

    threshold : float
        A float value between 0 and 1 representing the proportion of values to retain.

    Returns
    -------
        list
            A list containing the values above the specified threshold.

    Notes
    -----
        values = [1, 3, 5, 7, 9]
        threshold = 0.4
        values_above_threshold(values, threshold)
        [9, 7]
    """
    n = len(values)
    values_above_threshold_count = max(1, int(n * threshold))
    sorted_values = sorted(values, reverse=True)
    top_values = sorted_values[:values_above_threshold_count]
    return top_values
