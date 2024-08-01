"""Utility functions for parameter estimation."""

from juliacall import convert as jl_convert
from xhydro_temp.extreme_value_analysis.julia_import import Extremes
from xhydro.extreme_value_analysis.julia_import import jl
from xhydro.extreme_value_analysis.structures.conversions import (
    py_variable_to_jl_variable, jl_vector_to_py_list,
)
from xhydro.extreme_value_analysis.structures.dataitem import Variable
import math
import numpy as np
from copy import deepcopy

__all__ = [
    "jl_variable_fit_parameters",
    "param_cint",
    "insert_covariates",
    "match_length",
    "exponentiate_logscale"
]


def jl_variable_fit_parameters(covariate_list: list[list]):
    r"""
    Transform a list of lists, representing all the covariates' data for a single parameter,
    into a julia.Vector of julia.Extremes.Variable objects.

    Parameters
    ----------
    covariate_list : list[list]
        Covariates' data for a single parameter

    Returns
    -------
    julia.Vector[julia.Extremes.Variables]
        The sequence of julia Variables to be used in non-stationary parameter estimation.

    Notes
    -----
    This function is necessary for non-stationary parameter estimation: see example at extreme_value_analysis/parameterestimation.gevfit().
    """
    py_variables = [Variable("", values) for values in covariate_list] # it is not important that variables have a name
    jl_variables = [py_variable_to_jl_variable(py_variable) for py_variable in py_variables]
    jl_vector_variables = jl_convert(jl.Vector[jl.Extremes.Variable], jl_variables)
    return jl_vector_variables


def param_cint(jl_model, bayesian: bool = False, confidence_level: float = 0.95) -> dict[str, list[float]]:
    r"""
    Returns a list of parameters and confidence intervals for a given Julia fitted model.

    Parameters
    ----------
    jl_model :
        The fitted Julia model from which parameters and confidence intervals are to be extracted.
    bayesian : bool
        If True, the function will calculate parameters and confidence intervals based on Bayesian simulations.
        Defaults to False.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
        Defaults to 0.95.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    if bayesian:
        jl_params_sims = jl_model.sim.value
        py_params_sims = [jl_vector_to_py_list(jl.vec(jl_params_sims[i, :, :])) for i in range(len(jl_params_sims[:, 0, 0]))]
        params = [sum(x) / len(py_params_sims) for x in zip(*py_params_sims)]  # each parameter is estimated to be the average over all simulations
    else:
        params = jl_vector_to_py_list(getattr(jl_model, "θ̂"))
    cint = [jl_vector_to_py_list(interval) for interval in Extremes.cint(jl_model, confidence_level)]
    cint_lower = [interval[0] for interval in cint]
    cint_upper = [interval[1] for interval in cint]
    return {"params": params, "cint_lower": cint_lower, "cint_upper": cint_upper}


def insert_covariates(param_names: list[str], covariates: list[str], param_name: str) -> list[str]:
    r"""
    Insert appropriate covariate names in the parameter names list.

    Parameters
    ----------
    param_names : list[str]
        List of parameter names in which to insert the covariate names
    covariates : list[str]
        List of covariate names to insert in the parameter names
    param_name : str
        Name of the parameter (such as "loc", "shape", "scale") after which the covariates are inserted

    Returns
    -------
    list[str]
        Updated list of parameter names with the appropriate covariates in the right place

    Examples
    --------
    >>> insert_covariates(["shape", "loc", "scale"], ["temperature", "year"], "loc")
    >>> ["shape", "loc", "loc_temperature_covariate", "loc_year_covariate", "scale"]
    """
    index = param_names.index(param_name)
    return (
        param_names[:index + 1]
        + [f'{param_name}_{covariate}_covariate' for covariate in covariates]
        + param_names[index + 1:]
    )


def match_length(py_list: list, covariates: list[list]) -> list[list]:
    r"""
    Adjusts the length of covariate data to match the fitting data by removing entries corresponding to NaNs in the fitting data.

    This function is useful when the fitting data contains NaNs and needs to be pruned. To ensure that the covariate data
    remains aligned with the fitting data, the function removes the corresponding entries from the covariate data.

    Parameters
    ----------
    py_list : list
        The fitting data which may contain NaNs.
    covariates : list[list]
        List of covariate data lists to be adjusted.

    Returns
    -------
    list[list]
        A new list of covariates with entries removed to match the length of the fitting data without NaNs.

    Examples
    --------
    >>> fitting_data = [1, 2, np.nan, 4, 5]
    >>> loc_covariate = [6, 5, 7, 8, 9]
    >>> shape_covariate = [9, 7, 6, 5, 4]
    >>> match_length(fitting_data, [loc_covariate, shape_covariate])
    >>> [[6, 5, 8, 9], [9, 7, 5, 4]]
    """
    nan_indexes = [index for index, value in enumerate(py_list) if (math.isnan(value) or np.isnan(value))]
    covariates_copy = deepcopy(covariates)
    for sublist in covariates_copy:
        for index in sorted(nan_indexes, reverse=True):
            del sublist[index]
    return covariates_copy


def exponentiate_logscale(params: np.ndarray, locationcov_data: list[list], logscalecov_data: list[list]) -> np.ndarray:
    r"""
    Exponentiate logscale parameter as well as all its covariates to obtain actual scale parameter.

    Parameters
    ----------
    params : np.ndarray
        The fitted parameters, including covariates.
    locationcov_data : list[list]
        List of covariate data lists for the location parameter.
        This is needed to keep track of the index of the logscale parameter in params.
    logscalecov_data : list[list]
        List of covariate data lists for the scale parameter.

    Returns
    -------
    np.ndarray
        Updated parameter list of logscale parameter and covariates with both having been exponentiated.
    """
    scale_param_index = 1 + len(locationcov_data)
    for index in range(scale_param_index, scale_param_index + len(logscalecov_data) + 1):
        params[index] = np.exp(params[index])
    return params

