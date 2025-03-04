"""Utility functions for parameter estimation."""

import warnings
from typing import Any

import numpy as np

try:
    from juliacall import JuliaError
    from juliacall import convert as jl_convert

    from xhydro.extreme_value_analysis.julia_import import Extremes, jl
    from xhydro.extreme_value_analysis.structures.conversions import (
        jl_vector_to_py_list,
        py_variable_to_jl_variable,
    )
    from xhydro.extreme_value_analysis.structures.dataitem import Variable
except ImportError as e:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    raise ImportError(JULIA_WARNING) from e

__all__ = [
    "create_nan_mask",
    "exponentiate_logscale",
    "insert_covariates",
    "jl_variable_fit_parameters",
    "param_cint",
    "recover_nan",
    "remove_nan",
    "return_level_cint",
    "return_nan",
]

METHOD_NAMES = {
    "ML": "maximum likelihood",
    "PWM": "probability weighted moments",
    "BAYES": "bayesian",
}

DIST_NAMES = {
    "genextreme": "<class 'scipy.stats._continuous_distns.genextreme_gen'>",
    "gumbel_r": "<class 'scipy.stats._continuous_distns.gumbel_r_gen'>",
    "genpareto": "<class 'scipy.stats._continuous_distns.genpareto_gen'>",
}


def jl_variable_fit_parameters(covariate_list: list[list]) -> Any:
    r"""
    Transform a list of lists into a julia.Vector of julia.Extremes.Variable objects.

    Parameters
    ----------
    covariate_list : list[list]
        Covariates' data for a single parameter.

    Returns
    -------
    julia.Vector[julia.Extremes.Variables]
        The sequence of julia Variables to be used in non-stationary parameter estimation.

    Notes
    -----
    This function is necessary for non-stationary parameter estimation:
    see example at extreme_value_analysis/parameterestimation.gevfit().
    """
    py_variables = [
        Variable("", values) for values in covariate_list
    ]  # it is not important that variables have a name
    jl_variables = [
        py_variable_to_jl_variable(py_variable) for py_variable in py_variables
    ]
    jl_vector_variables = jl_convert(jl.Vector[jl.Extremes.Variable], jl_variables)
    return jl_vector_variables


def param_cint(
    jl_model: Any,
    method: str,
    confidence_level: float = 0.95,
) -> list[np.ndarray]:
    r"""
    Return a list of parameters and confidence intervals for a given Julia fitted model.

    Parameters
    ----------
    jl_model : Julia.Extremes.AbstractExtremeValueModel
        Fitted Julia model.
    method : {"ML", "PWM", "BAYES"}
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
        Defaults to 0.95.

    Returns
    -------
    list[np.ndarray]
        A list containing NumPy arrays for the estimated parameters, and upper bounds for the confidence interval
        of each parameter.
    """
    if method == "BAYES":
        jl_params_sims = jl_model.sim.value

        py_params_sims = [
            jl_vector_to_py_list(jl.vec(jl_params_sims[i, :, :]))
            for i in range(jl_params_sims.shape[0])
        ]
        params = np.mean(
            np.stack(py_params_sims), axis=0
        )  # each parameter is estimated to be the average over all simulations
    else:
        params = np.array(jl_vector_to_py_list(getattr(jl_model, "θ̂")))

    try:
        jl_cint = Extremes.cint(jl_model, confidence_level)
        cint = np.stack([jl_vector_to_py_list(interval) for interval in jl_cint])
        cint_lower = cint[:, 0]
        cint_upper = cint[:, 1]

        return [params, cint_lower, cint_upper]

    except JuliaError:
        warnings.warn(
            f"There was an error in computing confidence interval.", UserWarning
        )

        return [
            params,
            np.array([np.nan] * len(params)),
            np.array([np.nan] * len(params)),
        ]


def return_level_cint(
    jl_model,
    dist: str,
    method: str,
    confidence_level: float = 0.95,
    return_period: float = 100,
    threshold_pareto: float | None = None,
    nobs_pareto: int | None = None,
    nobsperblock_pareto: int | None = None,
) -> dict[str, list[float]]:
    r"""
    Return a list of return levels and confidence intervals for a given Julia fitted model.

    Parameters
    ----------
    jl_model : Julia.Extremes.AbstractExtremeValueModel
        Fitted Julia model.
    dist : str or rv_continuous
        Distribution, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    method : {"ML", "PWM", "BAYES"}
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    confidence_level : float
        The confidence level (between 0 and 1) for the confidence interval of the return level.
        Defaults to 0.95.
    return_period : float
        Return period used to compute the return level.
    threshold_pareto : float
        Threshold. Required when `dist=genpareto`.
    nobs_pareto : int
        Number of total observation. Required when `dist=genpareto`.
    nobsperblock_pareto : int
        Number of observation per block. Required when `dist=genpareto`.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    try:
        if dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            if (
                threshold_pareto is None
                or nobs_pareto is None
                or nobsperblock_pareto is None
            ):
                raise ValueError(
                    "'threshold_pareto', 'nobs_pareto', and 'nobsperblock_pareto' must be defined when jl_model use a 'genpareto'."
                )
            else:
                jl_return_level = Extremes.returnlevel(
                    jl_model,
                    threshold_pareto,
                    nobs_pareto,
                    nobsperblock_pareto,
                    return_period,
                )
        else:
            jl_return_level = Extremes.returnlevel(jl_model, return_period)

        if method == "BAYES":
            py_return_level_m = np.array(jl_vector_to_py_list(jl_return_level.value))
            shp = jl_return_level.value.shape
            py_return_level = np.mean(np.reshape(py_return_level_m, shp[::-1]), axis=1)
        else:
            py_return_level = np.array(jl_vector_to_py_list(jl_return_level.value))

    except JuliaError:
        raise ValueError(
            f"There was an error in computing confidence interval for the return level."
        )

    try:
        jl_cint = Extremes.cint(jl_return_level, confidence_level)
        cint = np.stack([jl_vector_to_py_list(interval) for interval in jl_cint])
        cint_lower = cint[:, 0]
        cint_upper = cint[:, 1]

        return [py_return_level, cint_lower, cint_upper]

    except JuliaError:
        raise ValueError(
            f"There was an error in computing confidence interval for the return level."
        )


def insert_covariates(
    param_names: list[str], covariates: list[str], param_name: str
) -> list[str]:
    r"""
    Insert appropriate covariate names in the parameter names list.

    Parameters
    ----------
    param_names : list[str]
        List of parameter names in which to insert the covariate names.
    covariates : list[str]
        List of covariate names to insert in the parameter names.
    param_name : str
        Name of the parameter (such as "loc", "shape", "scale") after which the covariates are inserted.

    Returns
    -------
    list[str]
        Updated list of parameter names with the appropriate covariates in the right place.

    Examples
    --------
    >>> insert_covariates(["shape", "loc", "scale"], ["temperature", "year"], "loc")
    >>> ["shape", "loc", "loc_temperature_covariate", "loc_year_covariate", "scale"]
    """
    index = param_names.index(param_name)
    return (
        param_names[: index + 1]
        + [f"{param_name}_{covariate}_covariate" for covariate in covariates]
        + param_names[index + 1 :]
    )


def remove_nan(mask: np.array, covariates: list[list]) -> list[list]:
    r"""
    Remove entries from a list of lists based on a boolean mask.

    Parameters
    ----------
    mask : np.array
        Array containing the True and False values.
    covariates : list[list]
        List of lists from which values will be removed according
        to the mask.

    Returns
    -------
    list[list]
        A new list containing the list without the masked values.
    """
    covariate_mask = [np.array(sublist)[~mask].tolist() for sublist in covariates]

    return covariate_mask


def exponentiate_logscale(
    params: np.ndarray,
    dist: str,
    n_loccov: int,
    n_scalecov: int,
) -> np.ndarray:
    r"""
    Exponentiate the logscale parameter along with covariates to obtain actual scale parameter.

    Parameters
    ----------
    params : np.ndarray
        The fitted parameters, including covariates.
    dist : str or rv_continuous
        The univariate distribution to fit, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    n_loccov : int
        Number of covariates for the location parameter.
    n_scalecov : int
        Number of covariates for the scale parameter.

    Returns
    -------
    np.ndarray
        Updated list with the exponential of the logscale parameter and covariates.
    """
    if dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
        scale_param_index = 0
    else:
        scale_param_index = 1 + n_loccov

    shape_param_index = scale_param_index + n_scalecov + 1

    for index in range(scale_param_index, shape_param_index):
        params[index] = np.exp(params[index])

    return params


def change_sign_param(param_list: list[np.array], pos: int, n: int) -> list[np.array]:
    """
    Change the sign of given parameter.

    Parameters
    ----------
    param_list : list[np.array]
        A list containing three numpy arrays of the parameters and their confidence intervals
        [params, lim_inf, lim_sup].
    pos : int
        The starting position for the parameters to change.
    n : int
        The number of parameters to change starting from the given position.

    Returns
    -------
    list of np.array
        The modified param_list with signs changed for the specified parameters.
    """
    param_list_c = [arr.copy() for arr in param_list]
    param_list_c[0][pos:] = -param_list[0][pos : pos + n]
    param_list_c[1][pos:] = -param_list[2][pos : pos + n]
    param_list_c[2][pos:] = -param_list[1][pos : pos + n]

    return param_list_c


def create_nan_mask(*nested_lists) -> list:
    r"""
    Create a mask indicating NaN positions across multiple nested lists.

    Parameters
    ----------
    \*nested_lists : tuple of list of lists # noqa: RST213
        Any number of nested lists (lists of lists).

    Returns
    -------
    mask :
        A single list mask with True where NaNs are present for all the nested lists.
        Example: np.array([True, False, True, True, False]).

    Notes
    -----
    This function is useful when the fitting data and covariates contains NaNs and needs to be pruned.
    To ensure that the covariate data remains aligned with the fitting data, the function returns a mask
    with True values where there is at least one NaN either in the data or in the covariates.

    Examples
    --------
    >>> fitting_data = [1, 2, np.nan, 4, 5]
    >>> loc_covariate = [6, 5, 7, 8, 9]
    >>> shape_covariate = [9, 7, 6, 5, np.nan]
    >>> match_length(fitting_data, [loc_covariate, shape_covariate])
    >>> [False, False, True, False, True]
    """
    arrays = [np.array(lst, dtype=float) for lst in nested_lists if lst]
    stack = np.vstack(arrays)
    mask = np.any(np.isnan(stack), axis=0)

    return mask


def recover_nan(
    mask: np.ndarray | list[bool], lists: np.ndarray | list[list[float]]
) -> list[list[float]]:
    """
    Recover the original length of lists by filling NaN in masked positions.

    Parameters
    ----------
    mask : np.ndarray
        A masked array indicating positions of valid data.
        Example: np.array([True, False, True, True, False]).
    lists : np.ndarray or list[list[float]]
        A list of arrays to be recovered.

    Returns
    -------
    list[list[float]]
        A list of lists with NaNs filled in the original masked positions.
    """
    reco_list = []
    for lst in lists:
        if np.all(np.isnan(lst)):
            reco_list.append(lst)
        else:
            recovered = np.full(mask.shape, np.nan, dtype=lst.dtype)
            recovered[~mask] = lst

            reco_list.append(recovered)

    return reco_list


def return_nan(length: int) -> list[np.ndarray]:
    """
    Return a list of three lists, each containing NaN values of the specified length.

    Parameters
    ----------
    length : int
        The length of each list.

    Returns
    -------
    list[np.ndarray]
        A list containing three lists, each of which contains NaN values of the given length.
    """
    return [np.array([np.nan] * length) for _ in range(3)]
