"""Utility functions for parameter estimation."""

try:
    import math
    import warnings
    from copy import deepcopy

    import numpy as np
    import xarray as xr
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
    "_create_nan_mask",
    "_recover_nan",
    "exponentiate_logscale",
    "insert_covariates",
    "jl_variable_fit_parameters",
    "param_cint",
    "remove_nan",
    "return_level_cint",
]


def jl_variable_fit_parameters(covariate_list: list[list]):
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
    jl_model,
    bayesian: bool = False,
    confidence_level: float = 0.95,
) -> list:
    r"""
    Return a list of parameters and confidence intervals for a given Julia fitted model.

    Parameters
    ----------
    jl_model : Julia.Extremes.AbstractExtremeValueModel
        The fitted Julia model from which parameters and confidence intervals are to be extracted.
    bayesian : bool
        If True, the function will calculate parameters and confidence intervals based on Bayesian simulations.
        Defaults to False.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
        Defaults to 0.95.

    Returns
    -------
    list
        A list containing NumPy arrays for the estimated parameters, and upper bounds for the confidence interval
        of each parameter.
    """
    if bayesian:
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
        warnings.warn(f"There was an error in computing confidence interval.")


def return_level_cint(
    jl_model,
    confidence_level: float = 0.95,
    return_period: float = 100,
    pareto: bool = False,
    threshold_pareto=None,
    nobs_pareto=None,
    nobsperblock_pareto=None,
    bayesian: bool = False,
) -> dict[str, list[float]]:
    r"""
    Return a list of retun level and confidence intervals for a given Julia fitted model.

    Parameters
    ----------
    jl_model : Julia.Extremes.AbstractExtremeValueModel
        The fitted Julia model from which parameters and confidence intervals are to be extracted.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
        Defaults to 0.95.
    return_period : float
        Return period used to compute the return level.
    pareto : bool
        If True, the return level parameters and confidence intervals will be based on the Pareto distribution.
        Defaults to False.
    threshold_pareto : float
        Threshold.
    nobs_pareto : int,
        Number of total observation.
    nobsperblock_pareto : int,
        Number of observation per block.
    bayesian : bool
        If True, the function will calculate parameters and confidence intervals based on Bayesian simulations.
        Defaults to False.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    try:
        if pareto:
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

        if bayesian:
            py_return_level_m = np.array(jl_vector_to_py_list(jl_return_level.value))
            shp = jl_return_level.value.shape
            py_return_level = np.mean(np.reshape(py_return_level_m, shp[::-1]), axis=1)
        else:
            py_return_level = np.array(jl_vector_to_py_list(jl_return_level.value))

    except JuliaError:
        warnings.warn(f"There was an error in computing return level.")
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
        warnings.warn(
            f"There was an error in computing confidence interval for the return level."
        )

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
    locationcov_data: list[list],
    logscalecov_data: list[list],
    pareto: bool = False,
) -> np.ndarray:
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
    pareto : bool
        Boolean value indicating whether we are dealing with the parameters of a pareto distribution.

    Returns
    -------
    np.ndarray
        Updated parameter list with logscale parameter and its covariates having been exponentiated.
    """
    scale_param_index = 1 + len(locationcov_data)

    if pareto:
        scale_param_index = 0

    for index in range(
        scale_param_index, scale_param_index + len(logscalecov_data) + 1
    ):
        params[index] = np.exp(params[index])
    return params


def _create_nan_mask(*nested_lists) -> list:
    """
    Create a mask indicating NaN positions across multiple nested lists.

    Parameters
    ----------
    nested_lists:
        Any number of nested lists (lists of lists).

    Returns
    -------
        A single list mask with True where NaNs are present for all the nested lists.

    Notes
    -----
    This function is useful when the fitting data and covariates contains NaNs and needs to be pruned.
    To ensure that the covariate data remains aligned with the fitting data, the function returns a mask
    with True values where ther is at leat one NaN.

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


def _recover_nan(mask, lists: list[list[float]]) -> list[list[float]]:
    """
    Recover the original length of lists by filling NaN in masked positions.

    Parameters
    ----------
    mask:
        A masked array indicating positions of valid data.
    lists:
        A list of arrays to be recovered.

    Returns
    -------
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
