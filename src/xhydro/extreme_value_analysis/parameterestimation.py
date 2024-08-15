"""Parameter estimation functions for the extreme value analysis module."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.stats
import xarray as xr
from juliacall import JuliaError
from xclim.core.formatting import prefix_attrs, update_history
from xclim.indices.stats import get_dist

try:
    from xhydro.extreme_value_analysis import Extremes, jl
    from xhydro.extreme_value_analysis.structures.conversions import (
        py_list_to_jl_vector,
    )
    from xhydro.extreme_value_analysis.structures.util import (
        CovariateIndex,
        exponentiate_logscale,
        insert_covariates,
        jl_variable_fit_parameters,
        match_length,
        param_cint,
    )
except (ImportError, ModuleNotFoundError) as e:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    raise ImportError(JULIA_WARNING) from e


warnings.simplefilter("always", UserWarning)
__all__ = ["fit"]

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

COVARIATE_INDEX = CovariateIndex()


# Maximum likelihood estimation
def gevfit(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    confidence_level: float = 0.95,
) -> dict[str, list[float]]:
    r"""
    Fit an array to a Genextreme distribution using the maximum likelihood algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[list]
        List of data lists to be used as covariates for the location parameter.
    logscalecov : list[list]
        List of data lists to be used as covariates for the logscale parameter.
    shapecov : list[list]
        List of data lists to be used as covariates for the shape parameter.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = (
        jl_variable_fit_parameters(locationcov),
        jl_variable_fit_parameters(logscalecov),
        jl_variable_fit_parameters(shapecov),
    )
    nparams = 3 + len(locationcov) + len(logscalecov) + len(shapecov)
    try:
        jl_model = Extremes.gevfit(
            jl_y,
            locationcov=jl_locationcov,
            logscalecov=jl_logscalecov,
            shapecov=jl_shapecov,
        )
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a genextreme distribution. "
            "Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


def gumbelfit(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    confidence_level: float = 0.95,
) -> dict[str, list[float]]:
    r"""
    Fit an array to a Gumbel distribution using the maximum likelihood algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[list]
        List of data lists to be used as covariates for the location parameter.
    logscalecov : list[list]
        List of data lists to be used as covariates for the logscale parameter.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters(
        locationcov
    ), jl_variable_fit_parameters(logscalecov)
    nparams = 2 + len(locationcov) + len(logscalecov)
    try:
        jl_model = Extremes.gumbelfit(
            jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov
        )
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a gumbel_r distribution. Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


def gpfit(
    y: list[float],
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    confidence_level: float = 0.95,
) -> dict[str, list[float]]:
    r"""
    Fit an array to a Pareto distribution using the maximum likelihood algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    logscalecov : list[list]
        List of data lists to be used as covariates for the logscale parameter.
    shapecov : list[list]
        List of data lists to be used as covariates for the shape parameter.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(
        logscalecov
    ), jl_variable_fit_parameters(shapecov)
    nparams = 2 + len(logscalecov) + len(shapecov)
    try:
        jl_model = Extremes.gpfit(
            jl_y, logscalecov=jl_logscalecov, shapecov=jl_shapecov
        )
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except JuliaError:
        warnings.warn(
            f"There was an error in fitting the data to a genpareto distribution."
            f"Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


# Probability weighted moment estimation
def gevfitpwm(y: list[float], confidence_level: float = 0.95) -> dict[str, list[float]]:
    r"""
    Fit an array to a Genextreme distribution using the probability weighted moment algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    nparams = 3
    try:
        jl_model = Extremes.gevfitpwm(jl_y)
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a genextreme distribution. "
            "Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


def gumbelfitpwm(
    y: list[float], confidence_level: float = 0.95
) -> dict[str, list[float]]:
    r"""
    Fit an array to a Gumbel distribution using the probability weighted moment algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    nparams = 2
    try:
        jl_model = Extremes.gumbelfitpwm(jl_y)
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a gumbel_r distribution. Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


def gpfitpwm(y: list[float], confidence_level: float = 0.95) -> dict[str, list[float]]:
    r"""
    Fit an array to a Pareto distribution using the probability weighted moment algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    nparams = 2
    try:
        jl_model = Extremes.gpfitpwm(jl_y)
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a genpareto distribution. Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


# Bayesian estimation
def gevfitbayes(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, list[float]]:
    r"""
    Fit an array to a Genextreme distribution using bayesian inference from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[list]
        List of data lists to be used as covariates for the location parameter.
    logscalecov : list[list]
        List of data lists to be used as covariates for the logscale parameter.
    shapecov : list[list]
        List of data lists to be used as covariates for the shape parameter.
    niter : int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = (
        jl_variable_fit_parameters(locationcov),
        jl_variable_fit_parameters(logscalecov),
        jl_variable_fit_parameters(shapecov),
    )
    nparams = 3 + len(locationcov) + len(logscalecov) + len(shapecov)
    try:
        jl_model = Extremes.gevfitbayes(
            jl_y,
            locationcov=jl_locationcov,
            logscalecov=jl_logscalecov,
            shapecov=jl_shapecov,
            niter=niter,
            warmup=warmup,
        )
        return param_cint(
            jl_model, nparams, bayesian=True, confidence_level=confidence_level
        )
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a genextreme distribution. Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


def gumbelfitbayes(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, list[float]]:
    r"""
    Fit an array to a Gumbel distribution using bayesian inference from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[list]
        List of data lists to be used as covariates for the location parameter.
    logscalecov : list[list]
        List of data lists to be used as covariates for the logscale parameter.
    niter : int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters(
        locationcov
    ), jl_variable_fit_parameters(logscalecov)
    nparams = 2 + len(locationcov) + len(logscalecov)
    try:
        jl_model = Extremes.gumbelfitbayes(
            jl_y,
            locationcov=jl_locationcov,
            logscalecov=jl_logscalecov,
            niter=niter,
            warmup=warmup,
        )
        return param_cint(
            jl_model, nparams, bayesian=True, confidence_level=confidence_level
        )
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a gumbel distribution. Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


def gpfitbayes(
    y: list[float],
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, list[float]]:
    r"""
    Fit an array to a Pareto distribution using bayesian inference from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    logscalecov : list[list]
        List of data lists to be used as covariates for the logscale parameter.
    shapecov : list[list]
        List of data lists to be used as covariates for the shape parameter.
    niter : int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level : float
        The confidence level for the confidence interval of each parameter.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(
        logscalecov
    ), jl_variable_fit_parameters(shapecov)
    nparams = 2 + len(logscalecov) + len(shapecov)
    try:
        jl_model = Extremes.gpfitbayes(
            jl_y,
            logscalecov=jl_logscalecov,
            shapecov=jl_shapecov,
            niter=niter,
            warmup=warmup,
        )
        return param_cint(
            jl_model, nparams, bayesian=True, confidence_level=confidence_level
        )
    except JuliaError:
        warnings.warn(
            "There was an error in fitting the data to a genpareto distribution. Returned parameters are numpy.nan."
        )
        return {
            "params": [np.nan for _ in range(nparams)],
            "cint_lower": [np.nan for _ in range(nparams)],
            "cint_upper": [np.nan for _ in range(nparams)],
        }


def fit(
    ds: xr.Dataset,
    locationcov: list[str] = [],
    logscalecov: list[str] = [],
    shapecov: list[str] = [],
    dist: str | scipy.stats.rv_continuous = "genextreme",
    method: str = "ML",
    dim: str = "time",
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
    distributed: bool = False,
) -> xr.Dataset:
    r"""Fit an array to a univariate distribution along the time dimension.

    Parameters
    ----------
    ds : xr.DataSet
        Time series to be fitted along the time dimension.
    locationcov : list[str]
        List of names of the covariates for the location parameter.
        Have to be names of coordinates in the original data.
    logscalecov : list[str]
        List of names of the covariates for the location parameter.
        Have to be names of coordinates in the original data.
    shapecov : list[str]
        List of names of the covariates for the location parameter.
        Have to be names of coordinates in the original data.
    dist : {"genextreme", "gumbel_r", "genpareto"} or rv_continuous distribution object
        Name of the univariate distributionor the distribution object itself.
        Supported distributions are genextreme, gumbel_r, genpareto.
    method : {"ML", "PWM", "BAYES}
        Fitting method, either maximum likelihood (ML), probability weighted moments (PWM) or bayesian (BAYES).
        The PWM method is usually more robust to outliers.
    dim : str
        The dimension upon which to perform the indexing (default: "time").
    niter : int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
    distributed : bool
        Boolean value indicating whether the covariate data (locationcov, logscalecov, shapecov) is a single array
        (typically found in the coordinates of the Dataset in that case) or if it is distributed along all
        dimensions like the fitting data (typically found in the data variables of the Dataset).

    Returns
    -------
    xr.DataArray
        An array of fitted distribution parameters.

    Notes
    -----
    Coordinates for which all values are NaNs will be dropped before fitting the distribution. If the array still
    contains NaNs or has less valid values than the number of parameters for that distribution,
    the distribution parameters will be returned as NaNs.
    """
    COVARIATE_INDEX.init_covariate_index()
    method = method.upper()
    _check_fit_params(
        dist, method, locationcov, logscalecov, shapecov, confidence_level, distributed
    )
    dist_params = get_params(dist, shapecov, locationcov, logscalecov)
    dist = get_dist(dist)

    # Covariates
    locationcov_data = [ds[covariate].values.tolist() for covariate in locationcov]
    logscalecov_data = [ds[covariate].values.tolist() for covariate in logscalecov]
    shapecov_data = [ds[covariate].values.tolist() for covariate in shapecov]

    out = []
    param_types = ["params", "cint_lower", "cint_upper"]
    results = {}
    for param_type in param_types:
        results[param_type] = apply_func(
            ds,
            _fitfunc,
            dim=dim,
            dist_params=dist_params,
            dist=dist,
            method=method,
            locationcov_data=locationcov_data,
            logscalecov_data=logscalecov_data,
            shapecov_data=shapecov_data,
            niter=niter,
            warmup=warmup,
            confidence_level=confidence_level,
            distributed=distributed,
            param_type=param_type,
        )

    params_data = results["params"]
    cint_lower_data = results["cint_lower"].rename(
        {var: var + "_lower" for var in results["cint_lower"].data_vars}
    )
    cint_upper_data = results["cint_upper"].rename(
        {var: var + "_upper" for var in results["cint_upper"].data_vars}
    )
    data = xr.merge([params_data, cint_lower_data, cint_upper_data])

    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "dparams" for d in ds.dims]
    out = data.assign_coords(dparams=dist_params).transpose(*dims)

    out.attrs = prefix_attrs(
        ds.attrs, ["standard_name", "long_name", "units", "description"], "original_"
    )
    attrs = dict(
        long_name=f"{dist.name} parameters",
        description=f"Parameters of the {dist.name} distribution",
        method=method,
        estimator=METHOD_NAMES[method].capitalize(),
        scipy_dist=dist.name,
        units="",
        history=update_history(
            f"Estimate distribution parameters by {METHOD_NAMES[method]} method along dimension {dim}.",
            new_name="fit",
            data=ds,
        ),
    )
    out.attrs.update(attrs)
    return out


def _fitfunc(
    arr,
    *,
    dist,
    nparams,
    method,
    locationcov_data: list[list],
    logscalecov_data: list[list],
    shapecov_data: list[list],
    niter: int,
    warmup: int,
    confidence_level: float = 0.95,
    distributed: bool = False,
    param_type: str,
):
    r"""Fit a univariate distribution to an array using specified covariate data.

    Parameters
    ----------
    arr : array-like
        Input array containing the data to be fitted to the distribution.
    dist : str or rv_continuous
        The univariate distribution to fit, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    nparams : int
        The number of parameters for the distribution.
    method : str
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    locationcov_data : list[list]
        Nested list containing the data for the location covariates. Each inner list corresponds to a specific
        covariate.
    logscalecov_data : list[list]
        Nested list containing the data for the log scale covariates. Each inner list corresponds to a specific
        covariate.
    shapecov_data : list[list]
        Nested list containing the data for the shape covariates. Each inner list corresponds to a specific
        covariate.
    niter : int
        The number of iterations for the Bayesian inference algorithm used for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations for the Bayesian inference algorithm used for parameter estimation (default: 2000).
    confidence_level : float, optional
        The confidence level for the confidence interval of each parameter (default: 0.95).
    distributed : bool, optional
        Boolean value indicating whether the covariate data is distributed along all dimensions (default: False).
    param_type : str
        The type of parameter to be estimated (e.g., "location", "scale", "shape").

    Returns
    -------
    params : list
        A list of fitted distribution parameters.
    """
    arr = arr.tolist()
    if distributed:
        locationcov_data = [
            locationcov_data[i][COVARIATE_INDEX.get()]
            for i in range(len(locationcov_data))
        ]
        logscalecov_data = [
            logscalecov_data[i][COVARIATE_INDEX.get()]
            for i in range(len(logscalecov_data))
        ]
        shapecov_data = [
            shapecov_data[i][COVARIATE_INDEX.get()] for i in range(len(shapecov_data))
        ]

    # removing NANs from fitting data and covariate data
    locationcov_data_pruned = match_length(arr, locationcov_data)
    logscalecov_data_pruned = match_length(arr, logscalecov_data)
    shapecov_data_pruned = match_length(arr, shapecov_data)

    arr_pruned = np.ma.masked_invalid(arr).compressed()  # pylint: disable=no-member
    # Return NaNs if array is empty, which could happen at previous line if array only contained NANs
    if len(arr_pruned) <= nparams:  # TODO: sanity check with Jonathan
        return np.array([np.nan] * nparams)
    arr_pruned = arr_pruned.tolist()

    if method == "ML":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfit(
                arr_pruned,
                locationcov=locationcov_data_pruned,
                logscalecov=logscalecov_data_pruned,
                shapecov=shapecov_data_pruned,
                confidence_level=confidence_level,
            )[param_type]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(
                params, 1 + num_shape_covariates
            )  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfit(
                arr_pruned,
                locationcov=locationcov_data_pruned,
                logscalecov=logscalecov_data_pruned,
                confidence_level=confidence_level,
            )[param_type]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfit(
                arr_pruned,
                logscalecov=logscalecov_data_pruned,
                shapecov=shapecov_data_pruned,
                confidence_level=confidence_level,
            )[param_type]
            params = np.array(param_list)
            params = exponentiate_logscale(
                params, locationcov_data, logscalecov_data, pareto=True
            )
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "PWM":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitpwm(arr_pruned, confidence_level=confidence_level)[
                param_type
            ]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            params = np.roll(params, 1)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitpwm(arr_pruned, confidence_level=confidence_level)[
                param_type
            ]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitpwm(arr_pruned, confidence_level=confidence_level)[
                param_type
            ]
            params = np.array(param_list)
            params = exponentiate_logscale(
                params, locationcov_data, logscalecov_data, pareto=True
            )
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "BAYES":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitbayes(
                arr_pruned,
                locationcov=locationcov_data_pruned,
                logscalecov=logscalecov_data_pruned,
                shapecov=shapecov_data_pruned,
                niter=niter,
                warmup=warmup,
                confidence_level=confidence_level,
            )[param_type]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(
                params, 1 + num_shape_covariates
            )  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitbayes(
                arr_pruned,
                locationcov=locationcov_data_pruned,
                logscalecov=logscalecov_data_pruned,
                niter=niter,
                warmup=warmup,
                confidence_level=confidence_level,
            )[param_type]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitbayes(
                arr_pruned,
                logscalecov=logscalecov_data_pruned,
                shapecov=shapecov_data_pruned,
                niter=niter,
                warmup=warmup,
                confidence_level=confidence_level,
            )[param_type]
            params = np.array(param_list)
            params = exponentiate_logscale(
                params, locationcov_data, logscalecov_data, pareto=True
            )
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")
    else:
        raise ValueError(f"Fitting method not recognized: {method}")
    COVARIATE_INDEX.inc_covariate_index()
    return params


def get_params(
    dist: str, shapecov: list[str], locationcov: list[str], logscalecov: list[str]
) -> list:
    r"""Return a list of parameter names based on the specified distribution and covariates.

    Parameters
    ----------
    dist : str
        The name of the distribution.
    shapecov : list[str]
        List of covariate names for the shape parameter.
    locationcov : list[str]
        List of covariate names for the location parameter.
    logscalecov : list[str]
        List of covariate names for the log scale parameter.

    Returns
    -------
    list
        A one-dimensional list of parameter names corresponding to the distribution and covariates.

    Examples
    --------
    >>> logscalecov = (["max_temp_yearly"],)
    >>> shapecov = (["qmax_yearly"],)
    >>> get_params("genextreme", shapecov, [], logscalecov)
    >>> [
    ...     "scale",
    ...     "scale_max_temp_yearly_covariate",
    ...     "shape",
    ...     "shape_qmax_yearly_covariate",
    ... ]
    """
    if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
        param_names = ["shape", "loc", "scale"]
        new_param_names = insert_covariates(param_names, locationcov, "loc")
        new_param_names = insert_covariates(new_param_names, logscalecov, "scale")
        new_param_names = insert_covariates(new_param_names, shapecov, "shape")
        return new_param_names

    elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
        param_names = ["loc", "scale"]
        new_param_names = insert_covariates(param_names, locationcov, "loc")
        new_param_names = insert_covariates(new_param_names, logscalecov, "scale")
        return new_param_names
    elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
        param_names = ["scale", "shape"]
        new_param_names = insert_covariates(param_names, logscalecov, "scale")
        new_param_names = insert_covariates(new_param_names, shapecov, "shape")
        return new_param_names
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def _check_fit_params(
    dist: str,
    method: str,
    locationcov: list[str],
    logscalecov: list[str],
    shapecov: list[str],
    confidence_level: float,
    distributed: bool,
):
    r"""Validate the parameters for fitting a univariate distribution. This function is called at the start of fit()
        to make sure that the parameters it is called with are valid.

    Parameters
    ----------
    dist : str
        The name of the distribution to fit.
    method : str
        The fitting method to be used.
    locationcov : list[str]
        List of covariate names for the location parameter.
    logscalecov : list[str]
        List of covariate names for the log scale parameter.
    shapecov : list[str]
        List of covariate names for the shape parameter.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
    distributed : bool
        Indicates whether the covariate data is distributed along all dimensions or not.

    Raises
    ------
    ValueError
        If the combination of arguments is incoherent or invalid for the specified distribution
        and fitting method.
    """
    # Method and distribution names have to be among the recognized ones
    if method not in METHOD_NAMES:
        raise ValueError(f"Fitting method not recognized: {method}")

    if dist not in DIST_NAMES.keys() and str(type(dist)) not in DIST_NAMES.values():
        raise ValueError(f"Fitting distribution not recognized: {dist}")

    # PWM estimation does not work in non-stationary context
    if method == "PWM" and (
        len(locationcov) != 0 or len(logscalecov) != 0 or len(shapecov) != 0
    ):
        covariates = locationcov + logscalecov + shapecov
        raise ValueError(
            f"Probability weighted moment parameter estimation cannot have covariates {covariates}"
        )

    # Gumbel dist has no shape covariate and Pareto dist has no location covariate
    if (dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]) and len(
        shapecov
    ) != 0:
        raise ValueError(
            f"Gumbel distribution has no shape parameter and thus cannot have shape covariates {shapecov}"
        )
    elif (dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]) and len(
        locationcov
    ) != 0:
        raise ValueError(
            f"Pareto distribution has no location parameter and thus cannot have location covariates {locationcov}"
        )

    # Confidence level must be between 0 and 1
    if confidence_level >= 1 or confidence_level <= 0:
        raise ValueError(
            f"Confidence level must be stricly smaller than 1 and strily larger than 0"
        )

    # Parameter 'distributed' should only be set to true if there are covariates
    if distributed and (
        len(locationcov) == 0 and len(logscalecov) == 0 and len(shapecov) == 0
    ):
        raise ValueError(
            f"Parameter 'distributed' is used to indicate how to fetch covariate data; "
            f"it should not be set left to default value of False if there are no covariates"
        )


def apply_func(ds: xr.Dataset, function, **kwargs):
    r"""Apply a specified function to each data variable in the Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset on which the function will be applied.
    function : callable
        The function to be applied to each data variable in the dataset.
    \*\*kwargs : dict
        Additional keyword arguments to be passed to the function.

    Returns
    -------
    xr.Dataset
        A new Dataset with the function applied to each data variable.
    """
    data = xr.Dataset()
    for data_var in ds.data_vars:
        temp_data = xr.apply_ufunc(
            function,
            ds[data_var],
            input_core_dims=[[kwargs["dim"]]],
            output_core_dims=[["dparams"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
            kwargs=dict(
                dist=kwargs["dist"],
                nparams=len(kwargs["dist_params"]),
                method=kwargs["method"],
                locationcov_data=kwargs["locationcov_data"],
                logscalecov_data=kwargs["logscalecov_data"],
                shapecov_data=kwargs["shapecov_data"],
                niter=kwargs["niter"],
                warmup=kwargs["warmup"],
                confidence_level=kwargs["confidence_level"],
                distributed=kwargs["distributed"],
                param_type=kwargs["param_type"],
            ),
            dask_gufunc_kwargs={
                "output_sizes": {"dparams": len(kwargs["dist_params"])}
            },
        )
        data = xr.merge([data, temp_data])
        COVARIATE_INDEX.init_covariate_index()
    return data
