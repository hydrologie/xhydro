"""Parameter estimation functions for the extreme value analysis module."""

from __future__ import annotations
import warnings
import numpy as np
import scipy.stats
import xarray as xr
from xclim.core.formatting import prefix_attrs, update_history
from xclim.indices.stats import get_dist
from xhydro.extreme_value_analysis import Extremes, jl
from xhydro.extreme_value_analysis.structures.conversions import (py_list_to_jl_vector,)
from xhydro.extreme_value_analysis.structures.util import jl_variable_fit_parameters
from xhydro.extreme_value_analysis.structures.util import param_cint, exponentiate_logscale, match_length, insert_covariates
warnings.simplefilter("always", UserWarning)


__all__ = [
    "fit",
    "gevfit",
    "gevfitbayes",
    "gevfitpwm",
    "gpfit",
    "gpfitbayes",
    "gpfitpwm",
    "gumbelfit",
    "gumbelfitbayes",
    "gumbelfitpwm",
]


# Maximum likelihood estimation
def gevfit(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    confidence_level: float = 0.95
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
    confidence_level: float
        The confidence level for the confidence interval of each parameter

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(
        locationcov), jl_variable_fit_parameters(logscalecov), jl_variable_fit_parameters(shapecov)
    nparams = 3 + len(locationcov) + len(logscalecov) + len(shapecov)
    try:
        jl_model = Extremes.gevfit(
            jl_y,
            locationcov=jl_locationcov,
            logscalecov=jl_logscalecov,
            shapecov=jl_shapecov,
        )
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except:
        warnings.warn("There was an error in fitting the data to a genextreme distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


def gumbelfit(
    y: list[float], locationcov: list[list] = (), logscalecov: list[list] = (), confidence_level: float = 0.95
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
    confidence_level: float
        The confidence level for the confidence interval of each parameter

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters(locationcov), jl_variable_fit_parameters(logscalecov)
    nparams = 2 + len(locationcov) + len(logscalecov)
    try:
        jl_model = Extremes.gumbelfit(
            jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov
        )
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except:
        warnings.warn("There was an error in fitting the data to a gumbel_r distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


def gpfit(
    y: list[float], logscalecov: list[list] = (), shapecov: list[list] = (), confidence_level: float = 0.95
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
    confidence_level: float
        The confidence level for the confidence interval of each parameter

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(logscalecov), jl_variable_fit_parameters(shapecov)
    nparams = 2 + len(logscalecov) + len(shapecov)
    try:
        jl_model = Extremes.gpfit(jl_y, logscalecov=jl_logscalecov, shapecov=jl_shapecov)
        return param_cint(jl_model, nparams, confidence_level=confidence_level)
    except:
        warnings.warn("There was an error in fitting the data to a genpareto distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


# Probability weighted moment estimation
def gevfitpwm(y: list[float], confidence_level: float = 0.95) -> dict[str, list[float]]:
    r"""
    Fit an array to a Genextreme distribution using the probability weighted moment algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    confidence_level: float
        The confidence level for the confidence interval of each parameter

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
    except:
        warnings.warn("There was an error in fitting the data to a genextreme distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


def gumbelfitpwm(y: list[float], confidence_level: float = 0.95) -> dict[str, list[float]]:
    r"""
    Fit an array to a Gumbel distribution using the probability weighted moment algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    confidence_level: float
        The confidence level for the confidence interval of each parameter

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
    except:
        warnings.warn("There was an error in fitting the data to a gumbel_r distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


def gpfitpwm(y: list[float], confidence_level: float = 0.95) -> dict[str, list[float]]:
    r"""
    Fit an array to a Pareto distribution using the probability weighted moment algorithm from Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    confidence_level: float
        The confidence level for the confidence interval of each parameter

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
    except:
        warnings.warn("There was an error in fitting the data to a genpareto distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


# Bayesian estimation
def gevfitbayes(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95
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
    niter: int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup: int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level: float
        The confidence level for the confidence interval of each parameter

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(locationcov), jl_variable_fit_parameters(logscalecov), jl_variable_fit_parameters(shapecov)
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
        return param_cint(jl_model, nparams, bayesian=True, confidence_level=confidence_level)
    except:
        warnings.warn("There was an error in fitting the data to a genextreme distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


def gumbelfitbayes(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95
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
    niter: int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup: int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level: float
        The confidence level for the confidence interval of each parameter

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters(locationcov), jl_variable_fit_parameters(logscalecov)
    nparams = 2 + len(locationcov) + len(logscalecov)
    try:
        jl_model = Extremes.gumbelfitbayes(
            jl_y,
            locationcov=jl_locationcov,
            logscalecov=jl_logscalecov,
            niter=niter,
            warmup=warmup,
        )
        return param_cint(jl_model, nparams, bayesian=True, confidence_level=confidence_level)
    except:
        warnings.warn("There was an error in fitting the data to a gumbel distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


def gpfitbayes(
    y: list[float],
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95
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
    niter: int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup: int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level: float
        The confidence level for the confidence interval of each parameter

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the estimated parameters and the lower and upper bounds for the confidence interval
        of each parameter.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(logscalecov), jl_variable_fit_parameters(shapecov)
    nparams = 2 + len(logscalecov) + len(shapecov)
    try:
        jl_model = Extremes.gpfitbayes(
            jl_y,
            logscalecov=jl_logscalecov,
            shapecov=jl_shapecov,
            niter=niter,
            warmup=warmup,
        )
        return param_cint(jl_model, nparams, bayesian=True, confidence_level=confidence_level)
    except:
        warnings.warn("There was an error in fitting the data to a genpareto distribution. Returned parameters are numpy.nan.")
        return {"params": [np.nan for _ in range(nparams)], "cint_lower": [np.nan for _ in range(nparams)], "cint_upper": [np.nan for _ in range(nparams)]}


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


# kamil
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
    confidence_level: float = 0.95
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
    dist : str or rv_continuous distribution object
        Name of the univariate distributionor the distribution object itself.
        Supported distributions are genextreme, gumbel_r, genpareto.
    method : {"ML","PWM", "BAYES}
        Fitting method, either maximum likelihood (ML), probability weighted moments (PWM) or bayesian (BAYES).
        The PWM method is usually more robust to outliers.
    dim : str
        The dimension upon which to perform the indexing (default: "time").
    niter: int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup: int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    confidence_level: float
        The confidence level for the confidence interval of each parameter


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
    method = method.upper()
    _check_fit_params(dist, method, locationcov, logscalecov, shapecov, confidence_level)
    dist_params = get_params(dist, shapecov, locationcov, logscalecov)
    dist = get_dist(dist)

    # Covariates
    locationcov_data = [ds[covariate].values.tolist() for covariate in locationcov]
    logscalecov_data = [ds[covariate].values.tolist() for covariate in logscalecov]
    shapecov_data = [ds[covariate].values.tolist() for covariate in shapecov]

    out = []
    params_data = xr.apply_ufunc(
        _fitfunc_params,
        ds,
        input_core_dims=[[dim]],
        output_core_dims=[["dparams"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        kwargs=dict(
            # Don't know how APP should be included, this works for now
            dist=dist,
            nparams=len(dist_params),
            method=method,
            locationcov_data=locationcov_data,
            logscalecov_data=logscalecov_data,
            shapecov_data=shapecov_data,
            niter=niter,
            warmup=warmup
        ),
        dask_gufunc_kwargs={"output_sizes": {"dparams": len(dist_params)}},
    )

    cint_lower_data = xr.apply_ufunc(
        _fitfunc_cint_lower,
        ds,
        input_core_dims=[[dim]],
        output_core_dims=[["dparams"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        kwargs=dict(
            # Don't know how APP should be included, this works for now
            dist=dist,
            nparams=len(dist_params),
            method=method,
            locationcov_data=locationcov_data,
            logscalecov_data=logscalecov_data,
            shapecov_data=shapecov_data,
            niter=niter,
            warmup=warmup,
            confidence_level=confidence_level
        ),
        dask_gufunc_kwargs={"output_sizes": {"dparams": len(dist_params)}},
    )

    cint_upper_data = xr.apply_ufunc(
        _fitfunc_cint_upper,
        ds,
        input_core_dims=[[dim]],
        output_core_dims=[["dparams"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        kwargs=dict(
            # Don't know how APP should be included, this works for now
            dist=dist,
            nparams=len(dist_params),
            method=method,
            locationcov_data=locationcov_data,
            logscalecov_data=logscalecov_data,
            shapecov_data=shapecov_data,
            niter=niter,
            warmup=warmup,
            confidence_level=confidence_level
        ),
        dask_gufunc_kwargs={"output_sizes": {"dparams": len(dist_params)}},
    )

    cint_lower_data = cint_lower_data.rename({var: var + '_lower' for var in cint_lower_data.data_vars})
    cint_upper_data = cint_upper_data.rename({var: var + '_upper' for var in cint_upper_data.data_vars})
    params_data = xr.merge([params_data, cint_lower_data, cint_upper_data])

    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "dparams" for d in ds.dims]
    out = params_data.assign_coords(dparams=dist_params).transpose(*dims)

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


def _fitfunc_params(arr, *, dist, nparams, method, locationcov_data: list[list], logscalecov_data: list[list], shapecov_data: list[list], niter: int, warmup: int):
    """Fit distribution parameters."""
    arr = arr.tolist()
    # removing NANs from fitting data and covariates
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
            param_list = gevfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfit(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "PWM":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitpwm(arr_pruned)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            params = np.roll(params, 1)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitpwm(arr_pruned)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitpwm(arr_pruned)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "BAYES":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, niter=niter, warmup=warmup)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitbayes(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup)["params"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")
    else:
        raise ValueError(f"Fitting method not recognized: {method}")
    return params

def _fitfunc_cint_lower(arr, *, dist, nparams, method, locationcov_data: list[list], logscalecov_data: list[list], shapecov_data: list[list], niter: int, warmup: int, confidence_level: float):
    """Fit distribution parameters."""
    arr = arr.tolist()
    # removing NANs from fitting data and covariates
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
            param_list = gevfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfit(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "PWM":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitpwm(arr_pruned, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            params = np.roll(params, 1)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitpwm(arr_pruned, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitpwm(arr_pruned, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "BAYES":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, niter=niter, warmup=warmup, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitbayes(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup, confidence_level=confidence_level)["cint_lower"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")
    else:
        raise ValueError(f"Fitting method not recognized: {method}")
    return params


def _fitfunc_cint_upper(arr, *, dist, nparams, method, locationcov_data: list[list], logscalecov_data: list[list], shapecov_data: list[list], niter: int, warmup: int, confidence_level: float):
    """Fit distribution parameters."""
    arr = arr.tolist()
    # removing NANs from fitting data and covariates
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
            param_list = gevfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfit(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "PWM":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitpwm(arr_pruned, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            params = np.roll(params, 1)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitpwm(arr_pruned, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitpwm(arr_pruned, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "BAYES":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, niter=niter, warmup=warmup, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitbayes(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup, confidence_level=confidence_level)["cint_upper"]
            params = np.array(param_list)
            params = exponentiate_logscale(params, locationcov_data, logscalecov_data, pareto=True)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")
    else:
        raise ValueError(f"Fitting method not recognized: {method}")
    return params

def get_params(dist: str, shapecov: list[str], locationcov: list[str], logscalecov: list[str]) -> list:
    r"""Return one-dimensional list of parameter names according to the distribution given."""
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


def _check_fit_params(dist: str, method: str, locationcov: list[str], logscalecov: list[str], shapecov:list[str], confidence_level: float):
    r"""Raises ValueError if fitting parameters are not valid"""

    # Method and distribution names have to be among the recognized ones
    if method not in METHOD_NAMES:
        raise ValueError(f"Fitting method not recognized: {method}")

    if dist not in DIST_NAMES.keys() and str(type(dist)) not in DIST_NAMES.values():
        raise ValueError(f"Fitting distribution not recognized: {dist}")

    # PWM estimation does not work in non-stationary context
    if method == "PWM" and (len(locationcov) != 0 or len(logscalecov) != 0 or len(shapecov) != 0):
        covariates = locationcov + logscalecov + shapecov
        raise ValueError(f"Probability weighted moment parameter estimation cannot have covariates {covariates}")

    # Gumbel dist has no shape covariate and Pareto dist has no location covariate
    if (dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]) and len(shapecov) != 0:
        raise ValueError(f"Gumbel distribution has no shape parameter and thus cannot have shape covariates {shapecov}")
    elif (dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]) and len(locationcov) != 0:
        raise ValueError(f"Pareto distribution has no location parameter and thus cannot have location covariates {locationcov}")

    # Confidence level must be between 0 and 1
    if confidence_level >= 1 or confidence_level <= 0:
        raise ValueError(f"Confidence level must be stricly smaller than 1 and strily larger than 0")












