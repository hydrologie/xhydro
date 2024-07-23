"""Parameter estimation functions for the extreme value analysis module."""

from __future__ import annotations

import math

import numpy as np
import scipy.stats
import xarray as xr
from xclim.core.formatting import prefix_attrs, update_history
from xclim.indices.stats import get_dist

from xhydro.extreme_value_analysis import Extremes
from xhydro.extreme_value_analysis.structures.conversions import (
    jl_matrix_tuple_to_py_list,
    jl_vector_to_py_list,
    jl_vector_tuple_to_py_list,
    py_list_to_jl_vector,
)
from xhydro.extreme_value_analysis.structures.dataitem import Variable
from xhydro.extreme_value_analysis.structures.util import jl_variable_fit_parameters

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
    locationcov: list[Variable] = [],
    logscalecov: list[Variable] = [],
    shapecov: list[Variable] = [],
) -> list:
    r"""
    Fit an array to a general extreme value distribution using Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[Variable]
        List of variables to be used as covariates for the location parameter.
    logscalecov : list[Variable]
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list[Variable]
        List of variables to be used as covariates for the shape parameter.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(
        [locationcov, logscalecov, shapecov]
    )
    jl_model = Extremes.gevfit(
        jl_y,
        locationcov=jl_locationcov,
        logscalecov=jl_logscalecov,
        shapecov=jl_shapecov,
    )
    # TODO: continue test for prametric quantile
    return _param_cint(jl_model)
    # return {"params": params, "jl_fm": jl_model}


def gumbelfit(
    y: list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = []
) -> list:
    r"""
    Fit an array to a Gumbel distribution using Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[Variable]
        List of variables to be used as covariates for the location parameter.
    logscalecov : list[Variable]
        List of variables to be used as covariates for the logscale parameter.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters(
        [locationcov, logscalecov]
    )
    jl_model = Extremes.gumbelfit(
        jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov
    )
    return _param_cint(jl_model)


def gpfit(
    y: list[float], logscalecov: list[Variable] = [], shapecov: list[Variable] = []
) -> list:
    r"""
    Fit an array to a generalized Pareto distribution using Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    logscalecov : list[Variable]
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list[Variable]
        List of variables to be used as covariates for the shape parameter.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([logscalecov, shapecov])
    jl_model = Extremes.gpfit(jl_y, logscalecov=jl_logscalecov, shapecov=jl_shapecov)
    return _param_cint(jl_model, pareto=True)


# Probability weighted moment estimation
def gevfitpwm(y: list[float]) -> list:
    r"""
    Fit an array to a general extreme value distribution using the probability weighted moments (PWM) method with Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_model = Extremes.gevfitpwm(jl_y)
    return _param_cint(jl_model)


def gumbelfitpwm(y: list[float]) -> list:
    r"""
    Fit an array to a Gumbel distribution using the probability weighted moments (PWM) method with Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_model = Extremes.gumbelfitpwm(jl_y)
    return _param_cint(jl_model)


def gpfitpwm(y: list[float]) -> list:
    r"""
    Fit an array to a generalized Pareto distribution using the probability weighted moments (PWM) method with Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_model = Extremes.gpfitpwm(jl_y)
    return _param_cint(jl_model)


# Bayesian estimation
def gevfitbayes(
    y: list[float],
    locationcov: list[Variable] = [],
    logscalecov: list[Variable] = [],
    shapecov: list[Variable] = [],
    niter: int = 5000,
    warmup: int = 2000,
) -> list:
    r"""
    Fit an array to a general extreme value distribution using Bayesian inference with Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[Variable]
        List of variables to be used as covariates for the location parameter.
    logscalecov : list[Variable]
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list[Variable]
        List of variables to be used as covariates for the shape parameter.
    niter : int
        Number of iterations for the Bayesian sampler.
    warmup : int
        Number of warmup iterations for the Bayesian sampler.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters(
        [locationcov, logscalecov, shapecov]
    )
    jl_model = Extremes.gevfitbayes(
        jl_y,
        locationcov=jl_locationcov,
        logscalecov=jl_logscalecov,
        shapecov=jl_shapecov,
        niter=niter,
        warmup=warmup,
    )
    return _param_cint(jl_model, bayesian=True)


def gumbelfitbayes(
    y: list[float],
    locationcov: list[Variable] = [],
    logscalecov: list[Variable] = [],
    niter: int = 5000,
    warmup: int = 2000,
) -> list:
    r"""
    Fit an array to a Gumbel distribution using Bayesian inference with Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    locationcov : list[Variable]
        List of variables to be used as covariates for the location parameter.
    logscalecov : list[Variable]
        List of variables to be used as covariates for the logscale parameter.
    niter : int
        Number of iterations for the Bayesian sampler.
    warmup : int
        Number of warmup iterations for the Bayesian sampler.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters(
        [locationcov, logscalecov]
    )
    jl_model = Extremes.gumbelfitbayes(
        jl_y,
        locationcov=jl_locationcov,
        logscalecov=jl_logscalecov,
        niter=niter,
        warmup=warmup,
    )
    return _param_cint(jl_model, bayesian=True)


def gpfitbayes(
    y: list[float],
    logscalecov: list[Variable] = [],
    shapecov: list[Variable] = [],
    niter: int = 5000,
    warmup: int = 2000,
) -> list:
    r"""
    Fit an array to a generalized Pareto distribution using Bayesian inference with Extremes.jl.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    logscalecov : list[Variable]
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list[Variable]
        List of variables to be used as covariates for the shape parameter.
    niter : int
        Number of iterations for the Bayesian sampler.
    warmup : int
        Number of warmup iterations for the Bayesian sampler.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([logscalecov, shapecov])
    jl_model = Extremes.gpfitbayes(
        jl_y,
        logscalecov=jl_logscalecov,
        shapecov=jl_shapecov,
        niter=niter,
        warmup=warmup,
    )
    return _param_cint(jl_model, bayesian=True)


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


def fit(
    ds: xr.Dataset,
    dist: str | scipy.stats.rv_continuous = "genextreme",
    method: str = "ML",
    dim: str = "time",
) -> xr.Dataset:
    r"""Fit an array to a univariate distribution along the time dimension.

    Parameters
    ----------
    ds : xr.DataSet
        Time series to be fitted along the time dimension.
    dist : str or rv_continuous distribution object
        Name of the univariate distributionor the distribution object itself.
        Supported distributions are genextreme, gumbel_r, genpareto.
    method : {"ML","PWM", "BAYES}
        Fitting method, either maximum likelihood (ML), probability weighted moments (PWM) or bayesian (BAYES).
        The PWM method is usually more robust to outliers.
    dim : str
        The dimension upon which to perform the indexing (default: "time").

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
    _check_fit_params(dist, method)
    dist_params = _get_params(dist)
    dist = get_dist(dist)

    out = []
    data = xr.apply_ufunc(
        _fitfunc_1d,
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
        ),
        dask_gufunc_kwargs={"output_sizes": {"dparams": len(dist_params)}},
    )
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


def _fitfunc_1d(arr, *, dist, nparams, method):
    """Fit distribution parameters."""
    x = np.ma.masked_invalid(arr).compressed()  # pylint: disable=no-member
    # Return NaNs if array is empty, which could happen at previous line if array only contained Nans
    # if len(x) <= 1:
    if len(x) <= nparams:  # TODO: sanity check with Jonathan
        return np.asarray([np.nan] * nparams)

    if method == "ML":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            # TODO: continue test for parametric quantile
            param_list = gevfit(x)
            params = np.asarray(param_list)
            params = np.roll(params, 3)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfit(x)
            params = np.asarray(param_list)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfit(x)
            params = np.asarray(param_list)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "PWM":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitpwm(x)
            params = np.asarray(param_list)
            params = np.roll(params, 3)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitpwm(x)
            params = np.asarray(param_list)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitpwm(x)
            params = np.asarray(param_list)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "BAYES":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitbayes(x)
            params = np.asarray(param_list)
            params = np.roll(params, 3)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitbayes(x)
            params = np.asarray(param_list)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitbayes(x)
            params = np.asarray(param_list)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")
    else:
        raise ValueError(f"Fitting method not recognized: {method}")
    return params


def _get_params(dist: str) -> list:
    r"""Returns one-dimensional list of parameter names according to the distribution given."""
    if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
        return [
            "shape",
            "shape_lower",
            "shape_upper",
            "loc",
            "loc_lower",
            "loc_upper",
            "scale",
            "scale_lower",
            "scale_upper",
        ]

    elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
        return ["loc", "loc_lower", "loc_upper", "scale", "scale_lower", "scale_upper"]
    elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
        return [
            "scale",
            "scale_lower",
            "scale_upper",
            "shape",
            "shape_lower",
            "shape_upper",
        ]
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def _check_fit_params(dist: str, method: str):
    r"""Raises ValueError if distribution or method is not valid."""
    if method not in METHOD_NAMES:
        raise ValueError(f"Fitting method not recognized: {method}")

    if dist not in DIST_NAMES.keys() and str(type(dist)) not in DIST_NAMES.values():
        raise ValueError(f"Fitting distribution not recognized: {dist}")


def _param_cint(jl_model, bayesian: bool = False, pareto: bool = False) -> list:
    r"""Returns list of parameters and confidence intervals for given julia fitted model."""
    if bayesian:
        params = jl_matrix_tuple_to_py_list(Extremes.params(jl_model))
        params = [
            sum(x) / len(params) for x in zip(*params)
        ]  # each parameter is estimated to be the average over all simulations
    else:
        params = jl_vector_tuple_to_py_list(Extremes.params(jl_model))
    cint = [jl_vector_to_py_list(interval) for interval in Extremes.cint(jl_model)]
    # confidence interval for scale parameter is on a logarithmic scale
    if pareto:
        params = params[
            1:
        ]  # we don't include the location in pareto distribution, because it is always equal to 0
        cint[0] = [math.exp(cint[0][i]) for i in range(len(cint[0]))]
    else:
        cint[1] = [math.exp(cint[1][i]) for i in range(len(cint[1]))]
    param_cint = [item for pair in zip(params, cint) for item in (pair[0], *pair[1])]
    return param_cint
