"""Parameter estimation functions for the extreme value analysis module."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.stats
import xarray as xr
from xclim.core.formatting import prefix_attrs, update_history
from xclim.indices.stats import get_dist

try:
    from xhydro.extreme_value_analysis import Extremes
    from xhydro.extreme_value_analysis.structures.conversions import (
        jl_matrix_tuple_to_py_list,
        jl_vector_to_py_list,
        jl_vector_tuple_to_py_list,
        py_list_to_jl_vector,
    )
    from xhydro.extreme_value_analysis.structures.dataitem import Variable
    from xhydro.extreme_value_analysis.structures.util import jl_variable_fit_parameters
except (ImportError, ModuleNotFoundError) as e:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    raise ImportError(JULIA_WARNING) from e


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
    "get_params",
]


# Maximum likelihood estimation
# kamil
def gevfit(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    shapecov: list[list] = ()
) -> list:
    r"""
    Fit an array to a general extreme value distribution using Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.
    locationcov : list of Variable
        List of variables to be used as covariates for the location parameter.
    logscalecov : list of Variable
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list of Variable
        List of variables to be used as covariates for the shape parameter.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters_debug(
        locationcov), jl_variable_fit_parameters_debug(logscalecov), jl_variable_fit_parameters_debug(shapecov)
    jl_model = Extremes.gevfit(
        jl_y,
        locationcov=jl_locationcov,
        logscalecov=jl_logscalecov,
        shapecov=jl_shapecov,
    )
    return _param_cint_debug(jl_model)


def gumbelfit(
    y: list[float], locationcov: list[list] = (), logscalecov: list[list] = ()
) -> list:
    r"""
    Fit an array to a Gumbel distribution using Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.
    locationcov : list of Variable
        List of variables to be used as covariates for the location parameter.
    logscalecov : list of Variable
        List of variables to be used as covariates for the logscale parameter.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters_debug(locationcov), jl_variable_fit_parameters_debug(logscalecov)
    jl_model = Extremes.gumbelfit(
        jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov
    )
    return _param_cint_debug(jl_model)


def gpfit(
    y: list[float], logscalecov: list[list] = (), shapecov: list[list] = ()
) -> list:
    r"""
    Fit an array to a generalized Pareto distribution using Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.
    logscalecov : list of Variable
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list of Variable
        List of variables to be used as covariates for the shape parameter.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters_debug(logscalecov), jl_variable_fit_parameters_debug(shapecov)
    jl_model = Extremes.gpfit(jl_y, logscalecov=jl_logscalecov, shapecov=jl_shapecov)
    return _param_cint_debug(jl_model)


# Probability weighted moment estimation
def gevfitpwm(y: list[float]) -> list:
    r"""
    Fit an array to a general extreme value distribution using the probability weighted moments (PWM) method with Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_model = Extremes.gevfitpwm(jl_y)
    return _param_cint_debug(jl_model)


def gumbelfitpwm(y: list[float]) -> list:
    r"""
    Fit an array to a Gumbel distribution using the probability weighted moments (PWM) method with Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_model = Extremes.gumbelfitpwm(jl_y)
    return _param_cint_debug(jl_model)


def gpfitpwm(y: list[float]) -> list:
    r"""
    Fit an array to a generalized Pareto distribution using the probability weighted moments (PWM) method with Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.

    Returns
    -------
    list
        An array of fitted distribution parameters and their confidence intervals at 95% confidence.
    """
    jl_y = py_list_to_jl_vector(y)
    jl_model = Extremes.gpfitpwm(jl_y)
    return _param_cint_debug(jl_model)


# Bayesian estimation
def gevfitbayes(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
) -> list:
    r"""
    Fit an array to a general extreme value distribution using Bayesian inference with Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.
    locationcov : list of Variable
        List of variables to be used as covariates for the location parameter.
    logscalecov : list of Variable
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list of Variable
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
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters_debug(locationcov), jl_variable_fit_parameters_debug(logscalecov), jl_variable_fit_parameters_debug(shapecov)
    jl_model = Extremes.gevfitbayes(
        jl_y,
        locationcov=jl_locationcov,
        logscalecov=jl_logscalecov,
        shapecov=jl_shapecov,
        niter=niter,
        warmup=warmup,
    )
    return _param_cint_debug(jl_model, bayesian=True)


def gumbelfitbayes(
    y: list[float],
    locationcov: list[list] = (),
    logscalecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
) -> list:
    r"""
    Fit an array to a Gumbel distribution using Bayesian inference with Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.
    locationcov : list of Variable
        List of variables to be used as covariates for the location parameter.
    logscalecov : list of Variable
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
    jl_locationcov, jl_logscalecov = jl_variable_fit_parameters_debug(locationcov), jl_variable_fit_parameters_debug(logscalecov)
    jl_model = Extremes.gumbelfitbayes(
        jl_y,
        locationcov=jl_locationcov,
        logscalecov=jl_logscalecov,
        niter=niter,
        warmup=warmup,
    )
    return _param_cint_debug(jl_model, bayesian=True)


def gpfitbayes(
    y: list[float],
    logscalecov: list[list] = (),
    shapecov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
) -> list:
    r"""
    Fit an array to a generalized Pareto distribution using Bayesian inference with Extremes.jl.

    Parameters
    ----------
    y : list of float
        Data to be fitted.
    logscalecov : list of Variable
        List of variables to be used as covariates for the logscale parameter.
    shapecov : list of Variable
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
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters_debug(logscalecov), jl_variable_fit_parameters_debug(shapecov)
    jl_model = Extremes.gpfitbayes(
        jl_y,
        logscalecov=jl_logscalecov,
        shapecov=jl_shapecov,
        niter=niter,
        warmup=warmup,
    )
    return _param_cint_debug(jl_model, bayesian=True)


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
    warmup: int = 2000
) -> xr.Dataset:
    r"""Fit an array to a univariate distribution along the time dimension.

    Parameters
    ----------
    ds : xr.DataSet
        Time series to be fitted along the time dimension.
    dist : {"genextreme", "gumbel_r", "genpareto"} or rv_continuous distribution object
        Name of the univariate distributionor the distribution object itself.
        Supported distributions are genextreme, gumbel_r, genpareto.
    method : {"ML", "PWM", "BAYES}
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
    distribution = get_dist(dist)

    # Covariates
    locationcov_data = [ds[covariate].values.tolist() for covariate in locationcov]
    logscalecov_data = [ds[covariate].values.tolist() for covariate in logscalecov]
    shapecov_data = [ds[covariate].values.tolist() for covariate in shapecov]

    out = []
    data = xr.apply_ufunc(
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
            dist=distribution,
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
    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "dparams" for d in ds.dims]
    out = data.assign_coords(dparams=dist_params).transpose(*dims)

    out.attrs = prefix_attrs(
        ds.attrs, ["standard_name", "long_name", "units", "description"], "original_"
    )
    attrs = dict(
        long_name=f"{distribution.name} parameters",
        description=f"Parameters of the {distribution.name} distribution",
        method=method,
        estimator=METHOD_NAMES[method].capitalize(),
        scipy_dist=distribution.name,
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
    locationcov_data_pruned = _match_length(arr, locationcov_data)
    logscalecov_data_pruned = _match_length(arr, logscalecov_data)
    shapecov_data_pruned = _match_length(arr, shapecov_data)
    arr_pruned = np.ma.masked_invalid(arr).compressed()  # pylint: disable=no-member
    # Return NaNs if array is empty, which could happen at previous line if array only contained NANs
    if len(arr_pruned) <= nparams:  # TODO: sanity check with Jonathan
        return np.asarray([np.nan] * nparams)
    arr_pruned = arr_pruned.tolist()
    if method == "ML":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned)
            params = np.asarray(param_list)
            params = _exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfit(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned)
            params = np.asarray(param_list)
            params = _exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfit(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned)
            params = np.asarray(param_list)
            params = _exponentiate_logscale(params, locationcov_data, logscalecov_data)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "PWM":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitpwm(arr_pruned)
            params = np.asarray(param_list)
            params = _exponentiate_logscale(params, locationcov_data, logscalecov_data)
            params = np.roll(params, 1)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitpwm(arr_pruned)
            params = np.asarray(param_list)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitpwm(arr_pruned)
            params = np.asarray(param_list)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "BAYES":
        if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
            param_list = gevfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup)
            params = np.asarray(param_list)
            params = _exponentiate_logscale(params, locationcov_data, logscalecov_data)
            num_shape_covariates = len(shapecov_data)
            params = np.roll(params, 1 + num_shape_covariates)  # to have [shape, loc, scale]
        elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
            param_list = gumbelfitbayes(arr_pruned, locationcov=locationcov_data_pruned, logscalecov=logscalecov_data_pruned, niter=niter, warmup=warmup)
            params = np.asarray(param_list)
            params = _exponentiate_logscale(params, locationcov_data, logscalecov_data)
        elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
            param_list = gpfitbayes(arr_pruned, logscalecov=logscalecov_data_pruned, shapecov=shapecov_data_pruned, niter=niter, warmup=warmup)
            params = np.asarray(param_list)
            params = _exponentiate_logscale(params, locationcov_data, logscalecov_data)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")
    else:
        raise ValueError(f"Fitting method not recognized: {method}")
    return params


def get_params(dist: str, shapecov: list[str], locationcov: list[str], logscalecov: list[str]) -> list:
    r"""Return one-dimensional list of parameter names according to the distribution given."""
    if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
        param_names = ["shape", "loc", "scale"]
        new_param_names = _insert_covariates(param_names, locationcov, "loc")
        new_param_names = _insert_covariates(new_param_names, logscalecov, "scale")
        new_param_names = _insert_covariates(new_param_names, shapecov, "shape")
        return new_param_names

    elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
        param_names = ["loc", "scale"]
        new_param_names = _insert_covariates(param_names, locationcov, "loc")
        new_param_names = _insert_covariates(new_param_names, logscalecov, "scale")
        return new_param_names
    elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
        param_names = ["shape", "scale"]
        new_param_names = _insert_covariates(param_names, shapecov, "shape")
        new_param_names = _insert_covariates(new_param_names, logscalecov, "scale")
        return new_param_names
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def _check_fit_params(dist: str, method: str, locationcov: list[str], logscalecov: list[str], shapecov:list[str]):
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
        cint[0] = [np.exp(cint[0][i]) for i in range(len(cint[0]))]
    else:
        cint[1] = [np.exp(cint[1][i]) for i in range(len(cint[1]))]
    param_cint = [item for pair in zip(params, cint) for item in (pair[0], *pair[1])]
    return param_cint

def _param_cint_debug(jl_model, bayesian: bool = False) -> list:
    r"""Returns list of parameters and confidence intervals for given julia fitted model."""
    if bayesian:
        jl_params_sims = jl_model.sim.value
        py_params_sims = [jl_vector_to_py_list(jl.vec(jl_params_sims[i, :, :])) for i in range(len(jl_params_sims[:, 0, 0]))]
        params = [sum(x) / len(py_params_sims) for x in zip(*py_params_sims)]  # each parameter is estimated to be the average over all simulations
    else:
        params = jl_vector_to_py_list(getattr(jl_model, "θ̂"))
    cint = [jl_vector_to_py_list(interval) for interval in Extremes.cint(jl_model)]
    cint_lower = [interval[0] for interval in cint]
    cint_upper = [interval[1] for interval in cint]
    # return [params, cint_lower, cint_upper]
    return params
def _insert_covariates(param_names: list[str], covariates: list[str], param_name: str):
    r"""Insert appropriate covariate variables names in the parameter names list"""
    index = param_names.index(param_name)
    return (
        param_names[:index + 1]
        + [f'{param_name}_{covariate}_covariate' for covariate in covariates]
        + param_names[index + 1:]
    )

def _match_length(py_list: list, covariates: list[list]) -> list[list]:
    r"""
    Weird function with very specific use case. When cleaning up the fitting data to
    get rid of NANs, the fitting data can have a different length than its corresponding
    covariate data. For example, you might have:

    fitting_data = [1,2, nan, 4, 5]
    and a covariate
    loc_covariate = [6,5,7,8,9]

    If you simply prune the fitting data, the covariate data will be of a different length
    and fitting will not work because both need to be of the same length to avoid a
    JuliaError: AssertionError: The explanatory variable length should match data length.

    Therefore, this function compares the original fitting data (py_list) with the
    covariate data (covariates) and returns a new list of covariates  that<s been pruned
    by deleting all values which correspond to a NAN in the fitting data (py_list).

    For example
    fitting_data = [1,2, nan, 4, 5]
    loc_covariate = [6,5,7,8,9]
    shape_covariate = [9,7,6,5,4]
    _match_length(fitting_data, [loc_covariate, shape_covariate])
    would return [[6,5,8,9], [9,7,5,4]]
    """
    nan_indexes = [index for index, value in enumerate(py_list) if (math.isnan(value) or np.isnan(value))]
    covariates_copy = copy.deepcopy(covariates)
    for sublist in covariates_copy:
        for index in sorted(nan_indexes, reverse=True):
            del sublist[index]
    return covariates_copy


def _exponentiate_logscale(params: np.ndarray, locationcov_data: list[list], logscalecov_data:list[list]) -> np.ndarray:
    scale_param_index = 1 + len(locationcov_data)
    for index in range(scale_param_index, scale_param_index + len(logscalecov_data) + 1):
        params[index] = np.exp(params[index])
    return params
