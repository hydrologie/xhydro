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
        DIST_NAMES,
        METHOD_NAMES,
        change_sign_param,
        create_nan_mask,
        exponentiate_logscale,
        insert_covariates,
        jl_variable_fit_parameters,
        param_cint,
        recover_nan,
        remove_nan,
        return_level_cint,
        return_nan,
    )
except (ImportError, ModuleNotFoundError) as e:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    raise ImportError(JULIA_WARNING) from e


warnings.simplefilter("always", UserWarning)
__all__ = ["fit", "return_level"]


def _fit_model(
    y: list[float],
    dist: str,
    method: str,
    location_cov: list[list] = (),
    scale_cov: list[list] = (),
    shape_cov: list[list] = (),
    niter: int = 5000,
    warmup: int = 2000,
) -> list:
    r"""
    Fit a distribution using the specified covariate data.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    dist : str or rv_continuous
        Distribution, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    method : str
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    location_cov : list[list]
        List of data lists to be used as covariates for the location parameter.
    scale_cov : list[list]
        List of data lists to be used as covariates for the scale parameter.
    shape_cov : list[list]
        List of data lists to be used as covariates for the shape parameter.l.
    niter : int
        Required when when method=BAYES. The number of iterations of the bayesian inference algorithm
        for parameter estimation (default: 5000).
    warmup : int
        Required when when method=BAYES. The number of warmup iterations of the bayesian inference
        algorithm for parameter estimation (default: 2000).

    Returns
    -------
    Julia.Extremes.AbstractExtremeValueModel
        Fitted Julia model.
    """
    jl_y = py_list_to_jl_vector(y)
    locationcov, logscalecov, shapecov = (
        jl_variable_fit_parameters(location_cov),
        jl_variable_fit_parameters(scale_cov),
        jl_variable_fit_parameters(shape_cov),
    )

    if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
        if method == "ML":
            distm = "gevfit"
        elif method == "PWM":
            distm = "gevfitpwm"
        elif method == "BAYES":
            distm = "gevfitbayes"
    elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
        if method == "ML":
            distm = "gumbelfit"
        elif method == "PWM":
            distm = "gumbelfitpwm"
        elif method == "BAYES":
            distm = "gumbelfitbayes"
    elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
        if method == "ML":
            distm = "gpfit"
        elif method == "PWM":
            distm = "gpfitpwm"
        elif method == "BAYES":
            distm = "gpfitbayes"
    else:
        raise ValueError(
            f"Fitting distribution {dist} or method {method} not recognized"
        )

    args_per_func = {
        "gevfit": {
            "locationcov": locationcov,
            "logscalecov": logscalecov,
            "shapecov": shapecov,
        },
        "gevfitpwm": {},
        "gevfitbayes": {
            "locationcov": locationcov,
            "logscalecov": logscalecov,
            "shapecov": shapecov,
            "niter": niter,
            "warmup": warmup,
        },
        "gumbelfit": {"locationcov": locationcov, "logscalecov": logscalecov},
        "gumbelfitpwm": {},
        "gumbelfitbayes": {
            "locationcov": locationcov,
            "logscalecov": logscalecov,
            "niter": niter,
            "warmup": warmup,
        },
        "gpfit": {"logscalecov": logscalecov, "shapecov": shapecov},
        "gpfitpwm": {},
        "gpfitbayes": {
            "logscalecov": logscalecov,
            "shapecov": shapecov,
            "niter": niter,
            "warmup": warmup,
        },
    }
    args = args_per_func.get(distm)

    try:
        return getattr(Extremes, distm)(jl_y, **args)

    except JuliaError:
        warnings.warn(
            f"There was an error in fitting the data to a {dist} distribution using {method}. "
            "Returned parameters are numpy.nan.",
            UserWarning,
        )

        return None


def fit(
    ds: xr.Dataset,
    locationcov: list[str] = [],
    scalecov: list[str] = [],
    shapecov: list[str] = [],
    vars: list[str] = [],
    dist: str | scipy.stats.rv_continuous = "genextreme",
    method: str = "ML",
    dim: str = "time",
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
) -> xr.Dataset:
    r"""Fit an array to a univariate distribution along a given dimension.

    Parameters
    ----------
    ds : xr.DataSet
        Xarray Dataset containing the data to be fitted.
    locationcov : list[str]
        List of names of the covariates for the location parameter.
        Have to be names of data variables / coordinates in the original data.
    scalecov : list[str]
        List of names of the covariates for the scale parameter.
        Have to be names of data variables / coordinates in the original data.
    shapecov : list[str]
        List of names of the covariates for the shape parameter.
        Have to be names of data variables / coordinates in the original data.
    vars : list[str]
        List of variables to be fitted.
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

    Returns
    -------
    xr.Dataset
        Dataset of fitted distribution parameters and confidence interval values.

    Notes
    -----
    Coordinates for which all values are NaNs will be dropped before fitting the distribution. If the array still
    contains NaNs or has less valid values than the number of parameters for that distribution,
    the distribution parameters will be returned as NaNs.
    """
    vars = vars or ds.data_vars
    method = method.upper()
    _check_fit_params(
        dist,
        method,
        locationcov,
        scalecov,
        shapecov,
        confidence_level,
        ds,
        vars,
    )
    dist_params = _get_params(dist, shapecov, locationcov, scalecov)

    # Covariates
    locationcov_data = [ds[covariate] for covariate in locationcov]
    scalecov_data = [ds[covariate] for covariate in scalecov]
    shapecov_data = [ds[covariate] for covariate in shapecov]

    result_params = xr.Dataset()
    result_lower = xr.Dataset()
    result_upper = xr.Dataset()

    for data_var in vars:
        args = [ds[data_var]] + locationcov_data + scalecov_data + shapecov_data
        results = xr.apply_ufunc(
            _fitfunc_param_cint,
            *args,
            input_core_dims=[[dim]] * len(args),
            output_core_dims=[["dparams"], ["dparams"], ["dparams"]],
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
            output_dtypes=[float, float, float],
            kwargs=dict(
                dist=dist,
                nparams=len(dist_params),
                method=method,
                n_loccov=len(locationcov),
                n_scalecov=len(scalecov),
                n_shapecov=len(shapecov),
                niter=niter,
                warmup=warmup,
                confidence_level=confidence_level,
            ),
            dask_gufunc_kwargs={"output_sizes": {"dparams": len(dist_params)}},
        )
        result_params = xr.merge([result_params, results[0]])
        result_lower = xr.merge([result_lower, results[1]])
        result_upper = xr.merge([result_upper, results[2]])

    cint_lower_data = result_lower.rename(
        {var: var + "_lower" for var in result_lower.data_vars}
    )
    cint_upper_data = result_upper.rename(
        {var: var + "_upper" for var in result_upper.data_vars}
    )
    data = xr.merge([result_params, cint_lower_data, cint_upper_data])

    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "dparams" for d in ds.dims]
    out = data.assign_coords(dparams=dist_params).transpose(*dims)

    out.attrs = prefix_attrs(
        ds.attrs, ["standard_name", "long_name", "units", "description"], "original_"
    )
    dist = get_dist(dist)
    attrs = dict(
        long_name=f"{dist.name} parameters",
        dist=dist.name,
        method=METHOD_NAMES[method].capitalize(),
        confidence_level=confidence_level,
    )
    out.attrs.update(attrs)
    return out


def _fitfunc_param_cint(
    *arg,
    dist,
    nparams,
    method,
    n_loccov: int,
    n_scalecov: int,
    n_shapecov: int,
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
):
    r"""Fit a univariate distribution to an array using specified covariate data.

    Parameters
    ----------
    arg : list
        Input list containing the data to be fitted and the covariates.
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
    scalecov_data : list[list]
        Nested list containing the data for the scale covariates. Each inner list corresponds to a specific
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
    param_type : str
        The type of parameter to be estimated (e.g., "location", "scale", "shape").

    Returns
    -------
    params : list
        A list of fitted distribution parameters.
    """
    arr = arg[0]

    locationcov_data = arg[1 : n_loccov + 1]
    scalecov_data = arg[n_loccov + 1 : n_loccov + n_scalecov + 1]
    shapecov_data = arg[
        n_loccov + n_scalecov + 1 : n_loccov + n_scalecov + n_shapecov + 1
    ]

    nan_mask = create_nan_mask([arr], locationcov_data, scalecov_data, shapecov_data)

    locationcov_data_pruned = remove_nan(nan_mask, locationcov_data)
    scalecov_data_pruned = remove_nan(nan_mask, scalecov_data)
    shapecov_data_pruned = remove_nan(nan_mask, shapecov_data)
    arr_pruned = remove_nan(nan_mask, [arr])[0]

    # Sanity check
    if len(arr_pruned) <= nparams:
        warnings.warn(
            "The fitting data contains fewer entries than the number of parameters for the given distribution. "
            "Returned parameters are numpy.nan.",
            UserWarning,
        )
        return tuple(return_nan(nparams))

    jl_model = _fit_model(
        arr_pruned,
        dist=dist,
        method=method,
        location_cov=locationcov_data_pruned,
        scale_cov=scalecov_data_pruned,
        shape_cov=shapecov_data_pruned,
        niter=niter,
        warmup=warmup,
    )

    if jl_model is None:
        param_list = return_nan(nparams)
    else:
        param_list = param_cint(
            jl_model, confidence_level=confidence_level, method=method
        )

    if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
        shape_pos = 2 + n_loccov + n_scalecov
        param_list = change_sign_param(param_list, shape_pos, n_shapecov + 1)

    params = [
        exponentiate_logscale(
            params_,
            dist=dist,
            n_loccov=n_loccov,
            n_scalecov=n_scalecov,
        )
        for params_ in param_list
    ]  # because Extremes.jl gives log(scale)

    if dist == "genextreme" or str(type(dist)) == DIST_NAMES["genextreme"]:
        params = np.roll(params, 1 + n_shapecov, axis=1)  # to have [shape, loc, scale]
    else:
        pass

    return tuple(params)


#


def return_level(
    ds: xr.Dataset,
    locationcov: list[str] = [],
    scalecov: list[str] = [],
    shapecov: list[str] = [],
    vars: list[str] = [],
    dist: str | scipy.stats.rv_continuous = "genextreme",
    method: str = "ML",
    dim: str = "time",
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
    return_period: float = 100,
    threshold_pareto=None,
    nobs_pareto=None,
    nobsperblock_pareto=None,
) -> xr.Dataset:
    r"""Compute the return level corresponding to a return period from a given distribution.

    Parameters
    ----------
    ds : xr.DataSet
        Xarray Dataset containing the data for return level calculations.
    locationcov : list[str]
        List of names of the covariates for the location parameter.
        Have to be names of data variables / coordinates in the original data.
    scalecov : list[str]
        List of names of the covariates for the scale parameter.
        Have to be names of data variables / coordinates in the original data.
    shapecov : list[str]
        List of names of the covariates for the shape parameter.
        Have to be names of data variables / coordinates in the original data.
    vars : list[str]
        List of variables to be fitted.
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
    return_period : float
        Return period used to compute the return level.
    threshold_pareto : float
        Threshold for when the pareto function is used.
    nobs_pareto : int,
        Number of total observation for when the pareto function is used..
    nobsperblock_pareto : int,
        Number of observation per block for when the pareto function is used..

    Returns
    -------
    xr.Dataset
        Dataset of with the return level and the confidence interval values.

    Notes
    -----
    Coordinates for which all values are NaNs will be dropped before fitting the distribution. If the array still
    contains NaNs or has less valid values than the number of parameters for that distribution,
    the distribution parameters will be returned as NaNs.
    """
    vars = vars or ds.data_vars
    method = method.upper()
    _check_fit_params(
        dist,
        method,
        locationcov,
        scalecov,
        shapecov,
        confidence_level,
        ds,
        vars,
        return_period=return_period,
        return_type="returnlevel",
        threshold_pareto=threshold_pareto,
        nobs_pareto=nobs_pareto,
        nobsperblock_pareto=nobsperblock_pareto,
    )

    stationary = len(locationcov) == 0 and len(scalecov) == 0 and len(shapecov) == 0
    return_level_dim = ["return_level"] if stationary else ds[dim].values

    dist_params = _get_params(dist, shapecov, locationcov, scalecov)

    # Covariates
    locationcov_data = [ds[covariate] for covariate in locationcov]
    scalecov_data = [ds[covariate] for covariate in scalecov]
    shapecov_data = [ds[covariate] for covariate in shapecov]

    result_return = xr.Dataset()
    result_lower = xr.Dataset()
    result_upper = xr.Dataset()

    for data_var in vars:
        args = [ds[data_var]] + locationcov_data + scalecov_data + shapecov_data
        results = xr.apply_ufunc(
            _fitfunc_return_level,
            *args,
            input_core_dims=[[dim]] * len(args),
            output_core_dims=[["return_level"], ["return_level"], ["return_level"]],
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
            output_dtypes=[float, float, float],
            kwargs=dict(
                dist=dist,
                nparams=len(dist_params),
                method=method,
                main_dim_length=len(return_level_dim),
                n_loccov=len(locationcov),
                n_scalecov=len(scalecov),
                n_shapecov=len(shapecov),
                niter=niter,
                warmup=warmup,
                confidence_level=confidence_level,
                return_period=return_period,
                threshold_pareto=threshold_pareto,
                nobs_pareto=nobs_pareto,
                nobsperblock_pareto=nobsperblock_pareto,
            ),
            dask_gufunc_kwargs={
                "output_sizes": {"return_level": len(return_level_dim)}
            },
        )
        result_return = xr.merge([result_return, results[0]])
        result_lower = xr.merge([result_lower, results[1]])
        result_upper = xr.merge([result_upper, results[2]])

    cint_lower_data = result_lower.rename(
        {var: var + "_lower" for var in result_lower.data_vars}
    )
    cint_upper_data = result_upper.rename(
        {var: var + "_upper" for var in result_upper.data_vars}
    )
    data = xr.merge([result_return, cint_lower_data, cint_upper_data])

    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "return_level" for d in ds.dims]
    out = data.assign_coords(return_level=return_level_dim)
    out = out.transpose(*dims)
    out.attrs = prefix_attrs(
        ds.attrs, ["standard_name", "long_name", "units", "description"], "original_"
    )
    dist = get_dist(dist)
    attrs = dict(
        long_name=f"Return level estimation",
        dist=dist.name,
        method=METHOD_NAMES[method].capitalize(),
        return_period=return_period,
        confidence_level=confidence_level,
    )
    out.attrs.update(attrs)
    return out


def _fitfunc_return_level(
    *arg,
    dist,
    method,
    nparams,
    main_dim_length,
    n_loccov: int,
    n_scalecov: int,
    n_shapecov: int,
    niter: int,
    warmup: int,
    confidence_level: float = 0.95,
    return_period: float = 100,
    threshold_pareto=None,
    nobs_pareto=None,
    nobsperblock_pareto=None,
):
    r"""Fit a univariate distribution to an array using specified covariate data.

    Parameters
    ----------
    arg : list
        Input list containing the data to be fitted and the covariates.
    dist : str or rv_continuous
        The univariate distribution to fit, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    method : str
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    locationcov_data : list[list]
        Nested list containing the data for the location covariates. Each inner list corresponds to a specific
        covariate.
    scalecov_data : list[list]
        Nested list containing the data for the scale covariates. Each inner list corresponds to a specific
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
    param_type : str
        The type of parameter to be estimated (e.g., "location", "scale", "shape").

    Returns
    -------
    params : list
        A list of fitted distribution parameters.
    """
    arr = arg[0]

    locationcov_data = arg[1 : n_loccov + 1]
    scalecov_data = arg[n_loccov + 1 : n_loccov + n_scalecov + 1]
    shapecov_data = arg[
        n_loccov + n_scalecov + 1 : n_loccov + n_scalecov + n_shapecov + 1
    ]

    nan_mask = create_nan_mask([arr], locationcov_data, scalecov_data, shapecov_data)

    locationcov_data_pruned = remove_nan(nan_mask, locationcov_data)
    scalecov_data_pruned = remove_nan(nan_mask, scalecov_data)
    shapecov_data_pruned = remove_nan(nan_mask, shapecov_data)
    arr_pruned = remove_nan(nan_mask, [arr])[0]

    stationary = not (
        locationcov_data_pruned or scalecov_data_pruned or shapecov_data_pruned
    )

    # Sanity check
    if len(arr_pruned) <= nparams:
        warnings.warn(
            "The fitting data contains fewer entries than the number of parameters for the given distribution. "
            "Returned parameters are numpy.nan.",
            UserWarning,
        )
        return tuple(return_nan(main_dim_length))

    jl_model = _fit_model(
        arr_pruned,
        dist=dist,
        method=method,
        location_cov=locationcov_data_pruned,
        scale_cov=scalecov_data_pruned,
        shape_cov=shapecov_data_pruned,
        niter=niter,
        warmup=warmup,
    )

    if jl_model is None:
        return tuple(return_nan(main_dim_length))
    else:
        return_level_list = return_level_cint(
            jl_model,
            confidence_level=confidence_level,
            return_period=return_period,
            dist=dist,
            threshold_pareto=threshold_pareto,
            nobs_pareto=nobs_pareto,
            nobsperblock_pareto=nobsperblock_pareto,
            method=method,
        )

    if not stationary:
        return_level_list = recover_nan(nan_mask, return_level_list)

    return tuple(return_level_list)


def _get_params(
    dist: str, shapecov: list[str], locationcov: list[str], scalecov: list[str]
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
    scalecov : list[str]
        List of covariate names for the scale parameter.

    Returns
    -------
    list
        A one-dimensional list of parameter names corresponding to the distribution and covariates.

    Examples
    --------
    >>> scalecov = (["max_temp_yearly"],)
    >>> shapecov = (["qmax_yearly"],)
    >>> get_params("genextreme", shapecov, [], scalecov)
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
        new_param_names = insert_covariates(new_param_names, scalecov, "scale")
        new_param_names = insert_covariates(new_param_names, shapecov, "shape")
        return new_param_names

    elif dist == "gumbel_r" or str(type(dist)) == DIST_NAMES["gumbel_r"]:
        param_names = ["loc", "scale"]
        new_param_names = insert_covariates(param_names, locationcov, "loc")
        new_param_names = insert_covariates(new_param_names, scalecov, "scale")
        return new_param_names
    elif dist == "genpareto" or str(type(dist)) == DIST_NAMES["genpareto"]:
        param_names = ["scale", "shape"]
        new_param_names = insert_covariates(param_names, scalecov, "scale")
        new_param_names = insert_covariates(new_param_names, shapecov, "shape")
        return new_param_names


def _check_fit_params(
    dist: str,
    method: str,
    locationcov: list[str],
    scalecov: list[str],
    shapecov: list[str],
    confidence_level: float,
    ds: xr.Dataset,
    vars: list[str],
    return_period: float = 1,
    return_type=None,
    threshold_pareto=None,
    nobs_pareto=None,
    nobsperblock_pareto=None,
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
    scalecov : list[str]
        List of covariate names for the scale parameter.
    shapecov : list[str]
        List of covariate names for the shape parameter.
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
    return_type : str
        Specifies whether to return the estimated parameters ('param') or the return level ('returnlevel').
    threshold_pareto : float
        Threshold used to compute the returnlevel with the pareto distribution .
    nobs_pareto : int,
        Number of total observation used to compute the returnlevel with the pareto distribution .
    nobsperblock_pareto : int,
        Number of observation per block used to compute the returnlevel with the pareto distribution

    Raises
    ------
    ValueError
        If the combination of arguments is incoherent or invalid for the specified distribution
        and fitting method.
    """
    # Method and distribution names have to be among the recognized ones
    if method not in METHOD_NAMES:
        raise ValueError(f"Unrecognized method: {method}")

    if dist not in DIST_NAMES.keys() and str(type(dist)) not in DIST_NAMES.values():
        raise ValueError(f"Unrecognized distribution: {dist}")

    # PWM estimation does not work in non-stationary context
    if method == "PWM" and (
        len(locationcov) != 0 or len(scalecov) != 0 or len(shapecov) != 0
    ):
        covariates = locationcov + scalecov + shapecov
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

    # Check
    if (
        return_type == "returnlevel"
        and dist == "genpareto"
        and (
            threshold_pareto is None
            or nobs_pareto is None
            or nobsperblock_pareto is None
        )
    ):
        raise ValueError(
            "'threshold_pareto', 'nobs_pareto', and 'nobsperblock_pareto' must be defined when using dist 'genpareto'."
        )

    # Confidence level must be between 0 and 1
    if confidence_level >= 1 or confidence_level <= 0:
        raise ValueError(
            f"Confidence level must be strictly smaller than 1 and strictly larger than 0"
        )

    # Vars has to contain data variables present in the Dataset
    for var in vars:
        if var not in ds.data_vars:
            raise ValueError(
                f"{var} is not a variable in the Dataset. "
                f"Dataset's variables are: {list(ds.data_vars)}"
            )

    # Return period has to be strictly positive
    if return_period <= 0:
        raise ValueError(
            f"Return period has to be strictly larger than 0. "
            f"Current return period value is {return_period}"
        )