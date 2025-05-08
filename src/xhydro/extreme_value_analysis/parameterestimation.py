"""Parameter estimation functions for the extreme value analysis module."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.stats
import xarray as xr
from xclim.indices.stats import get_dist

try:
    from juliacall import JuliaError

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
    location_cov: list[list] | None = None,
    scale_cov: list[list] | None = None,
    shape_cov: list[list] | None = None,
    niter: int = 5000,
    warmup: int = 2000,
) -> Any | None:
    r"""
    Fit a distribution using the specified covariate data.

    Parameters
    ----------
    y : list[float]
        Data to be fitted.
    dist : str or rv_continuous
        Distribution, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    method : {"ML", "PWM", "BAYES}
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    location_cov : list[list]
        List of data lists to be used as covariates for the location parameter.
    scale_cov : list[list]
        List of data lists to be used as covariates for the scale parameter.
    shape_cov : list[list]
        List of data lists to be used as covariates for the shape parameter.l.
    niter : int
        Required when method=BAYES. The number of iterations of the bayesian inference algorithm
        for parameter estimation (default: 5000).
    warmup : int
        Required when method=BAYES. The number of warmup iterations of the bayesian inference
        algorithm for parameter estimation (default: 2000).

    Returns
    -------
    Julia.Extremes.AbstractExtremeValueModel
        Fitted Julia model.
    """
    location_cov = location_cov or []
    scale_cov = scale_cov or []
    shape_cov = shape_cov or []

    jl_y = py_list_to_jl_vector(y)
    locationcov, logscalecov, shapecov = (
        jl_variable_fit_parameters(location_cov),
        jl_variable_fit_parameters(scale_cov),
        jl_variable_fit_parameters(shape_cov),
    )

    dist_methods = {
        "genextreme": {"ML": "gevfit", "PWM": "gevfitpwm", "BAYES": "gevfitbayes"},
        "gumbel_r": {
            "ML": "gumbelfit",
            "PWM": "gumbelfitpwm",
            "BAYES": "gumbelfitbayes",
        },
        "genpareto": {"ML": "gpfit", "PWM": "gpfitpwm", "BAYES": "gpfitbayes"},
    }

    distm = dist_methods.get(dist, {}).get(method)
    if not distm:
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


def fit(
    ds: xr.Dataset,
    locationcov: list[str] | None = None,
    scalecov: list[str] | None = None,
    shapecov: list[str] | None = None,
    variables: list[str] | None = None,
    dist: str | scipy.stats.rv_continuous = "genextreme",
    method: str = "ML",
    dim: str = "time",
    confidence_level: float = 0.95,
    niter: int = 5000,
    warmup: int = 2000,
) -> xr.Dataset:
    r"""
    Fit an array to a univariate distribution along a given dimension.

    Parameters
    ----------
    ds : xr.DataSet
        Xarray Dataset containing the data to be fitted.
    locationcov : list[str]
        List of names of the covariates for the location parameter.
    scalecov : list[str]
        List of names of the covariates for the scale parameter.
    shapecov : list[str]
        List of names of the covariates for the shape parameter.
    variables : list[str]
        List of variables to be fitted.
    dist : str or rv_continuous distribution object
        Name of the univariate distribution or the distribution object itself.
        Supported distributions are genextreme, gumbel_r, genpareto.
    method : {"ML", "PWM", "BAYES}
        Fitting method, either maximum likelihood (ML), probability weighted moments (PWM) or bayesian (BAYES).
    dim : str
        Specifies the dimension along which the fit will be performed (default: "time").
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
    niter : int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).

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
    locationcov = locationcov or []
    scalecov = scalecov or []
    shapecov = shapecov or []

    if any(var.chunks for var in ds.variables.values()):
        warnings.warn(
            "Dataset contains chunks. It is recommended to use scheduler='processes' to compute the results.",
            UserWarning,
        )

    variables = variables or ds.data_vars
    method = method.upper()
    _check_fit_params(
        dist,
        method,
        locationcov,
        scalecov,
        shapecov,
        confidence_level,
        ds,
        variables,
    )
    dist_params = _get_params(dist, shapecov, locationcov, scalecov)

    # Covariates
    locationcov_data = [ds[covariate] for covariate in locationcov]
    scalecov_data = [ds[covariate] for covariate in scalecov]
    shapecov_data = [ds[covariate] for covariate in shapecov]

    result_params = xr.Dataset()
    result_lower = xr.Dataset()
    result_upper = xr.Dataset()

    dist_scp = get_dist(dist)

    attrs_dist = dict(
        dist=dist_scp.name,
        method=METHOD_NAMES[method].capitalize(),
    )

    for data_var in variables:
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
        par, low, upp = results

        par.attrs.update(dict(long_name=f"Distribution parameters") | attrs_dist)
        low.attrs.update(
            dict(
                long_name="Lower limit of confidence interval for the distribution parameters",
                confidence_level=confidence_level,
            )
            | attrs_dist
        )
        upp.attrs.update(
            dict(
                long_name="Upper limit of confidence interval for the distribution parameters",
                confidence_level=confidence_level,
            )
            | attrs_dist
        )

        result_params = xr.merge([result_params, par])
        result_lower = xr.merge([result_lower, low])
        result_upper = xr.merge([result_upper, upp])

    cint_lower_data = result_lower.rename(
        {var: f"{var}_lower" for var in result_lower.data_vars}
    )
    cint_upper_data = result_upper.rename(
        {var: f"{var}_upper" for var in result_upper.data_vars}
    )
    data = xr.merge([result_params, cint_lower_data, cint_upper_data])

    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "dparams" for d in ds.dims]
    out = data.assign_coords(dparams=dist_params).transpose(*dims)

    out.attrs = ds.attrs
    return out


def _fitfunc_param_cint(
    *arg,
    dist: str | scipy.stats.rv_continuous,
    nparams: int,
    method: str,
    n_loccov: int,
    n_scalecov: int,
    n_shapecov: int,
    niter: int = 5000,
    warmup: int = 2000,
    confidence_level: float = 0.95,
) -> tuple:
    r"""
    Fit a univariate distribution to an array using specified covariate data.

    Parameters
    ----------
    arg : list
        Input list containing the data to be fitted and the covariates.
    dist : str or rv_continuous
        The univariate distribution to fit, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    nparams : int
        The number of parameters for the distribution.
    method : {"ML", "PWM", "BAYES}
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    n_loccov : list[list]
        Nested list containing the data for the location covariates. Each inner list corresponds to a specific covariate.
    n_scalecov : list[list]
        Nested list containing the data for the scale covariates. Each inner list corresponds to a specific covariate.
    n_shapecov : list[list]
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
    tuple
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


def return_level(
    ds: xr.Dataset,
    locationcov: list[str] | None = None,
    scalecov: list[str] | None = None,
    shapecov: list[str] | None = None,
    variables: list[str] | None = None,
    dist: str | scipy.stats.rv_continuous = "genextreme",
    method: str = "ML",
    dim: str = "time",
    confidence_level: float = 0.95,
    return_period: float = 100,
    niter: int = 5000,
    warmup: int = 2000,
    threshold_pareto: float | None = None,
    nobs_pareto: int | None = None,
    nobsperblock_pareto: int | None = None,
) -> xr.Dataset:
    r"""
    Compute the return level associated with a return period based on a given distribution.

    Parameters
    ----------
    ds : xr.DataSet
        Xarray Dataset containing the data for return level calculations.
    locationcov : list[str]
        List of names of the covariates for the location parameter.
    scalecov : list[str]
        List of names of the covariates for the scale parameter.
    shapecov : list[str]
        List of names of the covariates for the shape parameter.
    variables : list[str]
        List of variables to be fitted.
    dist : str or rv_continuous distribution object
        Name of the univariate distribution or the distribution object itself.
        Supported distributions are genextreme, gumbel_r, genpareto.
    method : {"ML", "PWM", "BAYES}
        Fitting method, either maximum likelihood (ML), probability weighted moments (PWM) or bayesian (BAYES).
    dim : str
        Specifies the dimension along which the fit will be performed (default: "time").
    confidence_level : float
        The confidence level for the confidence interval of each parameter.
    return_period : float
        Return period used to compute the return level.
    niter : int
        The number of iterations of the bayesian inference algorithm for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations of the bayesian inference algorithm for parameter estimation (default: 2000).
    threshold_pareto : float
        The value above which the Pareto distribution is applied.
    nobs_pareto : int
        The total number of observations used when applying the Pareto distribution.
    nobsperblock_pareto : int
        The number of observations per block when applying the Pareto distribution.

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
    locationcov = locationcov or []
    scalecov = scalecov or []
    shapecov = shapecov or []

    if any(var.chunks for var in ds.variables.values()):
        warnings.warn(
            "Dataset contains chunks. It is recommended to use scheduler='processes' to compute the results.",
            UserWarning,
        )

    variables = variables or ds.data_vars
    method = method.upper()
    _check_fit_params(
        dist,
        method,
        locationcov,
        scalecov,
        shapecov,
        confidence_level,
        ds,
        variables,
        return_period=return_period,
        return_type="returnlevel",
        threshold_pareto=threshold_pareto,
        nobs_pareto=nobs_pareto,
        nobsperblock_pareto=nobsperblock_pareto,
    )

    stationary = len(locationcov) == 0 and len(scalecov) == 0 and len(shapecov) == 0
    return_level_dim = ds[dim].values if not stationary else ["return_period"]

    dist_params = _get_params(dist, shapecov, locationcov, scalecov)

    # Covariates
    locationcov_data = [ds[covariate] for covariate in locationcov]
    scalecov_data = [ds[covariate] for covariate in scalecov]
    shapecov_data = [ds[covariate] for covariate in shapecov]

    result_return = xr.Dataset()
    result_lower = xr.Dataset()
    result_upper = xr.Dataset()

    dist_scp = get_dist(dist)

    attrs_dist = dict(
        dist=dist_scp.name,
        method=METHOD_NAMES[method].capitalize(),
    )

    for data_var in variables:
        args = [ds[data_var]] + locationcov_data + scalecov_data + shapecov_data
        results = xr.apply_ufunc(
            _fitfunc_return_level,
            *args,
            input_core_dims=[[dim]] * len(args),
            output_core_dims=(
                [["return_period"], ["return_period"], ["return_period"]]
                if stationary
                else [[dim], [dim], [dim]]
            ),
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
                "output_sizes": {
                    "return_period" if stationary else dim: len(return_level_dim)
                }
            },
        )

        par, low, upp = results

        par.attrs.update(dict(long_name=f"Distribution parameters") | attrs_dist)
        low.attrs.update(
            dict(
                long_name="Lower limit of confidence interval for the distribution parameters",
                confidence_level=confidence_level,
            )
            | attrs_dist
        )
        upp.attrs.update(
            dict(
                long_name="Upper limit of confidence interval for the distribution parameters",
                confidence_level=confidence_level,
            )
            | attrs_dist
        )

        result_return = xr.merge([result_return, results[0]])
        result_lower = xr.merge([result_lower, results[1]])
        result_upper = xr.merge([result_upper, results[2]])

    cint_lower_data = result_lower.rename(
        {var: f"{var}_lower" for var in result_lower.data_vars}
    )
    cint_upper_data = result_upper.rename(
        {var: f"{var}_upper" for var in result_upper.data_vars}
    )

    data = xr.merge([result_return, cint_lower_data, cint_upper_data])

    data = data.assign_coords({"return_period": [return_period]})

    data.attrs = ds.attrs

    return data


def _fitfunc_return_level(
    *arg,
    dist: str | scipy.stats.rv_continuous,
    method: str,
    nparams: int,
    main_dim_length: int,
    n_loccov: int,
    n_scalecov: int,
    n_shapecov: int,
    niter: int,
    warmup: int,
    confidence_level: float = 0.95,
    return_period: float = 100,
    threshold_pareto: float | None = None,
    nobs_pareto: int | None = None,
    nobsperblock_pareto: int | None = None,
) -> tuple:
    r"""
    Fit a univariate distribution to an array using specified covariate data.

    Parameters
    ----------
    arg : list
        Input list containing the data to be fitted and the covariates.
    dist : str or rv_continuous
        The univariate distribution to fit, either as a string or as a distribution object.
        Supported distributions include genextreme, gumbel_r, genpareto.
    method : {"ML", "PWM", "BAYES}
        The fitting method, which can be maximum likelihood (ML), probability weighted moments (PWM),
        or Bayesian inference (BAYES).
    nparams : int
        The number of parameters for the distribution.
    main_dim_length : int
        The length of the main dimension.
    n_loccov : list[list]
        Nested list containing the data for the location covariates. Each inner list corresponds to a specific
        covariate.
    n_scalecov : list[list]
        Nested list containing the data for the scale covariates. Each inner list corresponds to a specific
        covariate.
    n_shapecov : list[list]
        Nested list containing the data for the shape covariates. Each inner list corresponds to a specific
        covariate.
    niter : int
        The number of iterations for the Bayesian inference algorithm used for parameter estimation (default: 5000).
    warmup : int
        The number of warmup iterations for the Bayesian inference algorithm used for parameter estimation (default: 2000).
    confidence_level : float, optional
        The confidence level for the confidence interval of each parameter (default: 0.95).
    return_period : float
        The return period used to compute the return level.
    threshold_pareto : float
        The value above which the Pareto distribution is applied.
    nobs_pareto : int
        The total number of observations used when applying the Pareto distribution.
    nobsperblock_pareto : int
        The number of observations per block when applying the Pareto distribution.

    Returns
    -------
    tuple
        A tuple of fitted distribution parameters.
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
) -> list[str]:
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
    list of str
        A one-dimensional tuple of parameter names corresponding to the distribution and covariates.

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

    else:
        raise ValueError(f"Unrecognized distribution: {dist}")


def _check_fit_params(
    dist: str,
    method: str,
    locationcov: list[str],
    scalecov: list[str],
    shapecov: list[str],
    confidence_level: float,
    ds: xr.Dataset,
    variables: list[str],
    return_period: float = 1,
    return_type: str | None = None,
    threshold_pareto: float | None = None,
    nobs_pareto: int | None = None,
    nobsperblock_pareto: int | None = None,
) -> None:
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
    nobs_pareto : int
        Number of total observation used to compute the returnlevel with the pareto distribution .
    nobsperblock_pareto : int
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

    # Must contain data variables present in the Dataset
    for var in variables:
        if var not in ds.data_vars:
            raise ValueError(
                f"{var} is not a variable in the Dataset. "
                f"Dataset variables are: {list(ds.data_vars)}"
            )

    # Return period has to be strictly positive
    if return_period <= 0:
        raise ValueError(
            f"Return period has to be strictly larger than 0. "
            f"Current return period value is {return_period}"
        )
