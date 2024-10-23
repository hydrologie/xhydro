"""Local frequency analysis functions and utilities."""

import datetime
from typing import Optional, Union

import numpy as np
import statsmodels
import xarray as xr
import xclim.indices.stats
from scipy.stats.mstats import plotting_positions
from statsmodels.tools import eval_measures

__all__ = [
    "criteria",
    "fit",
    "parametric_quantiles",
]


def fit(
    ds,
    distributions: list[str] | None = None,
    min_years: int | None = None,
    method: str = "ML",
) -> xr.Dataset:
    """Fit multiple distributions to data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the data to fit. All variables will be fitted.
    distributions : list of str, optional
        List of distribution names as defined in `scipy.stats`. See https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions.
        Defaults to ["genextreme", "pearson3", "gumbel_r", "expon"].
    min_years : int, optional
        Minimum number of years required for a distribution to be fitted.
    method : str
        Fitting method. Defaults to "ML" (maximum likelihood).

    Returns
    -------
    xr.Dataset
        Dataset containing the parameters of the fitted distributions, with a new dimension `scipy_dist` containing the distribution names.

    Notes
    -----
    In order to combine the parameters of multiple distributions, the size of the `dparams` dimension is set to the
    maximum number of unique parameters between the distributions.
    """
    distributions = distributions or ["genextreme", "pearson3", "gumbel_r", "expon"]
    out = []
    for v in ds.data_vars:
        p = []
        for d in distributions:
            p.append(  # noqa: PERF401
                xclim.indices.stats.fit(
                    ds[v].chunk({"time": -1}), dist=d, method=method
                )
                .assign_coords(scipy_dist=d)
                .expand_dims("scipy_dist")
            )
        params = xr.concat(p, dim="scipy_dist")

        # Reorder dparams to match the order of the parameters across all distributions, since subsequent operations rely on this.
        p_order = sorted(set(params.dparams.values).difference(["loc", "scale"])) + [
            "loc",
            "scale",
        ]
        params = params.sel(dparams=p_order)

        if min_years is not None:
            params = params.where(ds[v].notnull().sum("time") >= min_years)
        params.attrs["scipy_dist"] = distributions
        params.attrs["description"] = "Parameters of the distributions"
        params.attrs["long_name"] = "Distribution parameters"
        params.attrs["min_years"] = min_years
        out.append(params)

    out = xr.merge(out)
    out = out.chunk({"dparams": -1})
    out.attrs = ds.attrs

    return out


def parametric_quantiles(
    p: xr.Dataset, t: float | list[float], mode: str = "max"
) -> xr.Dataset:
    """Compute quantiles from fitted distributions.

    Parameters
    ----------
    p : xr.Dataset
        Dataset containing the parameters of the fitted distributions.
        Must have a dimension `dparams` containing the parameter names and a dimension `scipy_dist` containing the distribution names.
    t : float or list of float
        Return period(s) in years.
    mode : {'max', 'min'}
        Whether the return period is the probability of exceedance (max) or non-exceedance (min).

    Returns
    -------
    xr.Dataset
        Dataset containing the quantiles of the distributions.
    """
    distributions = list(p["scipy_dist"].values)

    t = np.atleast_1d(t)
    if mode == "max":
        q = 1 - 1.0 / t
    elif mode == "min":
        q = 1.0 / t
    else:
        raise ValueError(f"'mode' must be 'max' or 'min', got '{mode}'.")

    out = []
    for v in p.data_vars:
        quantiles = []
        for d in distributions:
            dist_obj = xclim.indices.stats.get_dist(d)
            shape_params = [] if dist_obj.shapes is None else dist_obj.shapes.split(",")
            dist_params = shape_params + ["loc", "scale"]
            da = p[v].sel(scipy_dist=d, dparams=dist_params).transpose("dparams", ...)
            da.attrs["scipy_dist"] = d
            qt = (
                xclim.indices.stats.parametric_quantile(da, q=q)
                .rename({"quantile": "return_period"})
                .assign_coords(scipy_dist=d, return_period=t)
                .expand_dims("scipy_dist")
            )
            quantiles.append(qt)
        quantiles = xr.concat(quantiles, dim="scipy_dist")

        # Add the quantile as a new coordinate
        da_q = xr.DataArray(q, dims="return_period", coords={"return_period": t})
        da_q.attrs["long_name"] = (
            "Probability of exceedance"
            if mode == "max"
            else "Probability of non-exceedance"
        )
        da_q.attrs["description"] = (
            "Parametric distribution quantiles for the given return period."
        )
        da_q.attrs["mode"] = mode
        quantiles = quantiles.assign_coords(p_quantile=da_q)

        quantiles.attrs["scipy_dist"] = distributions
        quantiles.attrs["description"] = (
            f"Return period ({mode}) estimated with statistic distributions"
        )
        quantiles.attrs["long_name"] = "Return period"
        quantiles.attrs["mode"] = mode
        out.append(quantiles)

    out = xr.merge(out)
    out.attrs = p.attrs

    return out


def criteria(ds: xr.Dataset, p: xr.Dataset) -> xr.Dataset:
    """Compute information criteria (AIC, BIC, AICC) from fitted distributions, using the log-likelihood.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the yearly data that was fitted.
    p : xr.Dataset
        Dataset containing the parameters of the fitted distributions.
        Must have a dimension `dparams` containing the parameter names and a dimension `scipy_dist` containing the distribution names.

    Returns
    -------
    xr.Dataset
        Dataset containing the information criteria for the distributions.
    """

    def _get_criteria_1d(da, params, dist):
        da = np.atleast_2d(da).T
        params = np.atleast_2d(params).T

        llf = np.nansum(dist.logpdf(da, *params), axis=0)  # log-likelihood
        nobs = np.sum(np.isfinite(da), axis=0)
        df_modelwc = len(p)
        dof_eff = nobs - df_modelwc - 1.0

        aic = eval_measures.aic(llf=llf, nobs=nobs, df_modelwc=len(p))
        bic = eval_measures.bic(llf=llf, nobs=nobs, df_modelwc=len(p))
        # Custom AICC, because the one in statsmodels does not support multiple dimensions
        aicc = np.where(
            dof_eff > 0, -2.0 * llf + 2.0 * df_modelwc * nobs / dof_eff, np.nan
        )

        return np.stack([aic, bic, aicc], axis=1)

    out = []
    distributions = list(p["scipy_dist"].values)

    common_vars = list(set(ds.data_vars).intersection(p.data_vars))
    for v in common_vars:
        c = []
        for d in distributions:
            dist_obj = xclim.indices.stats.get_dist(d)
            shape_params = [] if dist_obj.shapes is None else dist_obj.shapes.split(",")
            dist_params = shape_params + ["loc", "scale"]
            da = ds[v].transpose("time", ...)
            params = (
                p[v].sel(scipy_dist=d, dparams=dist_params).transpose("dparams", ...)
            )

            if len(da.dims) == 1:
                crit = xr.apply_ufunc(
                    _get_criteria_1d,
                    da.expand_dims("tmp"),
                    params.expand_dims("tmp"),
                    kwargs={"dist": dist_obj},
                    input_core_dims=[["time"], ["dparams"]],
                    output_core_dims=[["criterion"]],
                    dask_gufunc_kwargs=dict(output_sizes={"criterion": 3}),
                    dask="parallelized",
                ).squeeze("tmp")
            else:
                crit = xr.apply_ufunc(
                    _get_criteria_1d,
                    da,
                    params,
                    kwargs={"dist": dist_obj},
                    input_core_dims=[["time"], ["dparams"]],
                    output_core_dims=[["criterion"]],
                    dask_gufunc_kwargs=dict(output_sizes={"criterion": 3}),
                    dask="parallelized",
                )

            # Add attributes
            crit = crit.assign_coords(criterion=["aic", "bic", "aicc"]).expand_dims(
                "scipy_dist"
            )
            crit.attrs = p[v].attrs
            crit.attrs["history"] = (
                crit.attrs.get("history", "")
                + f", [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"criteria: computed AIC, BIC and AICC. - statsmodels version: {statsmodels.__version__}"
            )
            crit.attrs["description"] = (
                "Information criteria for the distribution parameters."
            )
            crit.attrs["long_name"] = "Information criteria"

            # Remove a few attributes that are not relevant anymore
            crit.attrs.pop("method", None)
            crit.attrs.pop("estimator", None)
            crit.attrs.pop("min_years", None)
            c.append(crit)

        c = xr.concat(c, dim="scipy_dist")
        out.append(c)

    out = xr.merge(out)
    out.attrs = ds.attrs

    return out


def _get_plotting_positions(data_array, alpha=0.4, beta=0.4, return_period=True):
    """Calculate plotting positions for data.

    Parameters
    ----------
    data_array : xarray DataArray
        Input data.
    alpha : float, optional
        Plotting position parameter, by default 0.4.
    beta : float, optional
        Plotting position parameter, by default 0.4.
    return_period : bool, optional
        If True, return periods instead of probabilities, by default True.

    Returns
    -------
    xarray DataArray
        The data with plotting positions assigned.

    Notes
    -----
    See scipy.stats.mstats.plotting_positions for typical values for alpha and beta. (0.4, 0.4) : approximately quantile unbiased (Cunnane).
    """
    data_copy = data_array.copy(deep=True)

    def vec_plotting_positions(vec_data, alpha=0.4, beta=0.4):
        """Calculate plotting positions for vectorized data.

        Parameters
        ----------
        vec_data : ndarray
            Input data, with time dimension first.
        alpha : float, optional
            Plotting position parameter.
        beta : float, optional
            Plotting position parameter.

        Returns
        -------
        ndarray
            Array with plotting positions assigned to valid data points,
            and NaNs assigned to invalid data points.
        """
        out = []
        if vec_data.ndim == 1:
            valid = ~np.isnan(vec_data)
            pp = plotting_positions(vec_data[valid], alpha, beta)

            out = np.full_like(vec_data.astype(float), np.nan)
            out[valid] = pp

        else:
            for data in vec_data:
                valid = ~np.isnan(data)
                pp = plotting_positions(data[valid], alpha, beta)

                pp_full = np.full_like(data.astype(float), np.nan)
                pp_full[valid] = pp

                out.append(pp_full)
        return out

    pp = xr.apply_ufunc(
        vec_plotting_positions,
        data_array,
        alpha,
        beta,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["time"]],
    )

    if return_period:
        pp = -1 / (pp - 1)

    for name in pp.data_vars:
        pp = pp.rename({name: name + "_pp"})
        pp[name] = data_copy[name]

    return pp


def _prepare_plots(params, xmin=1, xmax=10000, npoints=100, log=True):
    """Prepare x-values for plotting frequency analysis results.

    Parameters
    ----------
    params : xarray DataArray
        Input data.
    xmin : float, optional
        Minimum x value, by default 1.
    xmax : float, optional
        Maximum x value, by default 10000.
    npoints : int, optional
        Number of x values, by default 100.
    log : bool, optional
        If True, return log-spaced x values, by default True.

    Returns
    -------
    xarray DataArray
        The data with plotting positions assigned.
    """
    if log:
        x = np.logspace(np.log10(xmin), np.log10(xmax), npoints, endpoint=True)
    else:
        x = np.linspace(xmin, xmax, npoints)
    return parametric_quantiles(params, x)
