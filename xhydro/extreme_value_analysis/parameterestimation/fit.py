from typing import Optional
import numpy as np
from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from xhydro.extreme_value_analysis.structures.conversions import *
from .bayesian import *
from .maximumlikelihood import *
from .probabilityweightedmoment import *
from xclim.indices.stats import get_dist
import scipy.stats
from xclim.core.formatting import prefix_attrs, update_history

METHOD_NAMES = {
    "ML": "maximum likelihood",
    "PWM": "probability weighted moments",
    "BAYES": "bayesian"
}

DIST_NAMES = ["genextreme", "gumbel_r", "genpareto"]

def fit(
    ds: xr.Dataset,
    dist: str | scipy.stats.rv_continuous = "genextreme",
    method: str = "ML",
    dim: str = "time",
    min_years: Optional[int] = None,
    ) -> xr.Dataset:

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
    x = np.ma.masked_invalid(arr).compressed()  # pylint: disable=no-member
    # Return NaNs if array is empty, which could happen at previous line if array only contained Nans
    if len(x) <= 1:
        return np.asarray([np.nan] * nparams)

    if method == "ML":
        # TODO: find cleaner way of checking dist type
        if str(type(dist)) == "<class 'scipy.stats._continuous_distns.genextreme_gen'>":
            param_list = gevfit_1(x).theta
            params = np.asarray(param_list)
        elif str(type(dist)) == "<class 'scipy.stats._continuous_distns.gumbel_r_gen'>":
            param_list = gumbelfit_1(x).theta
            param_list[1] = math.exp(param_list[1]) # because gumbelfit_1(x).theta gives us [loc, log(scale)]
            params = np.asarray(param_list)
        elif str(type(dist)) == "<class 'scipy.stats._continuous_distns.genpareto_gen'>":
            param_list = gpfit_1(x).theta
            params = np.asarray(param_list)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "PWM":
        if str(type(dist)) == "<class 'scipy.stats._continuous_distns.genextreme_gen'>":
            param_list = gevfitpwm_1(x).theta
            params = np.asarray(param_list)
        elif str(type(dist)) == "<class 'scipy.stats._continuous_distns.gumbel_r_gen'>":
            param_list = gumbelfitpwm_1(x).theta
            params = np.asarray(param_list)
        elif str(type(dist)) == "<class 'scipy.stats._continuous_distns.genpareto_gen'>":
            param_list = gpfitpwm_1(x).theta
            params = np.asarray(param_list)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")

    elif method == "BAYES":
        if str(type(dist)) == "<class 'scipy.stats._continuous_distns.genextreme_gen'>":
            fm = gevfitbayes_1(x, niter = 1000, warmup = 400)
            jl_fm = py_bayesian_aev_to_jl_aev(fm)
            param_list = [(interval[0] + interval[1])/2 for interval in jl_vector_to_py_list(jl.cint(jl_fm))]
            params = np.asarray(param_list)
        elif str(type(dist)) == "<class 'scipy.stats._continuous_distns.gumbel_r_gen'>":
            fm = gumbelfitbayes_1(x)
            jl_fm = py_bayesian_aev_to_jl_aev(fm)
            param_list = [(interval[0] + interval[1])/2 for interval in jl_vector_to_py_list(jl.cint(jl_fm))]
            params = np.asarray(param_list)
        elif str(type(dist)) == "<class 'scipy.stats._continuous_distns.genpareto_gen'>":
            fm = gpfitbayes_1(x)
            jl_fm = py_bayesian_aev_to_jl_aev(fm)
            param_list = [(interval[0] + interval[1])/2 for interval in jl_vector_to_py_list(jl.cint(jl_fm))]
            params = np.asarray(param_list)
        else:
            raise ValueError(f"Fitting distribution not recognized: {dist}")
    else:
        raise ValueError(f"Fitting method not recognized: {method}")
    return params

def _get_params(dist:str) -> list[str]:
    if dist == "genextreme":
        return ["loc", "scale", "shape"]
    elif dist == "gumbel_r":
        return ["loc", "scale"]
    elif dist == "genpareto":
        return ["scale", "shape"]
    else:
        raise ValueError(f"Unknown distribution: {dist}")

def _check_fit_params(dist: str, method: str):
    if method not in METHOD_NAMES:
        raise ValueError(f"Fitting method not recognized: {method}")

    if dist not in DIST_NAMES:
        raise ValueError(f"Fitting distribution not recognized: {dist}")
