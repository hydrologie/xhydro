from juliacall import convert as jl_convert
from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from xhydro.extreme_value_analysis.structures.dataitem import Variable
from xhydro.extreme_value_analysis.structures.conversions import *
from bayesian import *
from maximumlikelihood import *
from probabilityweightedmoment import *

def fit(
    ds: xr.Dataset,
    distributions: Optional[list[str]] = None,
    min_years: Optional[int] = None,
    ) -> xr.Dataset:
    distributions = distributions or ['gev', 'gumbel', 'gp']
    out = []
    for v in ds.data_vars:
        p = []
        for d in distributions:
            if d == 'gev':
                theta = xr.DataArray(gevfit_1(ds[v].values).theta)
            elif d == 'gumbel':
                theta = xr.DataArray(gumbelfit_1(ds[v].values).theta)
            elif d == 'gp':
                theta = xr.DataArray(gpfit_1(ds[v].values).theta)
            else:
                raise ValueError(f"Unsupported distribution type: {d}")

            p.append(
                theta
                .assign_coords(scipy_dist=d)
                .expand_dims("scipy_dist")
            )
        params = xr.concat(p, dim="scipy_dist")

        # Reorder dparams to match the order of the parameters across all distributions, since subsequent operations rely on this.
        p_order = sorted(set(params.dparams.values).difference(["location", "logscale", "shape"])) + [
            "location",
            "logscale",
            "shape",
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
    out.attrs = ds.attrs

    return out