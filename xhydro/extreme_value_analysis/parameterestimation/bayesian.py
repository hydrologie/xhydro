from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from xhydro.extreme_value_analysis.structures.abstract_extreme_value_model import *
from xhydro.extreme_value_analysis.structures.dataitem import *
from xhydro.extreme_value_analysis.structures.conversions import *

from xhydro.extreme_value_analysis import *
from xhydro.extreme_value_analysis.parameterestimation import *
import pandas as pd


# GEV
def gevfitbayes_1(y:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], shapecov: list[Variable] = [], niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([locationcov, logscalecov, shapecov])
    return jl_bayesian_aev_to_py_aev(Extremes.gevfitbayes(jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov, shapecov=jl_shapecov, niter=niter, warmup=warmup))

def gevfitbayes_2(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, locationcovid: list[str] = [], logscalecovid: list[str] = [], shapecovid: list[str] = [], niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_locationcovid, jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([locationcovid, logscalecovid, shapecovid])
    return jl_bayesian_aev_to_py_aev(Extremes.gevfitbayes(jl_df, jl_datacol, locationcovid = jl_locationcovid, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid, niter=niter, warmup=warmup))

#TODO: test when py_blockmaxima_to_jl_blockmaxima is fixed
def gevfitbayes_3(model: BlockMaxima, niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_model = py_blockmaxima_to_jl_blockmaxima(model)
    return jl_bayesian_aev_to_py_aev(Extremes.gevfitbayes(jl_model, niter=niter, warmup=warmup))


# Gumbel
def gumbelfitbayes_1(y: list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov= jl_variable_fit_parameters([locationcov, logscalecov])
    return jl_bayesian_aev_to_py_aev(Extremes.gumbelfitbayes(jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov, niter=niter, warmup=warmup))

def gumbelfitbayes_2(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, locationcovid: list[str] = [], logscalecovid: list[str] = [], niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_locationcovid, jl_logscalecovid= jl_symbol_fit_parameters([locationcovid, logscalecovid])
    return jl_bayesian_aev_to_py_aev(Extremes.gumbelfitbayes(jl_df, jl_datacol, locationcovid = jl_locationcovid, logscalecovid = jl_logscalecovid, niter=niter, warmup=warmup))

#TODO: test when py_blockmaxima_to_jl_blockmaxima is fixed
def gumbelfitbayes_3(model: BlockMaxima, niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_model = py_blockmaxima_to_jl_blockmaxima(model)
    return jl_bayesian_aev_to_py_aev(Extremes.gumbelfitbayes(jl_model, niter=niter, warmup=warmup))


# Gp
def gpfitbayes_1(y: list[float], logscalecov: list[Variable] = [], shapecov: list[Variable] = [],  niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov= jl_variable_fit_parameters([logscalecov, shapecov])
    return jl_bayesian_aev_to_py_aev(Extremes.gpfitbayes(jl_y, logscalecov=jl_logscalecov, shapecov=jl_shapecov, niter=niter, warmup=warmup))

def gpfitbayes_2(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, logscalecovid: list[str] = [], shapecovid: list[str] = [], niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([logscalecovid, shapecovid])
    return jl_bayesian_aev_to_py_aev(Extremes.gpfitbayes(jl_df, jl_datacol, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid, niter=niter, warmup=warmup))

#TODO: test when py_threshold_exceedance_to_jl_threshold_exceedance is tested
def gpfitbayes_3(model: ThresholdExceedance, niter: int = 5000, warmup: int = 2000) -> BayesianAbstractExtremeValueModel:
    jl_model = py_threshold_exceedance_to_jl_threshold_exceedance(model)
    return jl_bayesian_aev_to_py_aev(Extremes.gpfitbayes(jl_model, niter=niter, warmup=warmup))