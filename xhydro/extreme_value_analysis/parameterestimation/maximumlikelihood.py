from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from xhydro.extreme_value_analysis.structures.abstract_extreme_value_model import *
from xhydro.extreme_value_analysis.structures.dataitem import *
from xhydro.extreme_value_analysis.structures.data import *
from xhydro.extreme_value_analysis import *
from xhydro.extreme_value_analysis.parameterestimation import *
import pandas as pd

# GEV
def gevfit_1(y:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], shapecov: list[Variable] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([locationcov, logscalecov, shapecov])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gevfit(jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov, shapecov=jl_shapecov))

def gevfit_2(y:list[float], initialvalues:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], shapecov: list[Variable] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_y, jl_initialvalues = py_list_to_jl_vector(y), py_list_to_jl_vector(initialvalues)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([locationcov, logscalecov, shapecov])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gevfit(jl_y, jl_initialvalues, locationcov=jl_locationcov, logscalecov=jl_logscalecov, shapecov=jl_shapecov))

def gevfit_3(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, locationcovid: list[str] = [], logscalecovid: list[str] = [], shapecovid: list[str] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_locationcovid, jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([locationcovid, logscalecovid, shapecovid])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gevfit(jl_df, jl_datacol, locationcovid = jl_locationcovid, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid))

def gevfit_4(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, initialvalues: list[float], locationcovid: list[str] = [], logscalecovid: list[str] = [], shapecovid: list[str] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_initialvalues = py_list_to_jl_vector(initialvalues)
    jl_locationcovid, jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([locationcovid, logscalecovid, shapecovid])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gevfit(jl_df, jl_datacol, jl_initialvalues, locationcovid = jl_locationcovid, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid))

#TODO: test when py_blockmaxima_to_jl_blockmaxima is fixed
def gevfit_5(model: BlockMaxima, initialvalues: list[float]) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_model = py_blockmaxima_to_jl_blockmaxima(model)
    jl_initialvalues = py_list_to_jl_vector(initialvalues)
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gevfit(jl_model, jl_initialvalues))


# Gumbel 
def gumbelfit_1(y:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_y = py_list_to_jl_vector(y)
    jl_locationcov, jl_logscalecov= jl_variable_fit_parameters([locationcov, logscalecov])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gumbelfit(jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov))

def gumbelfit_2(y:list[float], initialvalues: list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_y, jl_initialvalues = py_list_to_jl_vector(y), py_list_to_jl_vector(initialvalues)
    jl_locationcov, jl_logscalecov= jl_variable_fit_parameters([locationcov, logscalecov])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gumbelfit(jl_y, jl_initialvalues, locationcov=jl_locationcov, logscalecov=jl_logscalecov))

def gumbelfit_3(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, locationcovid: list[str] = [], logscalecovid: list[str] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_locationcovid, jl_logscalecovid= jl_symbol_fit_parameters([locationcovid, logscalecovid])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gumbelfit(jl_df, jl_datacol, locationcovid=jl_locationcovid, logscalecovid=jl_logscalecovid))

def gumbelfit_4(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, initialvalues: list[float], locationcovid: list[str] = [], logscalecovid: list[str] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_initialvalues = py_list_to_jl_vector(initialvalues)
    jl_locationcovid, jl_logscalecovid= jl_symbol_fit_parameters([locationcovid, logscalecovid])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gumbelfit(jl_df, jl_datacol, jl_initialvalues, locationcovid=jl_locationcovid, logscalecovid=jl_logscalecovid))

#TODO: test when py_blockmaxima_to_jl_blockmaxima is fixed
def gumbelfit_5(model: BlockMaxima, initialvalues:list[float]) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_model = py_blockmaxima_to_jl_blockmaxima(model)
    jl_initialvalues = py_list_to_jl_vector(initialvalues)
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gumbelfit(jl_model, jl_initialvalues))


# GP
def gpfit_1(y:list[float], logscalecov: list[Variable] = [], shapecov: list[Variable] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_y = py_list_to_jl_vector(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([logscalecov, shapecov])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gpfit(jl_y, logscalecov=jl_logscalecov, shapecov=jl_shapecov))

def gpfit_2(y:list[float], initialvalues:list[float], logscalecov: list[Variable] = [], shapecov: list[Variable] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_y, jl_initialvalues = py_list_to_jl_vector(y), py_list_to_jl_vector(initialvalues)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([logscalecov, shapecov])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gpfit(jl_y, jl_initialvalues, logscalecov=jl_logscalecov, shapecov=jl_shapecov))

def gpfit_3(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, logscalecovid: list[str] = [], shapecovid: list[str] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([logscalecovid, shapecovid])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gpfit(jl_df, jl_datacol, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid))

def gpfit_4(py_dataframe: Union[pd.DataFrame, xr.DataArray], datacol: str, initialvalues: list[float], logscalecovid: list[str] = [], shapecovid: list[str] = []) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_df = py_dataframe_to_jl_dataframe(py_dataframe) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_initialvalues = py_list_to_jl_vector(initialvalues)
    jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([logscalecovid, shapecovid])
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gpfit(jl_df, jl_datacol, jl_initialvalues, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid))

#TODO: test when py_threshold_exceedance_to_jl_threshold_exceedance is tested
def gpfit_5(model:ThresholdExceedance , initialvalues: list[float]) -> MaximumLikelihoodAbstractExtremeValueModel:
    jl_model = py_threshold_exceedance_to_jl_threshold_exceedance(model)
    jl_initialvalues = py_list_to_jl_vector(initialvalues)
    return jl_maximumlikelihood_aev_to_py_aev(Extremes.gpfit(jl_model, jl_initialvalues))




