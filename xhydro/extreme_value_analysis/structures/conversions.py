from xhydro.extreme_value_analysis.structures.abstract_fitted_extreme_value_model import *
from typing import Union
from juliacall import Main as jl
import pandas as pd
import xarray as xr
from xhydro.extreme_value_analysis import *
from juliacall import convert as jl_convert
from xhydro.extreme_value_analysis.structures.cluster import Cluster
from xhydro.extreme_value_analysis.structures.returnlevel import ReturnLevel
jl.seval("using DataFrames")


# 1. abstract_extreme_value_model.py
def jl_blockmaxima_to_py_blockmaxima(jl_blockmaxima) -> BlockMaxima:
    py_data = jl_variable_to_py_variable(jl_blockmaxima.data)
    #TODO: create py_location, py_logscale, py_shape once julia Function <-> python equivalent conversion is implemented
    jl_location = jl_blockmaxima.location
    jl_logscale = jl_blockmaxima.logscale
    jl_shape = jl_blockmaxima.shape

    jl_type = str(jl.typeof(jl_blockmaxima))
    if jl_type == "BlockMaxima{Distributions.GeneralizedExtremeValue}" or jl_type == "BlockMaxima{Distributions.Gumbel}":
        py_type = jl_type
    else:
        raise ValueError(f"Unknown BlockMaxima type: {jl_type}")
    return BlockMaxima(py_data, jl_location, jl_logscale, jl_shape, py_type)

def py_blockmaxima_to_jl_blockmaxima(py_blockmaxima: BlockMaxima):
    jl_data = py_variable_to_jl_variable(py_blockmaxima.data)
    #TODO: for now, the location, logscale and shape of the py_blockmaxima are already in julia format because 
    # I have not been able to do a python paramfun <-> julia paramfun conversion
    # See the comment in jl_blockmaxima_to_py_blockmaxima
    jl_location = py_blockmaxima.location
    jl_logscale = py_blockmaxima.logscale
    jl_shape = py_blockmaxima.shape

    if py_blockmaxima.type == "BlockMaxima{Distributions.Gumbel}":
        return Extremes.seval('BlockMaxima{Distributions.Gumbel}')(jl_data, jl_location, jl_logscale, jl_shape)
    elif py_blockmaxima.type == "BlockMaxima{Distributions.GeneralizedExtremeValue}":
        return Extremes.seval('BlockMaxima{Distributions.GeneralizedExtremeValue}')(jl_data, jl_location, jl_logscale, jl_shape)
    else:
        raise ValueError("Unsupported BlockMaxima type: {}".format(py_blockmaxima.type))
    
def jl_threshold_exceedance_to_py_threshold_exceedance(jl_threshold_exceedence) -> ThresholdExceedance:
    py_data = jl_variable_to_py_variable(jl_threshold_exceedence.data)
    #TODO: create py_logscale, py_shape once julia Function <-> python equivalent conversion is implemented
    jl_logscale = jl_threshold_exceedence.logscale
    jl_shape = jl_threshold_exceedence.shape
    return ThresholdExceedance(py_data, jl_logscale, jl_shape)

def py_threshold_exceedance_to_jl_threshold_exceedance(py_threshold_exceedance: ThresholdExceedance):
    jl_data = py_variable_to_jl_variable(py_threshold_exceedance.data)
    #TODO: for now, logscale and shape of the py_threshold_exceedance are already in julia format because 
    # I have not been able to do a python paramfun <-> julia paramfun conversion
    # See the comment in jl_threshold_exceedance_to_py_threshold_exceedance
    jl_logscale = py_threshold_exceedance.logscale
    jl_shape = py_threshold_exceedance.shape
    return Extremes.seval('ThresholdExceedance')(jl_data, jl_logscale, jl_shape)



# 2. abstract_fitted_extreme_value_model.py
def py_aev_to_jl_aev(abstract_fitted_extreme_value_model: AbstractFittedExtremeValueModel):
    if (isinstance(abstract_fitted_extreme_value_model, MaximumLikelihoodAbstractExtremeValueModel)):
        return py_maximumlikelihood_aev_to_jl_aev(abstract_fitted_extreme_value_model)
    elif (isinstance(abstract_fitted_extreme_value_model, BayesianAbstractExtremeValueModel)):
        return py_bayesian_aev_to_jl_aev(abstract_fitted_extreme_value_model)
    elif (isinstance(abstract_fitted_extreme_value_model, PwmAbstractExtremeValueModel)):
        return py_pwm_aev_to_jl_aev(abstract_fitted_extreme_value_model)
    else:
        raise ValueError(f"Unknown model type: {type(abstract_fitted_extreme_value_model)}")

def py_maximumlikelihood_aev_to_jl_aev(py_model: MaximumLikelihoodAbstractExtremeValueModel):
    if(isinstance(py_model.model, BlockMaxima)):
        jl_model = py_blockmaxima_to_jl_blockmaxima(py_model.model)
    elif(isinstance(py_model.model, ThresholdExceedance)):
        jl_model = py_threshold_exceedance_to_jl_threshold_exceedance(py_model.model)
    else:
        raise ValueError(f"Unknown model type: {type(py_model.model)}")
    jl_theta = jl_convert(jl.Vector[jl.Float64], py_model.theta)
    return Extremes.MaximumLikelihoodAbstractExtremeValueModel(jl_model,jl_theta)

def jl_maximumlikelihood_aev_to_py_aev(jl_model) -> MaximumLikelihoodAbstractExtremeValueModel:
    if (str(jl.typeof(jl_model.model)) == "BlockMaxima{Distributions.GeneralizedExtremeValue}" or str(jl.typeof(jl_model.model)) == "BlockMaxima{Distributions.Gumbel}"): 
        py_model = jl_blockmaxima_to_py_blockmaxima(jl_model.model)
    elif (str(jl.typeof(jl_model.model)) == "ThresholdExceedance"):
        py_model = jl_threshold_exceedance_to_py_threshold_exceedance(jl_model.model)
    else:
        raise ValueError(f"Unknown model type: {jl.typeof(jl_model.model)}")
    py_theta = jl_vector_to_py_list(getattr(jl_model, "θ̂")) # Note : θ̂  is a julia symbol, the circumflex is not a typo
    return MaximumLikelihoodAbstractExtremeValueModel(py_model, py_theta)

def py_bayesian_aev_to_jl_aev(py_model: BayesianAbstractExtremeValueModel):
    if(isinstance(py_model.model, BlockMaxima)):
        jl_model = py_blockmaxima_to_jl_blockmaxima(py_model.model)
    elif(isinstance(py_model.model, ThresholdExceedance)):
        jl_model = py_threshold_exceedance_to_jl_threshold_exceedance(py_model.model)
    else:
        raise ValueError(f"Unknown model type: {type(py_model.model)}")
    jl_sim = py_model.sim
    return Extremes.BayesianAbstractExtremeValueModel(jl_model, jl_sim)

def jl_bayesian_aev_to_py_aev(jl_model) -> BayesianAbstractExtremeValueModel:
    if (str(jl.typeof(jl_model.model)) == "BlockMaxima{Distributions.GeneralizedExtremeValue}" or str(jl.typeof(jl_model.model)) == "BlockMaxima{Distributions.Gumbel}"): 
        py_model = jl_blockmaxima_to_py_blockmaxima(jl_model.model)
    elif(str(jl.typeof(jl_model.model)) == "ThresholdExceedance"):
        py_model = jl_threshold_exceedance_to_py_threshold_exceedance(jl_model.model)
    else:
        raise ValueError(f"Unknown model type: {jl.typeof(jl_model.model)}")
    py_sim = jl_model.sim
    return BayesianAbstractExtremeValueModel(py_model, py_sim)

def py_pwm_aev_to_jl_aev(py_model: MaximumLikelihoodAbstractExtremeValueModel):
    return py_maximumlikelihood_aev_to_jl_aev(py_model)

def jl_pwm_aev_to_py_aev(jl_model) -> PwmAbstractExtremeValueModel:
    if (str(jl.typeof(jl_model.model)) == "BlockMaxima{Distributions.GeneralizedExtremeValue}" or str(jl.typeof(jl_model.model)) == "BlockMaxima{Distributions.Gumbel}"):
        py_model = jl_blockmaxima_to_py_blockmaxima(jl_model.model)
    elif(str(jl.typeof(jl_model.model)) == "ThresholdExceedance"):
        py_model = jl_threshold_exceedance_to_py_threshold_exceedance(jl_model.model)
    else:
        raise ValueError(f"Unknown model type: {jl.typeof(jl_model.model)}")
    py_theta = jl_vector_to_py_list(getattr(jl_model, "θ̂"))
    return PwmAbstractExtremeValueModel(py_model, py_theta)


# 3. dataframes and xarrays
def py_dataframe_to_jl_dataframe(py_dataframe: Union[pd.DataFrame, xr.DataArray]):
    if isinstance(py_dataframe, pd.DataFrame):
        return pd_dataframe_to_jl_dataframe(py_dataframe)
    elif isinstance(py_dataframe, xr.DataArray):
        return xr_dataarray_to_jl_dataframe(py_dataframe)
    else:
        raise ValueError("The input should be a pandas DataFrame or an xarray DataArray.")

# pandas.DataFrame conversions
def jl_dataframe_to_pd_dataframe(jl_dataframe) -> pd.DataFrame:
    col_names = []
    values = []
    for name in jl.names(jl_dataframe):
        col_names.append(name)
    for col in jl.eachcol(jl_dataframe):
        values.append(jl_vector_to_py_list(col))
    data = {col_names[i]: values[i] for i in range(len(col_names))}
    return pd.DataFrame(data)

def pd_dataframe_to_jl_dataframe(df: pd.DataFrame):
    jl_columns = {jl.Symbol(col): py_list_to_jl_vector(df[col].values.tolist()) for col in df.columns}
    return jl.DataFrame(jl_columns)

# xarray.DataArray conversions
def jl_dataframe_to_xr_dataarray(jl_dataframe) -> xr.DataArray:
    xr_data = [jl_vector_to_py_list(col) for col in jl.eachcol(jl_dataframe)]
    xr_dims = ['x', 'y'] # columns along the x-axis, rows along the y-axis
    xr_coords = {'x': [name for name in jl.names(jl_dataframe)], 'y':[i for i in range(max(len(row) for row in xr_data))]}
    return xr.DataArray(xr_data, coords=xr_coords, dims=xr_dims)

def xr_dataarray_to_jl_dataframe(xr_dataarray: xr.DataArray):
    jl_columns = {jl.Symbol(str(col_name)): jl_convert(jl.Vector[jl.Real], xr_dataarray.sel(x=col_name))  for col_name in xr_dataarray.coords['x'].values}
    return jl.DataFrame(jl_columns)

# 4. dataitem.py
def py_variable_to_jl_variable(py_var: Variable):
    return jl.Extremes.Variable(py_var.name, jl_convert(jl.Vector[jl.Real], py_var.value))

def jl_variable_to_py_variable(jl_variable) -> Variable:
    return Variable(
        getattr(jl_variable, "name"),
        jl_vector_to_py_list(getattr(jl_variable, "value"))
    )

# 5. Basic conversions
def py_str_to_jl_symbol(str: str):
    return jl.Symbol(str)

def py_list_to_jl_vector(py_list: list):
    if all(isinstance(i, float) or isinstance(i, int) for i in py_list):
        return jl_convert(jl.Vector[jl.Real], py_list) 
    if all(isinstance(i, str) for i in py_list):
        return jl_convert(jl.Vector[jl.String], py_list) 
    else:
        return jl_convert(jl.Vector[jl.Any], py_list) # for other types of values
    
def jl_vector_to_py_list(jl_vector) -> list:
    return list(jl_vector)

# 6. returnlevel.py
def py_returnlevel_to_jl_returnlevel(py_returnlevel: ReturnLevel):
    jl_model = py_returnlevel.model 
    jl_returnperiod = jl.Real(py_returnlevel.returnperiod)
    jl_value = py_list_to_jl_vector(py_returnlevel.value)
    return Extremes.ReturnLevel(jl_model, jl_returnperiod, jl_value)

def jl_returnlevel_to_py_returnlevel(jl_returnlevel) -> ReturnLevel:
    return ReturnLevel(jl_returnlevel.model, float(jl_returnlevel.returnperiod), list(jl_returnlevel.value))

# 7. cluster.py
def jl_cluster_to_py_cluster(jl_cluster) -> Cluster:
    return Cluster(
        getattr(jl_cluster, 'u₁'), 
        getattr(jl_cluster, 'u₂'), 
        list(getattr(jl_cluster, 'position')), 
        list(getattr(jl_cluster, 'value'))
    )