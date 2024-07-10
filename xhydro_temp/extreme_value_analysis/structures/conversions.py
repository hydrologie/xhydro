from xhydro_temp.extreme_value_analysis.julia_import import jl
from typing import Union
import pandas as pd
import xarray as xr
from juliacall import convert as jl_convert
import math
import numpy as np
from xhydro_temp.extreme_value_analysis.structures.dataitem import Variable
jl.seval("using DataFrames")

# 1. dataframes and xarrays
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


# 2. dataitem.py
def py_variable_to_jl_variable(py_var: Variable):
    return jl.Extremes.Variable(py_var.name, jl_convert(jl.Vector[jl.Real], py_var.value))

def jl_variable_to_py_variable(jl_variable) -> Variable:
    return Variable(
        getattr(jl_variable, "name"),
        jl_vector_to_py_list(getattr(jl_variable, "value"))
    )

# 3. Basic conversions
def py_str_to_jl_symbol(str: str):
    return jl.Symbol(str)

def py_list_to_jl_vector(py_list: list):
    # Cleaning up nans and numpy.float32 elements
    py_list = [x for x in py_list if not math.isnan(x)] #TODO: deal with nans beter
    py_list = [float(i) if isinstance(i, np.float32) else i for i in py_list]

    if all(isinstance(i, float) or isinstance(i, int) for i in py_list):
        return jl_convert(jl.Vector[jl.Real], py_list)
    elif all(isinstance(i, str) for i in py_list):
        return jl_convert(jl.Vector[jl.String], py_list)
    elif not(all(isinstance(i, float) or isinstance(i, int) for i in py_list)) and not(all(isinstance(i, str) for i in py_list)):
        raise ValueError(f" Cannot convert unsupported type {type(py_list)} to julia vector: all values are not strings or numbers")
    else:
        raise ValueError(f" Cannot convert unsupported type {type(py_list)} to julia vector")

def jl_vector_to_py_list(jl_vector) -> list:
    return list(jl_vector)

# for a julia vecgor containing a single tuple, i.e. [(1,2,3)]
def jl_vector_tuple_to_py_list(jl_vector_tuple) -> list:
    jl_sub_tuple, = jl_vector_tuple  # Unpack the single tuple from the list
    py_sub_list = list(jl_sub_tuple)
    return py_sub_list

# for a juila matrix of tuples, i.e. [(1,2,3), (4,5,6), (7,8,9)]
def jl_matrix_tuple_to_py_list(jl_matrix_tuple):
    py_list = [tuple(row) for row in jl_matrix_tuple]
    return py_list

