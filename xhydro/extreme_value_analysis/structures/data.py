from typing import Union
from juliacall import Main as jl
import pandas as pd
import xarray as xr
from xhydro.extreme_value_analysis import *
from juliacall import convert as jl_convert
jl.seval("using DataFrames")

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

