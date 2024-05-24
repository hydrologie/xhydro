from juliacall import Main as jl
import pandas as pd
from xhydro.extreme_value_analysis import *
jl.seval("using DataFrames")

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
    columns = {jl.Symbol(col): py_list_to_julia_vector(df[col].values.tolist()) for col in df.columns}
    jl_df = jl.DataFrame(columns)
    return jl_df

#TODO: xarray conversions



