from juliacall import Main as jl
import pandas as pd

# Load the DataFrames package in Julia
jl.seval("using DataFrames")

# Define a function to convert Julia DataFrame to pandas DataFrame
def jldataframe_to_pandas_dataframe(jl_dataframe) -> pd.DataFrame:
    # Extract column names from the Julia DataFrame
    columns = jl.eval("names($jl_dataframe)")
    print(columns)
    # Extract data for each column and create a dictionary
    data = {col: list(jl.eval("[$jl_dataframe[!, $col][i] for i in 1:length($jl_dataframe[!, $col])]")) for col in columns}

    # Create a pandas DataFrame from the extracted data
    return pd.DataFrame(data)

def pandas_dataframe_to_julia_dataframe():
    pass




