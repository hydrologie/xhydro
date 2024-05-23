from xhydro.extreme_value_analysis import py_list_float_to_julia_vector_real
from xhydro.extreme_value_analysis.julia_import import Extremes, jl
import pandas as pd
from juliacall import convert as jl_convert
from xhydro.extreme_value_analysis.structures.data import jl_dataframe_to_pd_dataframe  


def probplot_data(fm):
    return Extremes.probplot_data(fm)


def qqplot_data(fm):
    return Extremes.qqplot_data(fm)


def returnlevelplot_data(fm):
    return Extremes.returnlevelplot_data(fm)


def histplot_data(fm):
    return Extremes.histplot_data(fm)


# TODO: test after type issue fix
def mrlplot_data(y: list[float], steps: int) -> pd.DataFrame:
    jl_y = py_list_float_to_julia_vector_real(y)
    jl_df = Extremes.mrlplot_data(jl_y, steps)
    pd_df = jl_dataframe_to_pd_dataframe(jl_df)
    return pd_df
