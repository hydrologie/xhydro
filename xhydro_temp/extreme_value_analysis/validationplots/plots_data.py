from xhydro_temp.extreme_value_analysis.julia_import import Extremes
import pandas as pd
from xhydro_temp.extreme_value_analysis.structures.abstract_fitted_extreme_value_model import AbstractFittedExtremeValueModel
from xhydro_temp.extreme_value_analysis.structures.conversions import *

def probplot_data(fm: AbstractFittedExtremeValueModel) -> pd.DataFrame:
    jl_fm = py_aev_to_jl_aev(fm)
    return jl_dataframe_to_pd_dataframe(Extremes.probplot_data(jl_fm))

def qqplot_data(fm: AbstractFittedExtremeValueModel) -> pd.DataFrame:
    jl_fm = py_aev_to_jl_aev(fm)
    return jl_dataframe_to_pd_dataframe(Extremes.qqplot_data(jl_fm))

def returnlevelplot_data(fm: AbstractFittedExtremeValueModel) -> pd.DataFrame:
    jl_fm = py_aev_to_jl_aev(fm)
    return jl_dataframe_to_pd_dataframe(Extremes.returnlevelplot_data(jl_fm))

#TODO: fix appearance
def histplot_data(fm: AbstractFittedExtremeValueModel) -> dict:
    jl_fm = py_aev_to_jl_aev(fm)
    return dict(Extremes.histplot_data(jl_fm))

def mrlplot_data(y: list[float], steps: int) -> pd.DataFrame:
    jl_y = py_list_to_jl_vector(y)
    jl_df = Extremes.mrlplot_data(jl_y, steps)
    pd_df = jl_dataframe_to_pd_dataframe(jl_df)
    return pd_df
