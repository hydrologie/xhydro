from xhydro.extreme_value_analysis import py_list_float_to_julia_vector
from xhydro.extreme_value_analysis.julia_import import Extremes, jl
import pandas as pd
from xhydro.extreme_value_analysis.structures.abstract_extreme_value_model import AbstractFittedExtremeValueModel, py_bayesian_aev_to_jl_aev
from xhydro.extreme_value_analysis.structures.data import jl_dataframe_to_pd_dataframe 


def probplot_data(fm: AbstractFittedExtremeValueModel) -> pd.DataFrame:
    jl_fm = fm.py_aev_to_jl_aev()
    return jl_dataframe_to_pd_dataframe(Extremes.probplot_data(jl_fm))


def qqplot_data(fm: AbstractFittedExtremeValueModel) -> pd.DataFrame:
    jl_fm = fm.py_aev_to_jl_aev()
    return jl_dataframe_to_pd_dataframe(Extremes.qqplot_data(jl_fm))


def returnlevelplot_data(fm: AbstractFittedExtremeValueModel) -> pd.DataFrame:
    jl_fm = fm.py_aev_to_jl_aev()
    return jl_dataframe_to_pd_dataframe(Extremes.returnlevelplot_data(jl_fm))

def histplot_data(fm: AbstractFittedExtremeValueModel) -> dict:
    jl_fm = fm.py_aev_to_jl_aev()
    return dict(Extremes.histplot_data(jl_fm))


# TODO: test after type issue fix
def mrlplot_data(y: list[float], steps: int) -> pd.DataFrame:
    jl_y = py_list_float_to_julia_vector(y)
    jl_df = Extremes.mrlplot_data(jl_y, steps)
    pd_df = jl_dataframe_to_pd_dataframe(jl_df)
    return pd_df
