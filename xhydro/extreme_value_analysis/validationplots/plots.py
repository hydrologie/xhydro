from xhydro.extreme_value_analysis.julia_import import Extremes
from xhydro.extreme_value_analysis.structures.abstract_extreme_value_model import *
from xhydro.extreme_value_analysis.structures.abstract_fitted_extreme_value_model import AbstractFittedExtremeValueModel
from xhydro.extreme_value_analysis.structures.conversions import *

def diagnosticplots(fm: AbstractFittedExtremeValueModel):
    jl_fm = py_aev_to_jl_aev(fm)
    return Extremes.diagnosticplots(jl_fm)
def probplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = py_aev_to_jl_aev(fm)
    return Extremes.probplot(jl_fm)

def qqplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = py_aev_to_jl_aev(fm)
    return Extremes.qqplot(jl_fm)

def qqplotci(fm: AbstractFittedExtremeValueModel):
    jl_fm = py_aev_to_jl_aev(fm)
    return Extremes.qqplotci(jl_fm)

def returnlevelplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = py_aev_to_jl_aev(fm)
    return Extremes.returnlevelplot(jl_fm)

def returnlevelplotci(fm: AbstractFittedExtremeValueModel):
    jl_fm = py_aev_to_jl_aev(fm)
    return Extremes.returnlevelplotci(jl_fm)

def histplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = py_aev_to_jl_aev(fm)
    return Extremes.histplot(jl_fm)

def mrlplot(y: list[float], steps:int):
    jl_y = py_list_to_jl_vector(y)
    return Extremes.mrlplot(jl_y, steps)
