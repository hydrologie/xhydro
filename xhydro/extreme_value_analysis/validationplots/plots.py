from xhydro.extreme_value_analysis import py_list_float_to_julia_vector
from xhydro.extreme_value_analysis.julia_import import Extremes
from xhydro.extreme_value_analysis.structures.abstract_extreme_value_model import *

#TODO: test after py_blockmaxima_to_jl_blockmaxima fix
def diagnosticplots(fm: AbstractFittedExtremeValueModel):
    jl_fm = fm.py_aev_to_jl_aev()
    return Extremes.diagnosticplots(jl_fm)


def probplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = fm.py_aev_to_jl_aev()
    return Extremes.probplot(jl_fm)


def qqplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = fm.py_aev_to_jl_aev()
    return Extremes.qqplot(jl_fm)


def qqplotci(fm: AbstractFittedExtremeValueModel):
    jl_fm = fm.py_aev_to_jl_aev()
    return Extremes.qqplotci(jl_fm)


def returnlevelplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = fm.py_aev_to_jl_aev()
    return Extremes.returnlevelplot(jl_fm)


def returnlevelplotci(fm: AbstractFittedExtremeValueModel):
    jl_fm = fm.py_aev_to_jl_aev()
    return Extremes.returnlevelplotci(jl_fm)


def histplot(fm: AbstractFittedExtremeValueModel):
    jl_fm = fm.py_aev_to_jl_aev()
    return Extremes.histplot(jl_fm)

def mrlplot(y: list[float], steps:int):
    jl_y = py_list_to_jl_vector(y)
    return Extremes.mrlplot(jl_y, steps) 
