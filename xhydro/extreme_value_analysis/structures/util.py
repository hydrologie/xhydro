from xhydro.extreme_value_analysis.structures.abstract_fitted_extreme_value_model import AbstractFittedExtremeValueModel
from xhydro.extreme_value_analysis.structures.conversions import *
from xhydro.extreme_value_analysis.structures.returnlevel import ReturnLevel


def returnlevel(fm: AbstractFittedExtremeValueModel, return_period: float) -> ReturnLevel:
    jl_fm = py_aev_to_jl_aev(fm)
    jl_return_period = jl.Real(return_period)
    return jl_returnlevel_to_py_returnlevel(Extremes.returnlevel(jl_fm, jl_return_period))

#TODO: see if the other returnlevel() signature is needed

def cint_1(fm: AbstractFittedExtremeValueModel, confidencelevel: float) -> list[list[float]]:
    jl_fm = py_aev_to_jl_aev(fm)
    jl_confidencelevel = jl.Real(confidencelevel)
    return [jl_vector_to_py_list(jl_vector) for jl_vector in Extremes.cint(jl_fm, jl_confidencelevel)]

def cint_2(rl: ReturnLevel, confidencelevel: float) -> list[list[float]]:
    jl_rl = py_returnlevel_to_jl_returnlevel(rl)
    jl_confidencelevel = jl.Real(confidencelevel)
    return [jl_vector_to_py_list(jl_vector) for jl_vector in Extremes.cint(jl_rl, jl_confidencelevel)]

#TODO: fix the return type, see about parametervar(fm::pwmAbstractExtremeValueModel, nboot::Int=1000) signature
def parametervar(fm: AbstractFittedExtremeValueModel) -> list[float]:
    jl_fm = py_aev_to_jl_aev(fm)
    # return jl_vector_to_py_list(Extremes.parametervar(jl_fm))
    return Extremes.parametervar(jl_fm)
