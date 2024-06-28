from xhydro.extreme_value_analysis.structures.abstract_fitted_extreme_value_model import AbstractFittedExtremeValueModel
from xhydro.extreme_value_analysis.structures.cluster import Cluster
from xhydro.extreme_value_analysis.structures.conversions import *
from xhydro.extreme_value_analysis.structures.returnlevel import ReturnLevel


def returnlevel(fm: AbstractFittedExtremeValueModel, return_period: float) -> ReturnLevel:
    jl_fm = py_aev_to_jl_aev(fm)
    jl_return_period = jl.Real(return_period)
    return jl_returnlevel_to_py_returnlevel(Extremes.returnlevel(jl_fm, jl_return_period))

#TODO: see if the other returnlevel() signature is needed

def cint_model(fm: AbstractFittedExtremeValueModel, confidencelevel: float) -> list[list[float]]:
    jl_fm = py_aev_to_jl_aev(fm)
    jl_confidencelevel = jl.Real(confidencelevel)
    return [jl_vector_to_py_list(jl_vector) for jl_vector in Extremes.cint(jl_fm, jl_confidencelevel)]

def cint_returnlevel(rl: ReturnLevel, confidencelevel: float) -> list[list[float]]:
    jl_rl = py_returnlevel_to_jl_returnlevel(rl)
    jl_confidencelevel = jl.Real(confidencelevel)
    return [jl_vector_to_py_list(jl_vector) for jl_vector in Extremes.cint(jl_rl, jl_confidencelevel)]

def parametervar(fm: AbstractFittedExtremeValueModel, nboot: int = None) -> list[list[float]]:
    jl_fm = py_aev_to_jl_aev(fm)
    if nboot is not None:
        if not isinstance(fm, PwmAbstractExtremeValueModel):
            raise ValueError("nboot is only supported for PwmAbstractExtremeValueModel")
        else:
            return [jl_vector_to_py_list(jl_vector) for jl_vector in jl.eachrow(Extremes.parametervar(jl_fm, nboot))]
    else:
        return [jl_vector_to_py_list(jl_vector) for jl_vector in jl.eachrow(Extremes.parametervar(jl_fm))]

def aic(fm: MaximumLikelihoodAbstractExtremeValueModel) -> float:
    jl_fm = py_aev_to_jl_aev(fm)
    return float(Extremes.aic(jl_fm))

def bic(fm: MaximumLikelihoodAbstractExtremeValueModel) -> float:
    jl_fm = py_aev_to_jl_aev(fm)
    return float(Extremes.bic(jl_fm))

#TODO: check return types for params(), location(), scale() and shape() once xkp specifications are clearer
def params(fm: AbstractFittedExtremeValueModel) -> list[float]:
    jl_fm = py_aev_to_jl_aev(fm)
    return jl_vector_to_py_list(Extremes.params(jl_fm))

def location(fm: AbstractFittedExtremeValueModel) -> list[float]:
    jl_fm = py_aev_to_jl_aev(fm)
    return  jl_vector_to_py_list(Extremes.location(jl_fm))

def scale(fm: AbstractFittedExtremeValueModel) -> list[float]:
    jl_fm = py_aev_to_jl_aev(fm)
    return  jl_vector_to_py_list(Extremes.scale(jl_fm))

def shape(fm: AbstractFittedExtremeValueModel) -> list[float]:
    jl_fm = py_aev_to_jl_aev(fm)
    return  jl_vector_to_py_list(Extremes.shape(jl_fm))

def getcluster_1(y: list[float], u_1: float = None, u_2: float = None) -> list[Cluster]:
    jl_y = py_list_to_jl_vector(y)
    u_1, u_2 = jl_convert(jl.Real, u_1), jl_convert(jl.Real, u_2)
    jl_clusters = Extremes.getcluster(jl_y, u_1, u_2) # juliacall.VectorValue
    return [jl_cluster_to_py_cluster(cluster) for cluster in jl_clusters]

def getcluster_2(y: list[float], u: float = None, runlength: int = None) -> list[Cluster]:
    jl_y = py_list_to_jl_vector(y)
    u = jl_convert(jl.Real, u)
    jl_clusters = Extremes.getcluster(jl_y, u, runlength=runlength)
    return [jl_cluster_to_py_cluster(cluster) for cluster in jl_clusters]

#8. Parameter estimation
def jl_symbol_fit_parameters(params: list[list[str]]) -> tuple:
    # python list of lists of julia Symbols
    symbols = [[py_str_to_jl_symbol(symbol) for symbol in params[i]] for i in range(len(params))]

    # python tuple of julia vectors of julia Symbols
    jl_params = tuple((jl_convert(jl.Vector[jl.Symbol], symbols[i])) for i in range(len(symbols)))
    return jl_params

def jl_variable_fit_parameters(params: list[list[Variable]]) -> tuple:
    # python list of lists of julia.Extremes Variables
    variables = [[py_variable_to_jl_variable(variable) for variable in params[i]] for i in range(len(params))]

    # python tuple of julia vectors of julia.Extremes Variables
    jl_params = tuple(jl_convert(jl.Vector[jl.Extremes.Variable], variables[i]) for i in range(len(variables)))
    return jl_params
