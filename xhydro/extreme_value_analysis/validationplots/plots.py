from xhydro.extreme_value_analysis import py_list_float_to_julia_vector
from xhydro.extreme_value_analysis.julia_import import Extremes


def diagnosticplots(fm):
    return Extremes.diagnosticplots(fm)


def probplot(fm):
    return Extremes.probplot(fm)


def qqplot(fm):
    return Extremes.qqplot(fm)


def qqplotci(fm):
    return Extremes.qqplotci(fm)


def returnlevelplot(fm):
    return Extremes.returnlevelplot(fm)


def returnlevelplotci(fm):
    return Extremes.returnlevelplotci(fm)


def histplot(fm):
    return Extremes.histplot(fm)

#TODO: test after type issue fix
def mrlplot(y: list[float], steps:int):
    jl_y = py_list_float_to_julia_vector(y)
    return Extremes.mrlplot(jl_y, steps) 
