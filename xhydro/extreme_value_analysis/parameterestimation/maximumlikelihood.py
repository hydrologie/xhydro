from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from juliacall import convert as jl_convert

from xhydro.extreme_value_analysis.structures.dataitem import *

__all__ = ["gevfit","_gevfit_1", "_gevfit_2", "gumbelfit", "gpfit"]


def gevfit(df, datacol: str):
    return Extremes.gevfit(df, jl.Symbol(datacol))

def _gevfit_1(y:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], shapecov: list[Variable] = []):
    y = jl_convert(jl.Vector[jl.Real], y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_fit_parameters(locationcov, logscalecov, shapecov)
    return Extremes.gevfit(y, locationcov=jl_locationcov, logscalecov=jl_logscalecov, shapecov=jl_shapecov )

def _gevfit_2(y:list[float], initialvalues:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], shapecov: list[Variable] = []):
    y, initialvalues = jl_convert(jl.Vector[jl.Real], y),  jl_convert(jl.Vector[jl.Real], initialvalues) 
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_fit_parameters(locationcov, logscalecov, shapecov)
    return Extremes.gevfit(y, initialvalues, locationcov=jl_locationcov, logscalecov=jl_logscalecov, shapecov=jl_shapecov )

def gumbelfit(df, datacol: str):
    return Extremes.gumbelfit(df, jl.Symbol(datacol))


def gpfit(df, datacol: str):
    return Extremes.gumbelfit(df, jl.Symbol(datacol))


def jl_fit_parameters(locationcov: list[Variable], logscalecov: list[Variable], shapecov: list[Variable]):
    jl_locationcov = [v.py_variable_to_jl_variable() for v in locationcov]
    jl_logscalecov = [v.py_variable_to_jl_variable() for v in logscalecov]
    jl_shapecov = [v.py_variable_to_jl_variable() for v in shapecov]

    locationcov = jl_convert(jl.Vector[jl.Extremes.Variable], jl_locationcov)
    logscalecov = jl_convert(jl.Vector[jl.Extremes.Variable], jl_logscalecov)
    shapecov = jl_convert(jl.Vector[jl.Extremes.Variable], jl_shapecov)
    return (locationcov, logscalecov, shapecov)