from xhydro.extreme_value_analysis import py_str_to_jl_symbol
from juliacall import convert as jl_convert
from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from xhydro.extreme_value_analysis.structures.dataitem import Variable


# Object conversions
def jl_symbol_fit_parameters(params: list[list[str]]) -> tuple:
    # python list of lists of julia Symbols
    symbols = [[py_str_to_jl_symbol(symbol) for symbol in params[i]] for i in range(len(params))]

    # python tuple of julia vectors of julia Symbols
    jl_params = tuple((jl_convert(jl.Vector[jl.Symbol], symbols[i])) for i in range(len(symbols)))
    return jl_params

def jl_variable_fit_parameters(params: list[list[Variable]]) -> tuple:
    # python list of lists of julia.Extremes Variables
    variables = [[variable.py_variable_to_jl_variable() for variable in params[i]] for i in range(len(params))]

    # python tuple of julia vectors of julia.Extremes Variables
    jl_params = tuple(jl_convert(jl.Vector[jl.Extremes.Variable], variables[i]) for i in range(len(variables)))
    return jl_params