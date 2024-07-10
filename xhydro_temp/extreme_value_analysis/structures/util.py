from xhydro_temp.extreme_value_analysis.structures.conversions import *

def jl_variable_fit_parameters(params: list[list[Variable]]) -> tuple:
    # python list of lists of julia.Extremes Variables
    variables = [[py_variable_to_jl_variable(variable) for variable in params[i]] for i in range(len(params))]

    # python tuple of julia vectors of julia.Extremes Variables
    jl_params = tuple(jl_convert(jl.Vector[jl.Extremes.Variable], variables[i]) for i in range(len(variables)))
    return jl_params
