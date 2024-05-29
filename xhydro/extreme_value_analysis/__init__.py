from .julia_import import Extremes, jl
from juliacall import convert as jl_convert

def py_str_to_jl_symbol(str: str):
    return jl.Symbol(str)

def py_list_to_jl_vector(py_list: list):
    if all(isinstance(i, float) or isinstance(i, int) for i in py_list):
        return jl_convert(jl.Vector[jl.Real], py_list) 
    if all(isinstance(i, str) for i in py_list):
        return jl_convert(jl.Vector[jl.String], py_list) 
    else:
        return jl_convert(jl.Vector[jl.Any], py_list) # for other types of values
    
def jl_vector_to_py_list(jl_vector) -> list:
    return list(jl_vector)