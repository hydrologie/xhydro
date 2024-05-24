from .julia_import import Extremes, jl
from juliacall import convert as jl_convert

def py_str_to_jl_symbol(str: str):
    return jl.Symbol(str)

def py_list_to_julia_vector(list: list):
    if all(isinstance(i, float) or isinstance(i, int) for i in list):
        return jl_convert(jl.Vector[jl.Real], list) # we will often be working with Vectors of <:Real values
    else:
        return jl_convert(jl.Vector[jl.Any], list)
    
def jl_vector_to_py_list(jl_vector) -> list:
    return list(jl_vector)