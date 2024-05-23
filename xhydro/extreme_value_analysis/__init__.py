from .julia_import import Extremes, jl
from juliacall import convert as jl_convert

def py_str_to_jl_symbol(str: str):
    return jl.Symbol(str)

def py_list_float_to_julia_vector_real(list: list[float]):
    return jl_convert(jl.Vector[jl.Real], list) 