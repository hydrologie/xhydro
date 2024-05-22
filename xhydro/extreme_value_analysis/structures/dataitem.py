from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from juliacall import convert as jl_convert

class DataItem:
    pass

class Variable(DataItem):
    name: str
    value: list[float]

    def __init__(self, name: str, value: list[float]):
        self.name, self.value = name, value

    def py_variable_to_jl_variable(self):
        return jl.Extremes.Variable(self.name, jl_convert(jl.Vector[jl.Real], self.value))

    
class VariableStd(DataItem):
    name: str
    value: list[float]
    offset: float
    scale: float

    def __init__(self, name: str, value: list[float], offset: float, scale: float):
        self.name, self.value, self.offset, self.scale = name, value, offset, scale






    