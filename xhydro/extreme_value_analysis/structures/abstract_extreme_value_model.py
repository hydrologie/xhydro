from xhydro.extreme_value_analysis import *
from xhydro.extreme_value_analysis.structures.dataitem import Variable
from julia_import import *

class paramfun:
    covariate: list[Variable]
    #TODO: give type to fun, which in julia is a Function
    def __init__(self, covariate: list[Variable], fun):
        self.covariate, self.fun = covariate, fun

# class Chains:
#     value: list[float]
#     names: list[str]
#     chains: list[int]

#     #TODO: convert range to python type once julia StepRange <-> python equivalent conversion is implemented
#     def __init__(self, value: list[float], range, names: list[str], chains: list[int]):
#         self.value, self.range, self.names, self.chains = value, range, names, chains

# def py_chains_to_jl_chains(py_chains: Chains):
#     jl_value = py_list_to_jl_vector(py_chains.value)
#     jl_range = py_list_to_jl_vector(py_chains.range)
#     jl_names = py_list_to_jl_vector(py_chains.names)
#     jl_chains = py_list_to_jl_vector(py_chains.chains)
#     # return MambaLite.Chains
#     pass

# def jl_chains_to_py_chains(jl_chains) -> Chains:
#     pass

class AbstractExtremeValueModel:
    pass

class BlockMaxima(AbstractExtremeValueModel):
    data: Variable
    location: paramfun
    logscale: paramfun
    shape: paramfun
    type: str
    def __init__(self, data: Variable, location: paramfun, logscale: paramfun, shape: paramfun, type: str):
        self.data, self.location, self.logscale, self.shape, self.type = data, location, logscale, shape, type
    def __repr__(self):
        return f"\t{self.type} :\n\t\t\tdata:\n {self.data}\n\t\t\tlocation: {self.location}\n\t\t\tlogscale: {self.logscale}\n\t\t\tshape: {self.shape}"

class ThresholdExceedance(AbstractExtremeValueModel):
    data: Variable
    logscale: paramfun
    shape: paramfun
    def __init__(self, data, logscale, shape):
        self.data, self.logscale, self.shape = data, logscale, shape
    def __repr__(self):
        return f"\tThresholdExceedance :\n\t\t\tdata:\n {self.data}\n\t\t\tlogscale: {self.logscale}\n\t\t\tshape: {self.shape}"
















