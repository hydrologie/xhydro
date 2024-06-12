from xhydro.extreme_value_analysis import *
from xhydro.extreme_value_analysis.structures.dataitem import Variable
from xhydro.extreme_value_analysis.julia_import import *

class paramfun:
    covariate: list[Variable]
    #TODO: give type to fun, which in julia is a Function
    def __init__(self, covariate: list[Variable], fun):
        self.covariate, self.fun = covariate, fun

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
















