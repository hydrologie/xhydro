from xhydro.extreme_value_analysis import *
from xhydro.extreme_value_analysis.structures.dataitem import Variable, jl_variable_to_py_variable
from julia_import import *

class paramfun:
    covariate: list[Variable]
    #TODO: give type to fun, which in julia is a Function
    def __init__(self, covariate: list[Variable], fun):
        self.covariate, self.fun = covariate, fun

class Chains:
    value: list[float]
    names: list[str]
    chains: list[int]

    #TODO: convert range to python type once julia StepRange <-> python equivalent conversion is implemented
    def __init__(self, value: list[float], range, names: list[str], chains: list[int]):
        self.value, self.range, self.names, self.chains = value, range, names, chains

def py_chains_to_jl_chains(py_chains: Chains):
    jl_value = py_list_to_jl_vector(py_chains.value)
    jl_range = py_list_to_jl_vector(py_chains.range)
    jl_names = py_list_to_jl_vector(py_chains.names)
    jl_chains = py_list_to_jl_vector(py_chains.chains)
    # return MambaLite.Chains
    pass

def jl_chains_to_py_chains(jl_chains) -> Chains:
    pass

class AbstractExtremeValueModel:
    pass


"""
TODO: figure out what the different BlockMaxima types mean
BlockMaxima{GeneralizedExtremeValue}
BlockMaxima{Gumbel}

BlockMaxima{T} <: AbstractExtremeValueModel
ThresholdExceedance <: AbstractExtremeValueModel
"""
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

def jl_blockmaxima_to_py_blockmaxima(jl_blockmaxima) -> BlockMaxima:
    py_data = jl_variable_to_py_variable(jl_blockmaxima.data)
    #TODO: create py_location, py_logscale, py_shape once julia Function <-> python equivalent conversion is implemented
    jl_location = jl_blockmaxima.location
    jl_logscale = jl_blockmaxima.logscale
    jl_shape = jl_blockmaxima.shape

    jl_type = str(jl.typeof(jl_blockmaxima))
    if jl_type == "BlockMaxima{Distributions.GeneralizedExtremeValue}" or jl_type == "BlockMaxima{Distributions.Gumbel}":
        py_type = jl_type
    else:
        raise ValueError(f"Unknown BlockMaxima type: {jl_type}")
    return BlockMaxima(py_data, jl_location, jl_logscale, jl_shape, py_type)

#TODO: fix
# def py_blockmaxima_to_jl_blockmaxima(py_blockmaxima: BlockMaxima):
#     jl_data = py_blockmaxima.data.py_variable_to_jl_variable()
#     jl_locationcov = py_blockmaxima.location.covariate
#     jl_logscalecov = py_blockmaxima.logscale.covariate
#     jl_shapecov = py_blockmaxima.shape.covariate

#     x = "GeneralizedExtremeValue"

#     if py_blockmaxima.type == "BlockMaxima{Distributions.GeneralizedExtremeValue}":
#         return jl.seval(f"""
#         Extremes.BlockMaxima{x}(
#             {jl_data},
#             locationcov={jl_locationcov},
#             logscalecov={jl_logscalecov},
#             shapecov={jl_shapecov}
#         )::Extremes.BlockMaxima{{Distributions.GeneralizedExtremeValue}}
#         """)
#     elif py_blockmaxima.type == "BlockMaxima{Distributions.Gumbel}":
#         return jl.seval(f"""
#         Extremes.BlockMaxima(
#             {jl_data},
#             locationcov={jl_locationcov},
#             logscalecov={jl_logscalecov}
#         )::Extremes.BlockMaxima{{Distributions.Gumbel}}
#         """)
#     else:
#         raise ValueError("Unsupported BlockMaxima type: {}".format(py_blockmaxima.type))

def py_blockmaxima_to_jl_blockmaxima(py_blockmaxima: BlockMaxima):
    jl_data = py_blockmaxima.data.py_variable_to_jl_variable()
    jl_locationcov = py_blockmaxima.location.covariate
    jl_logscalecov = py_blockmaxima.logscale.covariate
    jl_shapecov = py_blockmaxima.shape.covariate

    if py_blockmaxima.type == "BlockMaxima{Distributions.GeneralizedExtremeValue}":
        return Extremes.BlockMaxima(jl_data, jl_locationcov, jl_logscalecov, jl_shapecov)
    elif py_blockmaxima.type == "BlockMaxima{Distributions.Gumbel}":
        return Extremes.BlockMaxima(jl_data, jl_locationcov, jl_logscalecov)
    else:
        raise ValueError("Unsupported BlockMaxima type: {}".format(py_blockmaxima.type))


class ThresholdExceedance(AbstractExtremeValueModel):
    data: Variable
    logscale: paramfun
    shape: paramfun
    def __init__(self, data, logscale, shape):
        self.data, self.logscale, self.shape = data, logscale, shape
    def __repr__(self):
        return f"\tThresholdExceedance :\n\t\t\tdata:\n {self.data}\n\t\t\tlogscale: {self.logscale}\n\t\t\tshape: {self.shape}"

def jl_threshold_exceedance_to_py_threshold_exceedance(jl_threshold_exceedence) -> ThresholdExceedance:
    py_data = jl_variable_to_py_variable(jl_threshold_exceedence.data)
    #TODO: create py_logscale, py_shape once julia Function <-> python equivalent conversion is implemented
    jl_logscale = jl_threshold_exceedence.logscale
    jl_shape = jl_threshold_exceedence.shape
    return ThresholdExceedance(py_data, jl_logscale, jl_shape)

#TODO: test
def py_threshold_exceedance_to_jl_threshold_exceedance(py_threshold_exceedance: ThresholdExceedance):
    jl_data = py_threshold_exceedance.data.py_variable_to_jl_variable()
    jl_logscalecov = py_threshold_exceedance.logscale.covariate
    jl_shapecov = py_threshold_exceedance.shape.covariate
    return Extremes.ThresholdExceedance(jl_data, jl_logscalecov, jl_shapecov)


class AbstractFittedExtremeValueModel:
    def py_aev_to_jl_aev(self):
        pass

class MaximumLikelihoodAbstractExtremeValueModel(AbstractFittedExtremeValueModel):
    model: AbstractExtremeValueModel
    theta: list[float]
    def __init__(self, model: AbstractExtremeValueModel, theta: list[float]):
        self.model, self.theta = model, theta
    def __repr__(self) -> str:
        return f"MaximumLikelihoodAbstractExtremeValueModel \n\tmodel :\n\t\t{self.model} \n\tθ̂  : {self.theta}"
    def py_aev_to_jl_aev(self):
        return py_maximumlikelihood_aev_to_jl_aev(self)

#TODO: test when py_blockmaxima_to_jl_blockmaxima is fixed
def py_maximumlikelihood_aev_to_jl_aev(py_model: MaximumLikelihoodAbstractExtremeValueModel):
    #TODO: smarter way of checking wether it's a BlockMaxima or a ThresholdExceedance
    if(isinstance(py_model.model, BlockMaxima)):
        jl_model = py_blockmaxima_to_jl_blockmaxima(py_model.model)
    elif(isinstance(py_model.model, ThresholdExceedance)):
        jl_model = py_threshold_exceedance_to_jl_threshold_exceedance(py_model.model)
    jl_theta = jl_convert(jl.Vector[jl.Float64], py_model.theta)
    return Extremes.MaximumLikelihoodAbstractExtremeValueModel(jl_model,jl_theta)

#TODO: smarter way of checking wether it's a BlockMaxima or a ThresholdExceedance
def jl_maximumlikelihood_aev_to_py_aev(jl_model) -> MaximumLikelihoodAbstractExtremeValueModel:
    if (jl.typeof(jl_model) == Extremes.BlockMaxima): 
        py_model = jl_blockmaxima_to_py_blockmaxima(jl_model.model)
    else:
        py_model = jl_threshold_exceedance_to_py_threshold_exceedance(jl_model.model)
    py_theta = jl_vector_to_py_list(getattr(jl_model, "θ̂"))
    return MaximumLikelihoodAbstractExtremeValueModel(py_model, py_theta)


class BayesianAbstractExtremeValueModel(AbstractFittedExtremeValueModel):
    model: AbstractExtremeValueModel
    #TODO: give type to sim, which in julia is a MambaLite.Chains
    def __init__(self, model: AbstractExtremeValueModel, sim):
        self.model, self.sim = model, sim
    def __repr__(self) -> str:
        return f"BayesianAbstractExtremeValueModel \nmodel :\n {self.model} \nsim : \n{self.sim}"
    def py_aev_to_jl_aev(self):
        return py_bayesian_aev_to_jl_aev(self)
    
def py_bayesian_aev_to_jl_aev(py_model: BayesianAbstractExtremeValueModel):
    #TODO: smarter way of checking wether it's a BlockMaxima or a ThresholdExceedance
    if(isinstance(py_model.model, BlockMaxima)):
        jl_model = py_blockmaxima_to_jl_blockmaxima(py_model.model)
    elif(isinstance(py_model.model, ThresholdExceedance)):
        jl_model = py_threshold_exceedance_to_jl_threshold_exceedance(py_model.model)
    jl_sim = py_model.sim
    return Extremes.BayesianAbstractExtremeValueModel(jl_model, jl_sim)

#TODO: smarter way of checking wether it's a BlockMaxima or a ThresholdExceedance
def jl_bayesian_aev_to_py_aev(jl_model) -> BayesianAbstractExtremeValueModel:
    if (jl.typeof(jl_model) == Extremes.BlockMaxima): 
        py_model = jl_blockmaxima_to_py_blockmaxima(jl_model.model)
    else:
        py_model = jl_threshold_exceedance_to_py_threshold_exceedance(jl_model.model)
    py_sim = jl_model.sim
    return BayesianAbstractExtremeValueModel(py_model, py_sim)

class PwmAbstractExtremeValueModel(AbstractFittedExtremeValueModel):
    model: AbstractExtremeValueModel
    theta: list[float]
    def __init__(self, model: AbstractExtremeValueModel, theta: list[float]):
        self.model, self.theta = model, theta
    def __repr__(self) -> str:
        return f"PwmAbstractExtremeValueModel \nmodel :\n {self.model} \nθ̂  : \n{self.theta}"
    def py_aev_to_jl_aev(self):
        return py_pwm_aev_to_jl_aev(self)
    
#TODO: test when py_blockmaxima_to_jl_blockmaxima is fixed
def py_pwm_aev_to_jl_aev(py_model: MaximumLikelihoodAbstractExtremeValueModel):
    return py_maximumlikelihood_aev_to_jl_aev(py_model)

#TODO: smarter way of checking wether it's a BlockMaxima or a ThresholdExceedance
def jl_pwm_aev_to_py_aev(jl_model) -> PwmAbstractExtremeValueModel:
    if (jl.typeof(jl_model) == Extremes.BlockMaxima): 
        py_model = jl_blockmaxima_to_py_blockmaxima(jl_model.model)
    else:
        py_model = jl_threshold_exceedance_to_py_threshold_exceedance(jl_model.model)
    py_theta = jl_vector_to_py_list(getattr(jl_model, "θ̂"))
    return PwmAbstractExtremeValueModel(py_model, py_theta)










