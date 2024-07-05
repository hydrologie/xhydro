from xhydro_temp.extreme_value_analysis.structures.abstract_extreme_value_model import *

class AbstractFittedExtremeValueModel:
    pass

class MaximumLikelihoodAbstractExtremeValueModel(AbstractFittedExtremeValueModel):
    model: AbstractExtremeValueModel
    theta: list[float]
    def __init__(self, model: AbstractExtremeValueModel, theta: list[float]):
        self.model, self.theta = model, theta
    def __repr__(self) -> str:
        return f"MaximumLikelihoodAbstractExtremeValueModel \n\tmodel :\n\t\t{self.model} \n\tθ̂  : {self.theta}"

class BayesianAbstractExtremeValueModel(AbstractFittedExtremeValueModel):
    model: AbstractExtremeValueModel
    #TODO: give type to sim, which in julia is a MambaLite.Chains
    def __init__(self, model: AbstractExtremeValueModel, sim):
        self.model, self.sim = model, sim
    def __repr__(self) -> str:
        return f"BayesianAbstractExtremeValueModel \nmodel :\n {self.model} \nsim : \n{self.sim}"

class PwmAbstractExtremeValueModel(AbstractFittedExtremeValueModel):
    model: AbstractExtremeValueModel
    theta: list[float]
    def __init__(self, model: AbstractExtremeValueModel, theta: list[float]):
        self.model, self.theta = model, theta
    def __repr__(self) -> str:
        return f"PwmAbstractExtremeValueModel \nmodel :\n {self.model} \nθ̂  : \n{self.theta}"

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
