from xhydro.extreme_value_analysis.structures.abstract_extreme_value_model import *

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