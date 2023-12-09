#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:48:34 2023

@author: ets
"""

import numpy as np
from spotpy.objectivefunctions import rmse
from spotpy import analyser
from spotpy.parameter import Uniform
import spotpy
import hydroeval as he
from xhydro.hydrological_modelling import  hydrological_model_selector

class spot_setup(object):
    
    
    def __init__(self, model_config, bounds_high, bounds_low, obj_func=None):
        
        self.model_config = model_config
        self.obj_func = obj_func
    
        
        # Create the sampler for each parameter based on the bounds
        self.parameters=[Uniform("param"+str(i), bounds_low[i], bounds_high[i]) for i in range(0,len(bounds_high))]
        
    def simulation(self, x):
        
        self.model_config['parameters'] = x        
                
        return hydrological_model_selector(self.model_config)
        
    
    def evaluation(self):
        
        return self.model_config['Qobs']
        
    
    def objectivefunction(self, simulation, evaluation, params=None, transform=None, epsilon=None):
        
        return get_objective_function_value(evaluation, simulation, params, transform, epsilon, self.obj_func)
        
        
def perform_calibration(model_config, evaluation_metric, maximize, bounds_high, bounds_low, evaluations, algorithm="DDS"):
    
    """
    'model_config' is the model configuration object (dict type) that contains all info to run the model.
    The model function called to run this model should always use this object and read-in data it requires.
    It will be up to the user to provide the data that the model requires.
    It should also contain parameters, even NaN or dummy ones, and the optimizer will update these.
    """
        
    spotpy_setup = spot_setup(model_config, bounds_high=bounds_high, bounds_low=bounds_low, obj_func=evaluation_metric)
    
    if algorithm == "DDS":
        sampler = spotpy.algorithms.dds(spotpy_setup, dbname="DDS_optim", dbformat='ram', save_sim=False)
        sampler.sample(evaluations, trials=1)
        
    elif algorithm == "SCEUA":
        sampler = spotpy.algorithms.sceua(spotpy_setup, dbname="SCEUA_optim", dbformat='ram', save_sim=False)
        sampler.sample(evaluations, ngs=7, kstop=3, peps=0.1, pcento=0.1)
    
    results = sampler.getdata()
        
    best_parameters = analyser.get_best_parameterset(results,like_index=1, maximize=maximize)
    best_parameters=[best_parameters[0][i] for i in range(0,len(best_parameters[0]))]

    if maximize==True:
        bestindex, bestobjf = analyser.get_maxlikeindex(results)
    else:
        bestindex, bestobjf = analyser.get_minlikeindex(results)
    
    best_model_run = results[bestindex]
    
    model_config['parameters'] = best_parameters
    
    Qsim = hydrological_model_selector(model_config)
    
    #objfun = get_objective_function_value(model_config['Qobs'], Qsim, best_parameters, obj_func=evaluation_metric)
    
    return best_parameters, Qsim, bestobjf
       
 
    
def transform_flows(Qobs, Qsim, transform=None, epsilon=None):
    
    # This subset of code is taken from the hydroeval package:
    # https://github.com/ThibHlln/hydroeval/blob/main/hydroeval/hydroeval.py
    #
    # generate a subset of simulation and evaluation series where evaluation 
    # data is available
    Qsim = Qsim[~np.isnan(Qobs[:, 0]), :]
    Qobs = Qobs[~np.isnan(Qobs[:, 0]), :]
    
    # transform the flow series if required
    if transform == 'log':  # log transformation
        if not epsilon:
            # determine an epsilon value to avoid zero divide
            # (following recommendation in Pushpalatha et al. (2012))   
            epsilon = 0.01 * np.mean(Qobs)
        Qobs, Qsim = np.log(Qobs + epsilon), np.log(Qsim + epsilon)

    elif transform == 'inv':  # inverse transformation
        if not epsilon:   
            # determine an epsilon value to avoid zero divide
            # (following recommendation in Pushpalatha et al. (2012))
            epsilon = 0.01 * np.mean(Qobs)
        Qobs, Qsim = 1.0 / (Qobs + epsilon), 1.0 / (Qsim + epsilon)
        
    elif transform == 'sqrt':  # square root transformation
        Qobs, Qsim = np.sqrt(Qobs), np.sqrt(Qsim)
    
    return Qobs, Qsim


def get_objective_function_value(Qobs, Qsim, params, transform=None, epsilon=None, obj_func=None):
        
    # Transform flows if needed
    if transform:
        Qobs, Qsim = transform_flows(Qobs, Qsim, transform, epsilon)

    # Default objective function if none provided by the user
    if not obj_func:
        like = rmse(Qobs,Qsim)
        
    # Popular ones we can use hydroeval package
    elif obj_func.lower() in ['nse','kge','kgeprime','kgenp','rmse','mare','pbias']:
        
        # Get the correct function from the package
        like = he.evaluator(getattr(he,obj_func.lower()), simulations=Qsim, evaluation=Qobs)
        
    # In neither of these cases, use the obj_func method from spotpy
    else:
        like = obj_func(Qobs, Qsim)
        
    # Return results
    return like