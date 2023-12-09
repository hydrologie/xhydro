import numpy as np


def hydrological_model_selector(model_config):

    if model_config['model_name'] == 'Dummy':
        Qsim = dummy_model(model_config)
        
        # ADD OTHER MODELS HERE

    return Qsim


def dummy_model(model_config):    
    
    """
    Dummy model to show the implementation we should be aiming for. Each model 
    will have its own required data that users can pass. We could envision a setup
    where a class generates the model_config object for all model formats making it 
    possible to replace models on the fly
    """
    
    precip = model_config['precip']
    temperature = model_config['temperature']
    drainage_area = model_config['drainage_area']
    parameters = model_config['parameters']
    
    Q = np.empty(len(precip))
    
    for t in range(0,len(precip)):
     
        Q[t]= (precip[t] * parameters[0] + abs(temperature[t])*parameters[1])* parameters[2] * drainage_area
        
        
    return Q


