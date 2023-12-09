import pytest

from xhydro.calibration import (perform_calibration, 
                                get_objective_function_value, 
                                dummy_model, 
                                spot_setup)
import numpy as np


def test_spotpy_calibration():
    
    """
    Make sure the calibration works for a few test cases
    """
    bounds_low = np.array([0,0,0])
    bounds_high = np.array([10,10,10])
    
    model_config={
       'precip':np.array([10,11,12,13,14,15]),
       'temperature':np.array([10,3,-5,1,15,0]),
       'Qobs':np.array([120,130,140,150,160,170]),
       'drainage_area':np.array([10]),
       'model_name':'Dummy',
    }
       
    best_parameters, best_simulation, best_objfun = perform_calibration(model_config, 'nse', maximize = True, bounds_low=bounds_low, bounds_high=bounds_high, evaluations=1000, algorithm="DDS")

    # Test that the results have the same size as expected (number of parameters)
    assert len(best_parameters) == len(bounds_high)
    
    # Test that the objective function is calculated correctly
    objfun = get_objective_function_value(model_config['Qobs'], best_simulation,best_parameters, obj_func='nse')
    assert objfun == best_objfun
        
    # Test dummy model response
    model_config['parameters']=[5,5,5]
    Qsim = dummy_model(model_config)
    assert Qsim[3]==3500.00
    
    
    
    