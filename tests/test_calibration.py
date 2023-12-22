"""Test suite for the calibration algorithm in calibration.py."""

# Also tests the dummy model implementation.

import numpy as np

from xhydro.modelling.calibration import perform_calibration
from xhydro.modelling.hydrological_modelling import dummy_model
from xhydro.modelling.obj_funcs import get_objective_function


def test_spotpy_calibration():
    """Make sure the calibration works for a few test cases."""
    bounds_low = np.array([0, 0, 0])
    bounds_high = np.array([10, 10, 10])

    model_config = {
        "precip": np.array([10, 11, 12, 13, 14, 15]),
        "temperature": np.array([10, 3, -5, 1, 15, 0]),
        "Qobs": np.array([120, 130, 140, 150, 160, 170]),
        "drainage_area": np.array([10]),
        "model_name": "Dummy",
    }
    
    mask = np.array([0, 0, 0, 0, 1, 1])
    
    best_parameters, best_simulation, best_objfun = perform_calibration(
        model_config,
        "mae",
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        evaluations=1000,
        algorithm="DDS",
        mask=mask,
    )

    # Test that the results have the same size as expected (number of parameters)
    assert len(best_parameters) == len(bounds_high)

    # Test that the objective function is calculated correctly
    objfun = get_objective_function( 
                                    model_config["Qobs"], 
                                    best_simulation, 
                                    obj_func="mae",
                                    mask=mask,
                                    )
    
    assert objfun == best_objfun

    # Test dummy model response
    model_config["parameters"] = [5, 5, 5]
    Qsim = dummy_model(model_config)
    assert Qsim[3] == 3500.00