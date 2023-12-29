"""Test suite for the objective functions in obj_funcs.py."""

import numpy as np

from xhydro.modelling.obj_funcs import get_objective_function


def test_obj_funcs():
    """Series of tests to test all objective functions with fast test data"""
    Qobs = np.array([120, 130, 140, 150, 160, 170])
    Qsim = np.array([120, 125, 145, 140, 140, 180])

    # Test that the objective function is calculated correctly
    objfun = get_objective_function(Qobs, Qsim, obj_func="abs_bias")
    np.testing.assert_array_almost_equal(objfun, 3.3333333333333335, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="abs_pbias")
    np.testing.assert_array_almost_equal(objfun, 2.2988505747126435, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="abs_volume_error")
    np.testing.assert_array_almost_equal(objfun, 0.022988505747126436, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="agreement_index")
    np.testing.assert_array_almost_equal(objfun, 0.9171974522292994, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="bias")
    np.testing.assert_array_almost_equal(objfun, -3.3333333333333335, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="correlation_coeff")
    np.testing.assert_array_almost_equal(objfun, 0.8599102447336393, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="kge")
    np.testing.assert_array_almost_equal(objfun, 0.8077187696552522, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="kge_mod")
    np.testing.assert_array_almost_equal(objfun, 0.7888769531580001, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="mae")
    np.testing.assert_array_almost_equal(objfun, 8.333333333333334, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="mare")
    np.testing.assert_array_almost_equal(objfun, 0.05747126436781609, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="mse")
    np.testing.assert_array_almost_equal(objfun, 108.33333333333333, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="nse")
    np.testing.assert_array_almost_equal(objfun, 0.6285714285714286, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="pbias")
    np.testing.assert_array_almost_equal(objfun, -2.2988505747126435, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="r2")
    np.testing.assert_array_almost_equal(objfun, 0.7394456289978675, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="rmse")
    np.testing.assert_array_almost_equal(objfun, 10.408329997330663, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="rrmse")
    np.testing.assert_array_almost_equal(objfun, 0.07178158618848733, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="rsr")
    np.testing.assert_array_almost_equal(objfun, 0.6094494002200439, 8)

    objfun = get_objective_function(Qobs, Qsim, obj_func="volume_error")
    np.testing.assert_array_almost_equal(objfun, -0.022988505747126436, 8)
