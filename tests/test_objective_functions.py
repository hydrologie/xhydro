"""Test suite for the objective functions in obj_funcs.py."""
import numpy as np
import pytest

from xhydro.modelling.obj_funcs import (
    get_objective_function,
    get_objfun_minimize_or_maximize,
)


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


def test_objective_function_failure_data_length():
    """Test for the objective function calculation failure mode:
    Qobs and Qsim length are different
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objective_function(
            np.array([100, 110]),
            np.array([100, 110, 120]),
            obj_func="mae",
        )
        assert pytest_wrapped_e.type == SystemExit


def test_objective_function_failure_mask_length():
    """Test for the objective function calculation failure mode:
    Qobs and mask length are different
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objective_function(
            np.array([100, 100, 100]),
            np.array([100, 110, 120]),
            obj_func="mae",
            mask=np.array([0, 1, 0, 0]),
        )
        assert pytest_wrapped_e.type == SystemExit


def test_objective_function_failure_unknown_objfun():
    """Test for the objective function calculation failure mode:
    Objective function is unknown
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objective_function(
            np.array([100, 100, 100]),
            np.array([100, 110, 120]),
            obj_func="fake",
        )
        assert pytest_wrapped_e.type == SystemExit


def test_objective_function_failure_mask_contents():
    """Test for the objective function calculation failure mode:
    Mask contains other than 0 and 1
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objective_function(
            np.array([100, 100, 100]),
            np.array([100, 110, 120]),
            obj_func="mae",
            mask=np.array([0, 0.5, 1]),
        )
        assert pytest_wrapped_e.type == SystemExit


def test_maximizer_objfun_failure_modes_bias():
    """Test for maximize-minimize failure mode:
    Use of bias objfun which is unbounded
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objfun_minimize_or_maximize(obj_func="bias")
        assert pytest_wrapped_e.type == SystemExit


def test_maximizer_objfun_failure_modes_pbias():
    """Test for maximize-minimize failure mode:
    Use of pbias objfun which is unbounded
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objfun_minimize_or_maximize(obj_func="pbias")
        assert pytest_wrapped_e.type == SystemExit


def test_maximizer_objfun_failure_modes_volume_error():
    """Test for maximize-minimize failure mode:
    Use of volume_error objfun which is unbounded
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objfun_minimize_or_maximize(obj_func="volume_error")
        assert pytest_wrapped_e.type == SystemExit


def test_maximizer_objfun_failure_modes_unknown_metric():
    """Test for maximize-minimize failure mode:
    Use of unknown objfun
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        _ = get_objfun_minimize_or_maximize(obj_func="unknown_of")
        assert pytest_wrapped_e.type == SystemExit
