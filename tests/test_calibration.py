"""Test suite for the calibration algorithm in calibration.py."""

# Also tests the dummy model implementation.
import numpy as np
import pytest

from xhydro.modelling.calibration import perform_calibration
from xhydro.modelling.hydrological_modelling import _dummy_model
from xhydro.modelling.obj_funcs import get_objective_function, transform_flows


def test_spotpy_calibration():
    """Make sure the calibration works under possible test cases."""
    bounds_low = np.array([0, 0, 0])
    bounds_high = np.array([10, 10, 10])

    model_config = {
        "precip": np.array([10, 11, 12, 13, 14, 15]),
        "temperature": np.array([10, 3, -5, 1, 15, 0]),
        "qobs": np.array([120, 130, 140, 150, 160, 170]),
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
        sampler_kwargs=dict(trials=1),
    )

    # Test that the results have the same size as expected (number of parameters)
    assert len(best_parameters) == len(bounds_high)

    # Test that the objective function is calculated correctly
    objfun = get_objective_function(
        model_config["qobs"],
        best_simulation,
        obj_func="mae",
        mask=mask,
    )

    assert objfun == best_objfun

    # Test dummy model response
    model_config["parameters"] = [5, 5, 5]
    qsim = _dummy_model(model_config)
    assert qsim["qsim"].values[3] == 3500.00

    # Also test to ensure SCEUA and take_minimize is required.
    best_parameters_sceua, best_simulation, best_objfun = perform_calibration(
        model_config,
        "mae",
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        evaluations=10,
        algorithm="SCEUA",
    )

    assert len(best_parameters_sceua) == len(bounds_high)

    # Also test to ensure SCEUA and take_minimize is required.
    best_parameters_negative, best_simulation, best_objfun = perform_calibration(
        model_config,
        "nse",
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        evaluations=10,
        algorithm="SCEUA",
    )
    assert len(best_parameters_negative) == len(bounds_high)

    # Test to see if transform works
    best_parameters_transform, best_simulation, best_objfun = perform_calibration(
        model_config,
        "nse",
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        evaluations=10,
        algorithm="SCEUA",
        transform="inv",
        epsilon=0.01,
    )
    assert len(best_parameters_transform) == len(bounds_high)


def test_calibration_failure_mode_unknown_optimizer():
    """Test for maximize-minimize failure mode:
    use "OTHER" optimizer, i.e. an unknown optimizer. Should fail.
    """
    bounds_low = np.array([0, 0, 0])
    bounds_high = np.array([10, 10, 10])
    model_config = {
        "precip": np.array([10, 11, 12, 13, 14, 15]),
        "temperature": np.array([10, 3, -5, 1, 15, 0]),
        "qobs": np.array([120, 130, 140, 150, 160, 170]),
        "drainage_area": np.array([10]),
        "model_name": "Dummy",
    }
    with pytest.raises(NotImplementedError) as pytest_wrapped_e:
        best_parameters_transform, best_simulation, best_objfun = perform_calibration(
            model_config,
            "nse",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=10,
            algorithm="OTHER",
        )
        assert pytest_wrapped_e.type == NotImplementedError


def test_transform():
    """Test the flow transformer"""
    qsim = np.array([10, 10, 10])
    qobs = np.array([5, 5, 5])

    qsim_r, qobs_r = transform_flows(qsim, qobs, transform="inv", epsilon=0.01)
    np.testing.assert_array_almost_equal(qsim_r[1], 0.0995024, 6)
    np.testing.assert_array_almost_equal(qobs_r[1], 0.1980198, 6)

    qsim_r, qobs_r = transform_flows(qsim, qobs, transform="sqrt")
    np.testing.assert_array_almost_equal(qsim_r[1], 3.1622776, 6)
    np.testing.assert_array_almost_equal(qobs_r[1], 2.2360679, 6)

    qsim_r, qobs_r = transform_flows(qsim, qobs, transform="log", epsilon=0.01)
    np.testing.assert_array_almost_equal(qsim_r[1], 2.3075726, 6)
    np.testing.assert_array_almost_equal(qobs_r[1], 1.6193882, 6)

    # Test Qobs different length than Qsim
    with pytest.raises(NotImplementedError) as pytest_wrapped_e:
        qobs_r, qobs_r = transform_flows(qsim, qobs, transform="a", epsilon=0.01)
        assert pytest_wrapped_e.type == NotImplementedError
