"""Test suite for the calibration algorithm in calibration.py."""

# Also tests the dummy model implementation.
import numpy as np
import pytest

from xhydro.modelling.calibration import perform_calibration
from xhydro.modelling.hydrological_modelling import dummy_model
from xhydro.modelling.obj_funcs import get_objective_function, transform_flows


def test_spotpy_calibration():
    """Make sure the calibration works under possible test cases."""
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


def test_calibration_failures():
    """Test for the calibration algorithm failure modes"""
    bounds_low = np.array([0, 0, 0])
    bounds_high = np.array([10, 10, 10])
    model_config = {
        "precip": np.array([10, 11, 12, 13, 14, 15]),
        "temperature": np.array([10, 3, -5, 1, 15, 0]),
        "Qobs": np.array([120, 130, 140, 150, 160, 170]),
        "drainage_area": np.array([10]),
        "model_name": "Dummy",
    }

    # Test Qobs different length than Qsim
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        best_parameters_negative, best_simulation, best_objfun = perform_calibration(
            model_config.update(Qobs=np.array([100, 100, 100])),
            "nse",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=1000,
            algorithm="OTHER",
        )
        assert pytest_wrapped_e.type == SystemExit

        # Test mask not 1 or 0
        mask = np.array([0, 0, 0, 0.5, 1, 1])
        best_parameters_negative, best_simulation, best_objfun = perform_calibration(
            model_config,
            "nse",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=1000,
            algorithm="DDS",
            mask=mask,
        )
        assert pytest_wrapped_e.type == SystemExit

        # test not same length in mask
        mask = np.array([0, 0, 0, 1, 1])
        best_parameters_negative, best_simulation, best_objfun = perform_calibration(
            model_config,
            "nse",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=1000,
            algorithm="DDS",
            mask=mask,
        )
        assert pytest_wrapped_e.type == SystemExit

        # Test objective function fail is caught
        mask = np.array([0, 0, 0, 0, 1, 1])
        best_parameters_negative, best_simulation, best_objfun = perform_calibration(
            model_config,
            "nse_fake",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=1000,
            algorithm="DDS",
            mask=mask,
        )
        assert pytest_wrapped_e.type == SystemExit

        # Test objective function that cannot be minimized
        best_parameters_negative, best_simulation, best_objfun = perform_calibration(
            model_config,
            "bias",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=1000,
            algorithm="DDS",
            mask=mask,
        )
        assert pytest_wrapped_e.type == SystemExit


def test_transform():
    """Test the flow transformer"""
    Qsim = np.array([10, 10, 10])
    Qobs = np.array([5, 5, 5])

    Qsim_r, Qobs_r = transform_flows(Qsim, Qobs, transform="inv", epsilon=0.01)
    np.testing.assert_array_almost_equal(Qsim_r[1], 0.0995024, 6)
    np.testing.assert_array_almost_equal(Qobs_r[1], 0.1980198, 6)

    Qsim_r, Qobs_r = transform_flows(Qsim, Qobs, transform="sqrt")
    np.testing.assert_array_almost_equal(Qsim_r[1], 3.1622776, 6)
    np.testing.assert_array_almost_equal(Qobs_r[1], 2.2360679, 6)

    Qsim_r, Qobs_r = transform_flows(Qsim, Qobs, transform="log", epsilon=0.01)
    np.testing.assert_array_almost_equal(Qsim_r[1], 2.3075726, 6)
    np.testing.assert_array_almost_equal(Qobs_r[1], 1.6193882, 6)

    # Test Qobs different length than Qsim
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Qobs_r, Qobs_r = transform_flows(Qsim, Qobs, transform="a", epsilon=0.01)
        assert pytest_wrapped_e.type == SystemExit
