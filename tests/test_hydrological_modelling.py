"""Test suite for hydrological modelling in hydrological_modelling.py"""

import numpy as np
import pytest

from xhydro.modelling.hydrological_modelling import (
    get_hydrological_model_inputs,
    run_hydrological_model,
)


def test_hydrological_modelling():
    """Test the hydrological models as they become online"""
    # Test the dummy model
    model_config = {
        "precip": np.array([10, 11, 12, 13, 14, 15]),
        "temperature": np.array([10, 3, -5, 1, 15, 0]),
        "qobs": np.array([120, 130, 140, 150, 160, 170]),
        "drainage_area": np.array([10]),
        "model_name": "Dummy",
        "parameters": np.array([5, 5, 5]),
    }
    qsim = run_hydrological_model(model_config)
    assert qsim["qsim"].values[3] == 3500.00

    # Test the exceptions for new models
    model_config.update(model_name="ADD_OTHER_HERE")
    qsim = run_hydrological_model(model_config)
    assert qsim == 0


def test_import_unknown_model():
    """Test for unknown model"""
    with pytest.raises(NotImplementedError) as pytest_wrapped_e:
        model_config = {"model_name": "fake_model"}
        _ = run_hydrological_model(model_config)
        assert pytest_wrapped_e.type == NotImplementedError


def test_get_unknown_model_requirements():
    """Test for required inputs for models with unknown name"""
    with pytest.raises(NotImplementedError) as pytest_wrapped_e:
        model_name = "fake_model"
        _ = get_hydrological_model_inputs(model_name)
        assert pytest_wrapped_e.type == NotImplementedError


def test_get_model_requirements():
    """Test for required inputs for models"""
    model_name = "Dummy"
    required_config = get_hydrological_model_inputs(model_name)
    print(required_config.keys())
    assert len(required_config.keys()) == 4
