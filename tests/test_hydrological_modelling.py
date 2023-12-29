"""Test suite for hydrological modelling in hydrological_modelling.py"""
import numpy as np
import pytest

from xhydro.modelling.hydrological_modelling import hydrological_model_selector


def test_hydrological_modelling():
    """Test the hydrological models as they become online"""
    # Test the dummy model
    model_config = {
        "precip": np.array([10, 11, 12, 13, 14, 15]),
        "temperature": np.array([10, 3, -5, 1, 15, 0]),
        "Qobs": np.array([120, 130, 140, 150, 160, 170]),
        "drainage_area": np.array([10]),
        "model_name": "Dummy",
        "parameters": np.array([5, 5, 5]),
    }
    Qsim = hydrological_model_selector(model_config)
    assert Qsim[3] == 3500.00

    # Test the exceptions for new models
    model_config.update(model_name="ADD_OTHER_HERE")
    Qsim = hydrological_model_selector(model_config)
    assert Qsim == 0


def import_unknown_model():
    """Test for unknown model"""
    with pytest.raises(NotImplementedError):
        model_config = {"model_name": "fake_model"}
        Qsim = hydrological_model_selector(model_config)
        assert Qsim is None
