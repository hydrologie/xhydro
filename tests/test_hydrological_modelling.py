"""Test suite for hydrological modelling in hydrological_modelling.py"""

import numpy as np
import pytest

from xhydro.modelling import get_hydrological_model_inputs, hydrological_model


class TestHydrologicalModelling:
    def test_hydrological_modelling(self):
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
        qsim = hydrological_model(model_config).run()
        np.testing.assert_array_equal(qsim["streamflow"].values[3], 3500.00)

    def test_import_unknown_model(self):
        """Test for unknown model"""
        with pytest.raises(NotImplementedError) as pytest_wrapped_e:
            model_config = {"model_name": "fake_model"}
            _ = hydrological_model(model_config).run()
            assert pytest_wrapped_e.type == NotImplementedError

    def test_missing_name(self):
        with pytest.raises(ValueError, match="The model name must be provided"):
            model_config = {"parameters": [1, 2, 3]}
            hydrological_model(model_config).run()


class TestHydrologicalModelRequirements:
    def test_get_unknown_model_requirements(self):
        """Test for required inputs for models with unknown name"""
        with pytest.raises(NotImplementedError) as pytest_wrapped_e:
            model_name = "fake_model"
            _ = get_hydrological_model_inputs(model_name)
            assert pytest_wrapped_e.type == NotImplementedError

    @pytest.mark.parametrize("model_name", ["Dummy", "Hydrotel", "GR4JCN"])
    def test_get_model_requirements(self, model_name):
        """Test for required inputs for models"""
        expected_keys = {"Dummy": (6, 6), "Hydrotel": (8, 3), "GR4JCN": (5, 5)}

        all_config, _ = get_hydrological_model_inputs(model_name)
        assert len(all_config.keys()) == expected_keys[model_name][0]

        all_config, _ = get_hydrological_model_inputs(model_name, required_only=True)
        assert len(all_config.keys()) == expected_keys[model_name][1]


class TestDummyModel:
    def test_inputs(self):
        model_config = {
            "model_name": "Dummy",
            "precip": np.array([10, 11, 12, 13, 14, 15]),
            "temperature": np.array([10, 3, -5, 1, 15, 0]),
            "qobs": np.array([120, 130, 140, 150, 160, 170]),
            "drainage_area": np.array([10]),
            "parameters": np.array([5, 5, 5]),
        }
        dummy = hydrological_model(model_config)
        ds_in = dummy.get_inputs()
        np.testing.assert_array_equal(ds_in.precip, model_config["precip"])
        np.testing.assert_array_equal(ds_in.temperature, model_config["temperature"])
        assert len(ds_in.time) == len(model_config["precip"])

    def test_streamflow(self):
        model_config = {
            "model_name": "Dummy",
            "precip": np.array([10, 11, 12, 13, 14, 15]),
            "temperature": np.array([10, 3, -5, 1, 15, 0]),
            "qobs": np.array([120, 130, 140, 150, 160, 170]),
            "drainage_area": np.array([10]),
            "parameters": np.array([5, 5, 5]),
        }
        dummy = hydrological_model(model_config)
        ds_out = dummy.get_streamflow()
        np.testing.assert_array_equal(ds_out["streamflow"].values[3], 3500.00)
        assert dummy.qsim.equals(ds_out)
        assert dummy.get_streamflow().equals(ds_out)
