"""Test suite for the calibration algorithm in calibration.py."""

# Also tests the dummy model implementation.
import datetime as dt
import warnings
from copy import deepcopy

import numpy as np
import pooch
import pytest
import xarray as xr
from raven_hydro import __raven_version__

from xhydro.modelling import hydrological_model
from xhydro.modelling.calibration import perform_calibration
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
    qsim = hydrological_model(model_config).run()
    assert qsim["streamflow"].values[3] == 3500.00

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
    with pytest.raises(NotImplementedError):
        best_parameters_transform, best_simulation, best_objfun = perform_calibration(
            model_config,
            "nse",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=10,
            algorithm="OTHER",
        )


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
    with pytest.raises(NotImplementedError):
        qobs_r, qobs_r = transform_flows(qsim, qobs, transform="a", epsilon=0.01)


class TestRavenpyModelCalibration:
    """Test calibration of RavenPy models."""

    # Set Github URL for getting files for tests
    GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
    BRANCH_OR_COMMIT_HASH = "main"

    # Get data from xhydro-testdata repo
    meteo_file = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/ravenpy/ERA5_Riviere_Rouge_global.nc",
        known_hash="md5:de985fa27ddceac690aeb34182a93f11",
    )
    qobs_path = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/ravenpy/Debit_Riviere_Rouge.nc",
        known_hash="md5:5b0feedc34333244b1d9e9c251323478",
    )

    # List of types of data provided to Raven in the meteo file
    data_type = ["TEMP_MAX", "TEMP_MIN", "PRECIP"]

    # Alternate names in the netcdf file (for variables that have different names, map to the name in the file).
    alt_names_meteo = {"TEMP_MIN": "tmin", "TEMP_MAX": "tmax", "PRECIP": "pr"}
    alt_names_flow = "qobs"

    start_date = dt.datetime(1985, 1, 1)
    end_date = dt.datetime(1986, 12, 31)

    model_config = {
        "meteo_file": meteo_file,
        "qobs_path": qobs_path,
        "qobs": xr.open_dataset(qobs_path)
        .qobs.sel(time=slice(start_date, end_date))
        .values,
        "drainage_area": np.array([100.0]),
        "elevation": np.array([250.5]),
        "latitude": np.array([46.0]),
        "longitude": np.array([-80.75]),
        "start_date": start_date,
        "end_date": end_date,
        "data_type": data_type,
        "alt_names_meteo": alt_names_meteo,
        "alt_names_flow": alt_names_flow,
    }

    # Station properties. Using the same as for the catchment, but could be different.
    meteo_station_properties = {
        "ALL": {
            "elevation": model_config["elevation"],
            "latitude": model_config["latitude"],
            "longitude": model_config["longitude"],
        }
    }

    model_config.update({"meteo_station_properties": meteo_station_properties})

    def test_ravenpy_gr4jcn_calibration(self):
        """Test for GR4JCN ravenpy model"""
        bounds_low = [0.01, -15.0, 10.0, 0.0, 1.0, 0.0]
        bounds_high = [2.5, 10.0, 700.0, 7.0, 30.0, 1.0]

        model_config = deepcopy(self.model_config)
        model_config.update({"model_name": "GR4JCN"})

        best_parameters, best_simulation, best_objfun = perform_calibration(
            model_config,
            "mae",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=8,
            algorithm="DDS",
            sampler_kwargs=dict(trials=1),
        )

        # Test that the results have the same size as expected (number of parameters)
        assert len(best_parameters) == len(bounds_high)

    def test_ravenpy_hmets_calibration(self):
        """Test for HMETS ravenpy model"""
        bounds_low = [
            0.3,
            0.01,
            0.5,
            0.15,
            0.0,
            0.0,
            -2.0,
            0.01,
            0.0,
            0.01,
            0.005,
            -5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00001,
            0.0,
            0.00001,
            0.0,
            0.0,
        ]
        bounds_high = [
            20.0,
            5.0,
            13.0,
            1.5,
            20.0,
            20.0,
            3.0,
            0.2,
            0.1,
            0.3,
            0.1,
            2.0,
            5.0,
            1.0,
            3.0,
            1.0,
            0.02,
            0.1,
            0.01,
            0.5,
            2.0,
        ]

        model_config = deepcopy(self.model_config)
        model_config.update(
            {
                "model_name": "HMETS",
            }
        )

        best_parameters, best_simulation, best_objfun = perform_calibration(
            model_config,
            "mae",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=8,
            algorithm="DDS",
            sampler_kwargs=dict(trials=1),
        )

        # Test that the results have the same size as expected (number of parameters)
        assert len(best_parameters) == len(bounds_high)

    def test_ravenpy_mohyse_calibration(self):
        """Test for MOHYSE ravenpy model"""
        bounds_low = [0.01, 0.01, 0.01, -5.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        bounds_high = [20.0, 1.0, 20.0, 5.0, 0.5, 1.0, 1.0, 1.0, 15.0, 15.0]

        model_config = deepcopy(self.model_config)
        model_config.update({"model_name": "Mohyse"})

        best_parameters, best_simulation, best_objfun = perform_calibration(
            model_config,
            "mae",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=8,
            algorithm="DDS",
            sampler_kwargs=dict(trials=1),
        )

        # Test that the results have the same size as expected (number of parameters)
        assert len(best_parameters) == len(bounds_high)

    @pytest.mark.skip(
        reason="Weird error with negative simulated PET in ravenpy for HBVEC."
    )
    def test_ravenpy_hbvec_calibration(self):
        """Test for HBV-EC ravenpy model"""
        bounds_low = [
            -3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.3,
            0.0,
            0.0,
            0.01,
            0.05,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.01,
            0.0,
            0.05,
            0.8,
            0.8,
        ]
        bounds_high = [
            3.0,
            8.0,
            8.0,
            0.1,
            1.0,
            1.0,
            7.0,
            100.0,
            1.0,
            0.1,
            6.0,
            5.0,
            5.0,
            0.2,
            1.0,
            30.0,
            3.0,
            2.0,
            1.0,
            1.5,
            1.5,
        ]

        model_config = deepcopy(self.model_config)
        model_config.update({"model_name": "HBVEC"})

        best_parameters, best_simulation, best_objfun = perform_calibration(
            model_config,
            "mae",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=8,
            algorithm="DDS",
            sampler_kwargs=dict(trials=1),
        )

        # Test that the results have the same size as expected (number of parameters)
        assert len(best_parameters) == len(bounds_high)

    @pytest.mark.skip(
        reason="Weird error with negative simulated PET in ravenpy for HYPR."
    )
    def test_ravenpy_hypr_calibration(self):
        """Test for HYPR ravenpy model"""
        bounds_low = [
            -1.0,
            -3.0,
            0.0,
            0.3,
            -1.3,
            -2.0,
            0.0,
            0.1,
            0.4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.01,
            0.0,
            0.0,
            1.5,
            0.0,
            0.0,
            0.8,
        ]
        bounds_high = [
            1.0,
            3.0,
            0.8,
            1.0,
            0.3,
            0.0,
            30.0,
            0.8,
            2.0,
            100.0,
            0.5,
            5.0,
            1.0,
            1000.0,
            6.0,
            7.0,
            8.0,
            3.0,
            5.0,
            5.0,
            1.2,
        ]

        model_config = deepcopy(self.model_config)
        model_config.update({"model_name": "HYPR"})

        best_parameters, best_simulation, best_objfun = perform_calibration(
            model_config,
            "mae",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=8,
            algorithm="DDS",
            sampler_kwargs=dict(trials=1),
        )

        # Test that the results have the same size as expected (number of parameters)
        assert len(best_parameters) == len(bounds_high)

    def test_ravenpy_sacsma_calibration(self):
        """Test for SAC-SMA ravenpy model"""
        bounds_low = [
            -3.0,
            -1.52287874,
            -0.69897,
            0.025,
            0.01,
            0.075,
            0.015,
            0.04,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.3,
            0.01,
            0.8,
            0.8,
        ]
        bounds_high = [
            -1.82390874,
            -0.69897,
            -0.30102999,
            0.125,
            0.075,
            0.3,
            0.3,
            0.6,
            0.5,
            3.0,
            80.0,
            0.8,
            0.05,
            0.2,
            0.1,
            0.4,
            8.0,
            20.0,
            5.0,
            1.2,
            1.2,
        ]

        model_config = deepcopy(self.model_config)
        model_config.update({"model_name": "SACSMA"})

        best_parameters, best_simulation, best_objfun = perform_calibration(
            model_config,
            "mae",
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            evaluations=8,
            algorithm="DDS",
            sampler_kwargs=dict(trials=1),
        )

        # Test that the results have the same size as expected (number of parameters)
        assert len(best_parameters) == len(bounds_high)

    def test_ravenpy_blended_calibration(self):
        """Test for Blended ravenpy model"""
        bounds_low = [
            0.0,
            0.1,
            0.5,
            -5.0,
            0.0,
            0.5,
            5.0,
            0.0,
            0.0,
            0.0,
            -5.0,
            0.5,
            0.0,
            0.01,
            0.005,
            -5.0,
            0.0,
            0.0,
            0.0,
            0.3,
            0.01,
            0.5,
            0.15,
            1.5,
            0.0,
            -1.0,
            0.01,
            0.00001,
            0.0,
            0.0,
            -3.0,
            0.5,
            0.8,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        bounds_high = [
            1.0,
            3.0,
            3.0,
            -1.0,
            100.0,
            2.0,
            10.0,
            3.0,
            0.05,
            0.45,
            -2.0,
            2.0,
            0.1,
            0.3,
            0.1,
            2.0,
            1.0,
            5.0,
            0.4,
            20.0,
            5.0,
            13.0,
            1.5,
            3.0,
            5.0,
            1.0,
            0.2,
            0.02,
            0.5,
            2.0,
            3.0,
            4.0,
            1.2,
            1.2,
            0.02,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        model_config = deepcopy(self.model_config)
        model_config.update({"model_name": "Blended"})

        if __raven_version__ == "3.8.1":
            warnings.warn("Blended model does not work with RavenHydroFramework v3.8.1")
            with pytest.raises(OSError):
                perform_calibration(
                    model_config,
                    "mae",
                    bounds_low=bounds_low,
                    bounds_high=bounds_high,
                    evaluations=8,
                    algorithm="DDS",
                    sampler_kwargs=dict(trials=1),
                )
        else:
            best_parameters, best_simulation, best_objfun = perform_calibration(
                model_config,
                "mae",
                bounds_low=bounds_low,
                bounds_high=bounds_high,
                evaluations=8,
                algorithm="DDS",
                sampler_kwargs=dict(trials=1),
            )

            # Test that the results have the same size as expected (number of parameters)
            assert len(best_parameters) == len(bounds_high)
