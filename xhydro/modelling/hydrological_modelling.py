"""Hydrological modelling framework.

This collection of functions should serve as the main entry point for
hydrological modelling. The entire framework is based on the "model_config"
object. This object is meant to be a container that can be used as needed by
any hydrologic model. For example, it can store datasets directly, paths to
datasets (nc files or other), csv files, basically anything that can be stored
in a dictionary.

It then becomes the user's responsibility to ensure that required data for a
given model be provided in the model_config object both in the data preparation
stage and in the hydrological model implementation. This can be addressed by
a set of pre-defined codes for given model structures. This present package
(hydrological_modelling.py) should contain the logic to:

    1. From the model_config["model_name"] key, select the appropriate function
       (hydrological model) to run.
    2. Pass the model_config object to the correct hydrological modelling
       function.
    3. Parse the model_config object to extract required data for the given
       model, such as: parameters, meteorological data, paths to input files, and catchment characteristics as required
    4. Run the hydrological model with the given data
    5. Return the streamflow (Qsim).

This will make it very easy to add hydrological models, no matter their
complexity. It will also make it easy for newer Python users to implement
models as the logic is clearly exposed and does not make use of complex classes
or other dynamic objects. Models can be added here, and a specific routine
should also be defined to produce the required model_config for each model.

Once this is accomplished, running the model from a model_config object becomes
trivial and allows for easy calibration, regionalisation, analysis and any
other type of interaction.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ._hydrotel import Hydrotel

__all__ = ["HydrologicalModel", "get_hydrological_model_inputs"]


class HydrologicalModel:
    """Hydrological model class.

    This class is a wrapper for the different hydrological models that can be used in xhydro.

    Parameters
    ----------
    model_name : str
        The model name. One of ["Hydrotel", "Dummy"].
    model_config : dict
        The model configuration object that contains all info required to run the model.
        Use the get_hydrological_model_inputs() function to get the required list of keys.

    Notes
    -----
    Available functions, regardless of the model:
    - run: Run the hydrological model.
    - get_input: Return the input data for the model.
    - get_streamflow: Return the simulated streamflow from the model.
    """

    def __init__(self, model_name: str, model_config: dict):
        """Initialize the HydrologicalModel object."""
        self.model_name = model_name

        if model_name == "Hydrotel":
            self.model = Hydrotel(**model_config)
        elif model_name == "Dummy":
            self.model = DummyModel(**model_config)
        else:
            raise NotImplementedError(f"The model '{model_name}' is not recognized.")

    def run(self, **kwargs) -> xr.Dataset:
        r"""Run the hydrological model.

        Parameters
        ----------
        \*\*kwargs : dict
            Arbitrary keyword arguments to pass to the model.

        Returns
        -------
        xr.Dataset
            Simulated streamflow from the model, in xarray Dataset format.
        """
        return self.model.run(**kwargs)

    def get_input(self, **kwargs) -> xr.Dataset:
        r"""Return the input data for the model.

        Parameters
        ----------
        \*\*kwargs : dict
            Arbitrary keyword arguments to pass to the model.

        Returns
        -------
        xr.Dataset
            Input data for the model, in xarray Dataset format.
        """
        return self.model.get_input(**kwargs)

    def get_streamflow(self, **kwargs) -> xr.Dataset:
        r"""Return the simulated streamflow from the model.

        Parameters
        ----------
        \*\*kwargs : dict
            Arbitrary keyword arguments to pass to the model.

        Returns
        -------
        xr.Dataset
            Simulated streamflow from the model, in xarray Dataset format.
        """
        return self.model.get_streamflow(**kwargs)

    def __repr__(self):
        """Return a string representation of the object."""
        return f"xhydro.modelling.HydrologicalModel(model_name={self.model_name})"


def get_hydrological_model_inputs(model_name: str):
    """Required hydrological model inputs for model_config objects.

    Parameters
    ----------
    model_name : str
        Model name that must be one of the models in the list of possible
        models.

    Returns
    -------
    dict
        Elements that must be found in the model_config object.
    """
    if model_name == "Dummy":
        required_config = inspect.getfullargspec(DummyModel.__init__).annotations
    elif model_name == "Hydrotel":
        # FIXME: This is a temporary solution that does not give a description of the required inputs, only the type.
        required_config = inspect.getfullargspec(Hydrotel.__init__).annotations
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")

    return required_config


class DummyModel:
    """Dummy model.

    Dummy model to use as a placeholder for testing purposes.

    Parameters
    ----------
    precip : xr.DataArray
        Daily precipitation in mm.
    temperature : xr.DataArray
        Daily average air temperature in °C.
    drainage_area : float
        Drainage area of the catchment.
    parameters : np.ndarray
        Model parameters, length 3.
    """

    def __init__(
        self,
        precip: xr.DataArray,
        temperature: xr.DataArray,
        drainage_area: float,
        parameters: np.ndarray,
    ):
        """Initialize the DummyModel object."""
        warnings.warn(
            "This is a dummy model. Do not use for real applications.", UserWarning
        )
        self.precip = precip
        self.temperature = temperature
        self.drainage_area = drainage_area
        self.parameters = parameters
        self.qsim = None

    def run(self):
        """Run the Dummy model.

        Returns
        -------
        xr.Dataset
            Simulated streamflow from the Dummy model, in xarray Dataset format.
        """
        qsim = np.empty(len(self.precip))
        for t in range(0, len(self.precip)):
            qsim[t] = (
                (
                    self.precip[t] * self.parameters[0]
                    + abs(self.temperature[t]) * self.parameters[1]
                )
                * self.parameters[2]
                * self.drainage_area
            )

        time = pd.date_range("2024-01-01", periods=len(self.precip))
        qsim = xr.Dataset(
            data_vars=dict(
                qsim=(["time"], qsim),
            ),
            coords=dict(time=time),
            attrs=dict(description="streamflow simulated by the Dummy Model"),
        )
        self.qsim = qsim

        return qsim

    def get_input(self):
        """Return the input data for the Dummy model.

        Returns
        -------
        xr.Dataset
            Input data for the Dummy model, in xarray Dataset format.
        """
        return xr.Dataset(
            data_vars=dict(
                precip=(["time"], self.precip),
                temperature=(["time"], self.temperature),
            ),
            coords=dict(time=pd.date_range("2024-01-01", periods=len(self.precip))),
            attrs=dict(description="input data for the Dummy Model"),
        )

    def get_streamflow(self):
        """Return the simulated streamflow from the Dummy model.

        Returns
        -------
        xr.Dataset
            Simulated streamflow from the Dummy model, in xarray Dataset format.
        """
        if self.qsim is None:
            self.run()
        else:
            return self.qsim
