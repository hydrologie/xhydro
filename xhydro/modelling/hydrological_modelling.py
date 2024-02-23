"""Hydrological modelling framework.

FIXME: This text should be moved to the documentation.
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
from copy import deepcopy

from ._hydrotel import Hydrotel
from ._simplemodels import DummyModel

__all__ = ["get_hydrological_model_inputs", "hydrological_model"]


def hydrological_model(model_config):
    """Initialize an instance of a hydrological model.

    Parameters
    ----------
    model_config : dict
        A dictionary containing the configuration for the hydrological model.
        Must contain a key "model_name" with the name of the model to use: "Hydrotel".
        The required keys depend on the model being used. Use the function
        `get_hydrological_model_inputs` to get the required keys for a given model.

    Returns
    -------
    Hydrotel or DummyModel
        An instance of the hydrological model.
    """
    if "model_name" not in model_config:
        raise ValueError("The model name must be provided in the model configuration.")

    model_config = deepcopy(model_config)
    model_name = model_config.pop("model_name")

    if model_name == "Hydrotel":
        return Hydrotel(**model_config)
    elif model_name == "Dummy":
        return DummyModel(**model_config)
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")


def get_hydrological_model_inputs(model_name):
    """Get the required inputs for a given hydrological model.

    Parameters
    ----------
    model_name : str
        The name of the hydrological model to use.
        Currently supported models are: "Hydrotel".

    Returns
    -------
    dict
        A dictionary containing the required configuration for the hydrological model.
    """
    if model_name == "Dummy":
        required_config = inspect.getfullargspec(DummyModel.__init__).annotations
    elif model_name == "Hydrotel":
        required_config = inspect.getfullargspec(Hydrotel.__init__).annotations
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")

    return required_config
