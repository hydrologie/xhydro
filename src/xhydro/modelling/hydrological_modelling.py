"""Hydrological modelling framework."""

import inspect
from copy import deepcopy

from ._hydrotel import Hydrotel
from ._ravenpy_models import RavenpyModel
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
    model_name = model_config["model_name"]

    if model_name == "Hydrotel":
        model_config.pop("model_name")
        return Hydrotel(**model_config)

    elif model_name == "Dummy":
        model_config.pop("model_name")
        return DummyModel(**model_config)

    elif model_name in [
        "Blended",
        "GR4JCN",
        "HBVEC",
        "HMETS",
        "HYPR",
        "Mohyse",
        "SACSMA",
    ]:
        return RavenpyModel(**model_config)
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")


def get_hydrological_model_inputs(
    model_name, required_only: bool = False
) -> tuple[dict, str]:
    """Get the required inputs for a given hydrological model.

    Parameters
    ----------
    model_name : str
        The name of the hydrological model to use.
        Currently supported models are: "Hydrotel".
    required_only : bool
        If True, only the required inputs will be returned.

    Returns
    -------
    dict
        A dictionary containing the required configuration for the hydrological model.
    str
        The documentation for the hydrological model.
    """
    if model_name == "Dummy":
        model = DummyModel
    elif model_name == "Hydrotel":
        model = Hydrotel
    elif model_name in [
        "Blended",
        "GR4JCN",
        "HBVEC",
        "HMETS",
        "HYPR",
        "Mohyse",
        "SACSMA",
    ]:
        model = RavenpyModel
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")

    all_config = inspect.getfullargspec(model.__init__).annotations
    if required_only:
        all_config = {
            k: v
            for k, v in all_config.items()
            if k in inspect.getfullargspec(model.__init__).args
        }

    # Add the model name to the configuration
    all_config = {"model_name": model_name, **all_config}

    return all_config, model.__doc__
