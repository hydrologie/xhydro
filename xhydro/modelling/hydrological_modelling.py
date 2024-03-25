"""Hydrological modelling framework."""

import inspect
from copy import deepcopy

from ._hydrotel import Hydrotel
from ._simplemodels import DummyModel
from ._ravenpy_models import RavenpyModel

import tempfile
from pathlib import Path


from ravenpy.config import commands as rc
from ravenpy.config.emulators import GR4JCN, HBVEC, HMETS, HYPR, SACSMA, Blended, Mohyse


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
    elif model_name.lower() in [
        "gr4jcn",
        "hmets",
        "mohyse",
        "hbvec",
        "hypr",
        "sacsma",
        "blended",
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
    elif model_name.lower() in ["blended", "gr4jcn", "hbvec", "hmets", "hypr", "mohyse", "sacsma"]:
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


def _ravenpy_model(model_config: dict):

    # Create HRU object for ravenpy based on catchment properties
    hru = dict(
        area=model_config["drainage_area"],
        elevation=model_config["elevation"],
        latitude=model_config["latitude"],
        longitude=model_config["longitude"],
        hru_type="land",
    )

    # Create the emulator configuration
    default_emulator_config = dict(
        HRUs=[hru],
        StartDate=model_config["start_date"],
        EndDate=model_config["end_date"],
        ObservationData=[
            rc.ObservationData.from_nc(
                model_config["qobs_path"], alt_names=model_config["alt_names_flow"]
            )
        ],
        Gauge=[
            rc.Gauge.from_nc(
                model_config[
                    "meteo_file"
                ],  # Chemin d'accès au fichier contenant la météo
                data_type=model_config[
                    "data_type"
                ],  # Liste de toutes les variables contenues dans le fichier
                alt_names=model_config[
                    "alt_names_meteo"
                ],  # Mapping entre les noms des variables requises et celles dans le fichier.
                data_kwds=model_config["meteo_station_properties"],
            )
        ],
        RainSnowFraction="RAINSNOW_DINGMAN",
        Evaporation="PET_PRIESTLEY_TAYLOR",
    )

    model_name = model_config["model_name"].lower()

    if model_name == "gr4jcn":
        m = GR4JCN(params=model_config["parameters"], **default_emulator_config)
    elif model_name == "hmets":
        m = HMETS(params=model_config["parameters"], **default_emulator_config)
    elif model_name == "mohyse":
        m = Mohyse(params=model_config["parameters"], **default_emulator_config)
    elif model_name == "hbvec":
        default_emulator_config.pop("RainSnowFraction")
        m = HBVEC(params=model_config["parameters"], **default_emulator_config)
    elif model_name == "hypr":
        m = HYPR(params=model_config["parameters"], **default_emulator_config)
    elif model_name == "sacsma":
        m = SACSMA(params=model_config["parameters"], **default_emulator_config)
    elif model_name == "blended":
        m = Blended(params=model_config["parameters"], **default_emulator_config)
    else:
        raise ValueError("Hydrological model is an unknown Ravenpy variant.")

    return m
