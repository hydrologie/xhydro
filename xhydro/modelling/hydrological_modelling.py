"""Hydrological modelling framework."""

import inspect
from copy import deepcopy

from ._hydrotel import Hydrotel
from ._simplemodels import DummyModel
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ravenpy import OutputReader
from ravenpy.config import commands as rc
from ravenpy.config.emulators import GR4JCN, HBVEC, HMETS, HYPR, SACSMA, Blended, Mohyse
from ravenpy.ravenpy import run

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

        required_config = dict(
            precip="Daily precipitation in mm.",
            temperature="Daily average air temperature in °C",
            drainage_area="Drainage area of the catchment, km²",
            parameters="Model parameters, length 3",
        )
    elif model_name.lower() in [
        "gr4jcn",
        "hmets",
        "mohyse",
        "hbvec",
        "hypr",
        "sacsma",
        "blended",
    ]:
        # TODO ADD THIS
        required_config = dict(
            temperature="Daily average air temperature in °C.",
            precip="Daily precipitation in mm.",
            drainage_area="Drainage area of the catchment, km²",
        )
    elif model_name == "ADD_OTHER_HERE":
        # ADD OTHER MODELS HERE
        required_config = {}
        model = DummyModel

    elif model_name == "Hydrotel":
        model = Hydrotel

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

def _dummy_model(model_config: dict):
    """Dummy model.

    Dummy model to show the implementation we should be aiming for. Each model
    will have its own required data that users can pass.

    Parameters
    ----------
    model_config : dict
        The model configuration object that contains all info to run the model.
        The model function called to run this model shouls always use this object
        and read-in data it requires. It will be up to the user to provide the
        data that the model requires.

    Returns
    -------
    xr.Dataset
        Simulated streamflow from the model, in xarray Dataset format.
    """
    # Parse the model_config object to extract required information
    precip = model_config["precip"]
    temperature = model_config["temperature"]
    area = model_config["drainage_area"]
    x = model_config["parameters"]

    # Run the dummy model using these data. Keeping it simple to calculate by
    # hand to ensure the calibration algorithm is working correctly and data
    # are handled correctly
    qsim = np.empty(len(precip))
    for t in range(0, len(precip)):
        qsim[t] = (precip[t] * x[0] + abs(temperature[t]) * x[1]) * x[2] * area

    # For this model, we can convert to xr.dataset by supposing dates. Other
    # models will require some dates in some form (to add to QC checks) in their
    # inputs.
    time = pd.date_range("2024-01-01", periods=len(precip))
    qsim = xr.Dataset(
        data_vars=dict(
            qsim=(["time"], qsim),
        ),
        coords=dict(time=time),
        attrs=dict(description="streamflow simulated by the Dummy Model"),
    )

    return qsim


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

    workdir = Path(tempfile.mkdtemp(prefix="NB4"))
    m.write_rv(workdir=workdir)

    outputs_path = run(modelname="raven", configdir=workdir)
    outputs = OutputReader(path=outputs_path)

    qsim = xr.open_dataset(outputs.files["hydrograph"]).q_sim.to_dataset(name="qsim")

    if "nbasins" in qsim.dims:
        qsim = qsim.squeeze()

    return qsim
