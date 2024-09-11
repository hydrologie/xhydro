"""Implement the ravenpy handler class for emulating raven models in ravenpy."""

import os
import tempfile
from typing import Optional, Union

import numpy as np
import ravenpy.config.emulators
import xarray as xr
from ravenpy import OutputReader
from ravenpy.config import commands as rc
from ravenpy.ravenpy import run

from ._hm import HydrologicalModel

__all__ = ["RavenpyModel"]


class RavenpyModel(HydrologicalModel):
    r"""Implement the RavenPy model class to build and run ravenpy models.

    Parameters
    ----------
    model_name : {"Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"}
        The name of the ravenpy model to run.
    parameters : np.ndarray
        The model parameters for simulation or calibration.
    drainage_area : float
        The watershed drainage area, in km².
    elevation : float
        The elevation of the watershed, in meters.
    latitude : float
        The latitude of the watershed centroid.
    longitude : float
        The longitude of the watershed centroid.
    start_date : dt.datetime
        The first date of the simulation.
    end_date : dt.datetime
        The last date of the simulation.
    qobs_path : Union[str, os.PathLike]
        The path to the dataset containing the observed streamflow.
    alt_names_flow : dict
        A dictionary that allows users to change the names of flow variables of their dataset to cf-compliant names.
    meteo_file : Union[str, os.PathLike]
        The path to the file containing the observed meteorological data.
    data_type : dict
        The dictionary necessary to tell raven which variables are being fed such that it can adjust it's processes
        internally.
    alt_names_meteo : dict
        A dictionary that allows users to change the names of meteo variables of their dataset to cf-compliant names.
    meteo_station_properties : dict
        The properties of the weather stations providing the meteorological data. Used to adjust weather according to
        differences between station and catchment elevations (adiabatic gradients, etc.).
    workdir : Union[str, os.PathLike]
        Path to save the .rv files and model outputs.
    rain_snow_fraction : str
        The method used by raven to split total precipitation into rain and snow.
    evaporation : str
        The evapotranspiration function used by raven.
    \*\*kwargs : dict
        Dictionary of other parameters to feed to raven according to special cases and that are allowed by the raven
        documentation.
    """

    def __init__(
        self,
        model_name: str,
        parameters: np.ndarray,
        drainage_area: str | os.PathLike,
        elevation: str,
        latitude,
        longitude,
        start_date,
        end_date,
        qobs_path,
        alt_names_flow,
        meteo_file,
        data_type,
        alt_names_meteo,
        meteo_station_properties,
        workdir: str | os.PathLike | None = None,
        rain_snow_fraction="RAINSNOW_DINGMAN",
        evaporation="PET_PRIESTLEY_TAYLOR",
        **kwargs,
    ):
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix=model_name)
        self.workdir = workdir

        self.model_simulations = None
        self.qsim = None

        # Create HRU object for ravenpy based on catchment properties
        hru = dict(
            area=drainage_area,
            elevation=elevation,
            latitude=latitude,
            longitude=longitude,
            hru_type="land",
        )

        # Create the emulator configuration
        self.default_emulator_config = dict(
            HRUs=[hru],
            params=parameters,
            StartDate=start_date,
            EndDate=end_date,
            ObservationData=[
                rc.ObservationData.from_nc(qobs_path, alt_names=alt_names_flow)
            ],
            Gauge=[
                rc.Gauge.from_nc(
                    meteo_file,  # Chemin d'accès au fichier contenant la météo
                    data_type=data_type,  # Liste de toutes les variables contenues dans le fichier
                    alt_names=alt_names_meteo,
                    # Mapping entre les noms des variables requises et celles dans le fichier.
                    data_kwds=meteo_station_properties,
                )
            ],
            RainSnowFraction=rain_snow_fraction,
            Evaporation=evaporation,
            **kwargs,
        )
        self.meteo_file = meteo_file
        self.qobs = xr.open_dataset(qobs_path)
        self.model_name = model_name

    def run(self) -> str | xr.Dataset:
        """Run the ravenpy hydrological model and return simulated streamflow.

        Returns
        -------
        xr.dataset
            The simulated streamflow from the selected ravenpy model.
        """
        default_emulator_config = self.default_emulator_config
        model_name = self.model_name
        workdir = self.workdir

        if model_name not in [
            "Blended",
            "GR4JCN",
            "HBVEC",
            "HMETS",
            "HYPR",
            "Mohyse",
            "SACSMA",
        ]:
            raise ValueError("The selected model is not available in RavenPy.")

        # Need to remove qobs as pydantic forbids extra inputs...
        if "qobs" in default_emulator_config:
            default_emulator_config.pop("qobs")

        if model_name == "HBVEC":
            default_emulator_config.pop("RainSnowFraction")

        self.model = getattr(ravenpy.config.emulators, model_name)(
            **default_emulator_config
        )
        self.model.write_rv(workdir=workdir)

        outputs_path = run(modelname="raven", configdir=workdir, overwrite=True)
        outputs = OutputReader(path=outputs_path)

        qsim = (
            xr.open_dataset(outputs.files["hydrograph"])
            .q_sim.to_dataset(name="qsim")
            .rename({"qsim": "streamflow"})
        )

        if "nbasins" in qsim.dims:
            qsim = qsim.squeeze()

        self.qsim = qsim
        self.model_simulations = outputs

        return qsim

    def get_streamflow(self):
        """Return the precomputed streamflow.

        Returns
        -------
        xr.dataset
            The simulated streamflow from the selected ravenpy model.
        """
        return self.qsim

    def get_inputs(self) -> xr.Dataset:
        """Return the inputs used to run the ravenpy model.

        Returns
        -------
        xr.dataset
            The observed meteorological data used to run the ravenpy model simulation.
        """
        ds = xr.open_dataset(self.meteo_file)

        start_date = self.default_emulator_config["StartDate"]
        end_date = self.default_emulator_config["EndDate"]
        ds = ds.sel(time=slice(start_date, end_date))

        return ds
