import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ravenpy import OutputReader
from ravenpy.config import commands as rc
from ravenpy.config.emulators import GR4JCN, HBVEC, HMETS, HYPR, SACSMA, Blended, Mohyse
from ravenpy.ravenpy import run

import os
import re
import shutil
import subprocess
import warnings
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import xclim as xc
from xscen.io import estimate_chunks, save_to_netcdf

from xhydro.utils import health_checks

from ._hm import HydrologicalModel

__all__ = ["RavenpyModel"]


class RavenpyModel(HydrologicalModel):

    def __init__(
        self,
        model_name: str,
        parameters: np.ndarray,
        drainage_area: Union[str, os.PathLike],
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
        rain_snow_fraction="RAINSNOW_DINGMAN",
        evaporation="PET_PRIESTLEY_TAYLOR",
        **kwargs,
    ):

        # Create HRU object for ravenpy based on catchment properties
        self.model_simulations = None
        self.qsim = None
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
            ObservationData=[rc.ObservationData.from_nc(qobs_path, alt_names=alt_names_flow)],
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
            **kwargs
        )
        self.meteo_file = meteo_file
        self.qobs = xr.open_dataset(qobs_path)
        self.model_name = model_name.lower()

    def run(self) -> Union[str, xr.Dataset]:

        default_emulator_config = self.default_emulator_config
        model_name = self.model_name

        if model_name == "gr4jcn":
            self.model = GR4JCN(**default_emulator_config)
        elif model_name == "hmets":
            self.model = HMETS(**default_emulator_config)
        elif model_name == "mohyse":
            self.model = Mohyse(**default_emulator_config)
        elif model_name == "hbvec":
            default_emulator_config.pop("RainSnowFraction")
            self.model = HBVEC(**default_emulator_config)
        elif model_name == "hypr":
            self.model = HYPR(**default_emulator_config)
        elif model_name == "sacsma":
            self.model = SACSMA(**default_emulator_config)
        elif model_name == "blended":
            self.model = Blended(**default_emulator_config)
        else:
            raise ValueError("Hydrological model is an unknown Ravenpy variant.")

        workdir = Path(tempfile.mkdtemp(prefix="NB4"))
        self.model.write_rv(workdir=workdir)

        outputs_path = run(modelname="raven", configdir=workdir)
        outputs = OutputReader(path=outputs_path)

        qsim = xr.open_dataset(outputs.files["hydrograph"]).q_sim.to_dataset(name="qsim")

        if "nbasins" in qsim.dims:
            qsim = qsim.squeeze()

        self.qsim = qsim
        self.model_simulations = outputs

        return qsim

    def get_streamflow(self):

        return self.qsim

    def get_inputs(self) -> xr.Dataset:
        ds = xr.open_dataset(self.meteo_file)

        start_date = self.default_emulator_config["StartDate"]
        end_date = self.default_emulator_config["EndDate"]
        ds = ds.sel(time=slice(start_date, end_date))

        return ds
