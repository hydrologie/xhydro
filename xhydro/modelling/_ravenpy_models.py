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
        project_dir: Union[str, os.PathLike],
        project_file: str,
        *,
        project_config: Optional[dict] = None,
        simulation_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        use_defaults: bool = True,
        executable: Union[str, os.PathLike] = "hydrotel",
    ):


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
