"""Implement the ravenpy handler class for emulating raven models in ravenpy."""

import datetime as dt
import logging
import os
import shutil
import subprocess  # noqa: S404
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from xclim.core.units import convert_units_to
from xscen.io import estimate_chunks, save_to_netcdf


try:
    import ravenpy
    import ravenpy.config as rc
    import ravenpy.extractors

    run = ravenpy.ravenpy.run

    ravenpy_err_msg = None
except (ImportError, RuntimeError) as e:
    run = None
    ravenpy_err_msg = e

from ._hm import HydrologicalModel


logger = logging.getLogger(__name__)

__all__ = ["RavenpyModel"]


class RavenpyModel(HydrologicalModel):
    r"""
    Initialize the RavenPy model class.

    Parameters
    ----------
    overwrite : bool
        If True, overwrite the existing project files. Default is False.
    workdir : str | Path | None
        Path to save the .rv files and model outputs. Default is None, which creates a temporary directory.
    executable : str | os.PathLike | None, optional
        Path to the Raven executable, bypassing RavenPy.
        If None (default), the Raven executable from your current Python environment ('raven-hydro') will be used.
    run_name : str, optional
        Name of the run, which will be used to name the project files. Defaults to "raven" if not provided.
    model_name : {"Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"}, optional
        The name of the RavenPy model to run. Only optional if the project files already exist.
    start_date : dt.datetime | str, optional
        The first date of the simulation. Only optional if the project files already exist.
    end_date : dt.datetime | str, optional
        The last date of the simulation. Only optional if the project files already exist.
    parameters : np.ndarray | list[float], optional
        The model parameters for simulation or calibration. Only optional if the project files already exist.
    qobs_file : str | Path, optional
        Path to the file containing the observed streamflow data.
        If there are multiple stations, the file should contain a 'basin_id' variable that identifies the subbasin for each time series.
        If a 'station_id' variable is present, it will be used to identify the station.
    alt_name_flow : str, optional
        Name of the streamflow variable in the observed data file. If not provided, it will be assumed to be "q".
    hru : gpd.GeoDataFrame | dict | os.PathLike, optional
        A GeoDataFrame, or dictionary containing the HRU properties. Only optional if the project files already exist.
        For distributed models, it should be readable by ravenpy.extractors.BasinMakerExtractor.
        For lumped models, should contain the following variables:
        - area: The watershed drainage area, in km².
        - elevation: The elevation of the watershed, in meters.
        - latitude: The latitude of the watershed centroid.
        - longitude: The longitude of the watershed centroid.
        - HRU_ID: The ID of the HRU (required for gridded data, optional for station data).
        If the meteorological data is gridded, the HRU dataset must also contain a SubId, DowSubId, valid geometry and crs.
        If the input is modified, a new shapefile will be created in the workdir/weights subdirectory.
    output_subbasins : {"all", "qobs"} | list[int] | None, optional
        If "all", all subbasins will be outputted. If "qobs", only the subbasins with observed flow will be outputted.
        Leave as None to use the value as defined in the HRU file ('Has_Gauge' column). Only applicable for distributed HBVEC models.
    minimum_reservoir_area : str, optional
        Quantified string (e.g. "20 km2") representing the minimum lake area to consider the lake explicitly as a reservoir.
        If not provided, all lakes with the 'HRU_IsLake' column set to 1 in the HRU file will be considered as reservoirs.
        Note that 'reservoirs' in Raven can also refer to natural lakes with weir-like outflows.
        Only applicable for distributed HBVEC models.
    meteo_file : str | Path, optional
        Path to the file containing the observed meteorological data. Only optional if the project files already exist.
        The meteorological data can be either station or gridded data. Use the 'xhydro.modelling.format_input' function to ensure the data
        is in the correct format. Unless the input is a single station accompanied by 'meteo_station_properties', the file should contain
        the following coordinates:
        - elevation: The elevation of the station / grid cell, in meters.
        - latitude: The latitude of the station / grid cell centroid.
        - longitude: The longitude of the station / grid cell centroid.
    data_type : list[str], optional
        The list of types of data provided to Raven in the meteorological file. Only optional if the project files already exist.
        See https://github.com/CSHS-CWRA/RavenPy/blob/master/src/ravenpy/config/conventions.py for the list of available types.
    alt_names_meteo : dict, optional
        A dictionary that allows users to link the names of meteorological variables in their dataset to Raven-compliant names.
        The keys should be the Raven names as listed in the data_type parameter.
    meteo_station_properties : dict, optional
        Additional properties of the weather stations providing the meteorological data. Only required if absent from the 'meteo_file'.
        For single stations, the format is {"ALL": {"elevation": elevation, "latitude": latitude, "longitude": longitude}}.
        This has not been tested for multiple stations or gridded data.
    gridweights : str | Path | None
        If using gridded meteorological data, path to a text file containing the weights linking the grid cells to the HRUs.
        If None, the weights will be computed using ravenpy.extractors.GridWeightExtractor and saved in a 'weights' subdirectory
        of the project folder, using a "{meteo_file}_vs_{hru_file}_weights.txt" pattern.
    \*\*kwargs : dict, optional
        Additional parameters to pass to the RavenPy emulator, to modify the default modules used by a given hydrological model.
        Typical entries include RainSnowFraction, Evaporation, GlobalParameters, etc.
        See https://raven.uwaterloo.ca/Downloads.html for the latest Raven documentation. Currently, model templates are listed in Appendix F.
    """

    def __init__(
        self,
        overwrite: bool = False,
        *,
        workdir: str | os.PathLike | None = None,
        executable: str | os.PathLike | None = None,
        run_name: str | None = None,
        model_name: (Literal["Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"] | None) = None,
        start_date: dt.datetime | str | None = None,
        end_date: dt.datetime | str | None = None,
        parameters: np.ndarray | list[float] | None = None,
        qobs_file: os.PathLike | str | None = None,
        alt_name_flow: str | None = "q",
        hru: gpd.GeoDataFrame | dict | os.PathLike | str | None = None,
        output_subbasins: Literal["all", "qobs"] | list[int] | None = None,
        minimum_reservoir_area: str | None = None,
        meteo_file: os.PathLike | str | None = None,
        data_type: list[str] | None = None,
        alt_names_meteo: dict | None = None,
        meteo_station_properties: dict | None = None,
        gridweights: str | os.PathLike | None = None,
        **kwargs,
    ):
        """Initialize the RavenPy model class."""
        # Set a few variables
        if workdir is None:
            self.workdir = Path(tempfile.mkdtemp(prefix="raven-hydro_"))
        else:
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
        self.executable = executable

        self.model_name = model_name
        self.run_name = run_name or "raven"
        self.parameters = parameters
        self.start_date = start_date
        self.end_date = end_date
        self.kwargs = deepcopy(kwargs)

        # Updated when the project files are created and never really changed afterwards,
        # but useful to have it here to allow the user to access it.
        self.emulator_config = None

        self.qobs = None
        self.hru = None
        self.meteo = None
        if any(opt is not None for opt in [qobs_file, hru, meteo_file]):
            self.update_data(
                qobs_file=qobs_file,
                alt_name_flow=alt_name_flow,
                hru=hru,
                output_subbasins=output_subbasins,
                minimum_reservoir_area=minimum_reservoir_area,
                meteo_file=meteo_file,
                data_type=data_type,
                alt_names_meteo=alt_names_meteo,
                meteo_station_properties=meteo_station_properties,
                gridweights=gridweights,
            )

        # Check for existing project files
        files = [f for f in self.workdir.rglob(f"{self.run_name}.rv*")]

        # If the project files already exist and overwrite is False, we stop here
        if len(files) > 0 and not overwrite:
            if any(opt is not None for opt in [self.meteo, self.hru, self.qobs]):
                warnings.warn(
                    "Meteorological data, HRU, or observed streamflow data were provided, but the project files already exist "
                    "and 'overwrite' is set to False. These data will not be used. If you want to update the "
                    "project files, use the 'update_config' method.",
                    stacklevel=2,
                )
            else:
                logger.info("Project already exists and files will not be overwritten.")
            return

        # If the project files do not exist, we create them if the required data is provided
        else:
            if self.meteo is None or self.hru is None:
                warnings.warn(
                    "The meteorological data and/or HRU are not provided. Project files will not be created. Please provide the missing data, "
                    "then call the 'create_rv' method to create the project files.",
                    stacklevel=2,
                )
                return
            if len(files) > 0:
                logger.info("Project already exists, but will be overwritten.")

            # Create the project files
            self.create_rv(
                overwrite=overwrite,
            )

    def create_rv(
        self,
        *,
        overwrite: bool = False,
        parameters=None,
        model_name=None,
        hru=None,
        meteo_file=None,
        data_type=None,
        start_date=None,
        end_date=None,
        alt_names_meteo=None,
        meteo_station_properties=None,
        qobs_file=None,
        alt_name_flow="q",
        minimum_reservoir_area=None,
        output_subbasins=None,
        **kwargs,
    ):
        r"""
        Write the RavenPy project files.

        Parameters
        ----------
        overwrite : bool
            If True, overwrite the existing project files. Default is False.
            Note that to prevent inconsistencies, all files containing the 'run_name' will be removed, including the output files.
        parameters : None
            Deprecated. Use the class attribute instead.
        model_name : None
            Deprecated. Use the class attribute instead.
        hru : None
            Deprecated. Use the 'update_data' method instead.
        meteo_file : None
            Deprecated. Use the 'update_data' method instead.
        data_type : None
            Deprecated. Use the 'update_data' method instead.
        start_date : None
            Deprecated. Use the class attribute instead.
        end_date : None
            Deprecated. Use the class attribute instead.
        alt_names_meteo : None
            Deprecated. Use the 'update_data' method instead.
        meteo_station_properties : None
            Deprecated. Use the 'update_data' method instead.
        qobs_file : None
            Deprecated. Use the 'update_data' method instead.
        alt_name_flow : str
            Deprecated. Use the 'update_data' method instead.
        minimum_reservoir_area : None
            Deprecated. Use the 'update_data' method instead.
        output_subbasins : None
            Deprecated. Use the 'update_data' method instead.
        \*\*kwargs : dict
            Deprecated. Instantiate the model with the them or pass them to the class attribute.
        """
        if run is None:
            raise RuntimeError(
                "RavenPy is not installed or not properly configured. The RavenpyModel.create_rv method cannot be used without it."
                f" Original error: {ravenpy_err_msg}"
            )

        if parameters is not None:
            warnings.warn(
                "The 'parameters' parameter is deprecated and will be removed in a future version. "
                "Please set the 'parameters' attribute directly on the RavenPy model instance.",
                FutureWarning,
                stacklevel=2,
            )
            self.parameters = parameters
        if model_name is not None:
            warnings.warn(
                "The 'model_name' parameter is deprecated and will be removed in a future version. "
                "Please set the 'model_name' attribute directly on the RavenPy model instance.",
                FutureWarning,
                stacklevel=2,
            )
            self.model_name = model_name
        if start_date is not None:
            warnings.warn(
                "The 'start_date' parameter is deprecated and will be removed in a future version. "
                "Please set the 'start_date' attribute directly on the RavenPy model instance.",
                FutureWarning,
                stacklevel=2,
            )
            self.start_date = start_date
        if end_date is not None:
            warnings.warn(
                "The 'end_date' parameter is deprecated and will be removed in a future version. "
                "Please set the 'end_date' attribute directly on the RavenPy model instance.",
                FutureWarning,
                stacklevel=2,
            )
            self.end_date = end_date
        if len(kwargs) > 0:
            warnings.warn(
                "Kwargs in the 'create_rv' method are deprecated and will be removed in a future version. "
                "Set them when first instantiating the RavenPy model, or later by setting the attributes in "
                "a dictionary under the 'kwargs' attribute of the class.",
                FutureWarning,
                stacklevel=2,
            )
            self.kwargs = kwargs
        if any(
            opt is not None
            for opt in [
                qobs_file,
                alt_name_flow,
                hru,
                output_subbasins,
                minimum_reservoir_area,
                meteo_file,
                data_type,
                alt_names_meteo,
                meteo_station_properties,
            ]
        ):
            warnings.warn(
                "Data-related parameters are deprecated and will be removed in a future version. Please use the 'update_data' method instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.update_data(
                qobs_file=qobs_file,
                alt_name_flow=alt_name_flow,
                hru=hru,
                output_subbasins=output_subbasins,
                minimum_reservoir_area=minimum_reservoir_area,
                meteo_file=meteo_file,
                data_type=data_type,
                alt_names_meteo=alt_names_meteo,
                meteo_station_properties=meteo_station_properties,
            )

        required = [
            "model_name",
            "run_name",
            "start_date",
            "end_date",
            "parameters",
            "meteo",
            "hru",
        ]
        missing = [input for input in required if getattr(self, input) is None]
        if missing:
            raise ValueError(f"The following required inputs are missing: {', '.join(missing)}")

        # Remove any existing files in the project directory
        if len([f for f in self.workdir.rglob(f"{self.run_name}.rv*")]) > 0:
            if overwrite:
                for file in self.workdir.rglob(f"{self.run_name}*.*"):
                    file.unlink()
            else:
                raise FileExistsError(f"Project {self.run_name} in {self.workdir} already exists, but 'overwrite' is set to False.")

        kwargs = deepcopy(self.kwargs)

        # Add the observed streamflow data if provided
        if self.qobs is not None:
            kwargs["ObservationData"] = self.qobs["rc"]

        # Add the HRU properties
        kwargs = kwargs | self.hru["keys"]

        # Add the meteorological data
        kwargs = kwargs | self.meteo["keys"]

        # Create the emulator configuration
        self.emulator_config = dict(
            RunName=self.run_name,
            params=self.parameters,
            StartDate=self.start_date,
            EndDate=self.end_date,
            **kwargs,
        )
        model = getattr(rc.emulators, self.model_name)(
            **self.emulator_config,
        )
        model.write_rv(workdir=self.workdir, overwrite=overwrite)

    def update_data(
        self,
        *,
        qobs_file: os.PathLike | str | None = None,
        alt_name_flow: str | None = "q",
        hru: gpd.GeoDataFrame | dict | os.PathLike | str | None = None,
        output_subbasins: Literal["all", "qobs"] | list[int] | None = None,
        minimum_reservoir_area: str | None = None,
        meteo_file: os.PathLike | str | None = None,
        data_type: list[str] | None = None,
        alt_names_meteo: dict | None = None,
        meteo_station_properties: dict | None = None,
        gridweights: str | os.PathLike | None = None,
    ):
        """
        Update the model configuration with new observed data (self.qobs), HRU properties (self.hru), or meteorological data (self.meteo).

        Parameters
        ----------
        qobs_file : os.PathLike | str
            Path to the NetCDF file containing the observed streamflow data.
            If there are multiple stations, the file should contain a 'basin_id' variable that identifies the subbasin for each time series.
            If a 'station_id' variable is present, it will be used to identify the station.
        alt_name_flow : str, optional
            Alternative name for the streamflow variable in the observed data.
        hru : gpd.GeoDataFrame | dict | os.PathLike | str
            A GeoDataFrame, or dictionary containing the HRU properties. Alternatively, a path to a shapefile containing the HRU properties.
            For distributed models, it should be readable by ravenpy.extractors.BasinMakerExtractor.
            For lumped models, should contain the following variables:
            - area: The watershed drainage area, in km².
            - elevation: The elevation of the watershed, in meters.
            - latitude: The latitude of the watershed centroid.
            - longitude: The longitude of the watershed centroid.
            - HRU_ID: The ID of the HRU (required for gridded data, optional for station data).
            If the meteorological data is gridded, the HRU dataset must also contain a SubId, DowSubId, valid geometry and crs.
            If the input is modified, a new shapefile will be created in the workdir/weights subdirectory.
        output_subbasins : {"all", "qobs"} | list[int] | None, optional
            If "all", all subbasins will be outputted.
            If "qobs", subbasins with observed flow will be outputted, as defined by the basin IDs in the observed streamflow data.
            If a list of integers is provided, it should contain the basin IDs to output.
            Leave as None to use the value as defined in the HRU file ('Has_Gauge' column).
        minimum_reservoir_area : str, optional
            Quantified string (e.g. "20 km2") representing the minimum lake area to consider the lake explicitly as a reservoir.
            If not provided, all lakes with the 'HRU_IsLake' column set to 1 in the HRU file will be considered as reservoirs.
            Note that 'reservoirs' in Raven can also refer to natural lakes with weir-like outflows.
            Only applicable for distributed HBVEC models.
        meteo_file : str | Path, optional
            Path to the file containing the observed meteorological data. Only optional if the project files already exist.
            The meteorological data can be either station or gridded data. Use the 'xhydro.modelling.format_input' function to ensure the data
            is in the correct format. Unless the input is a single station accompanied by 'meteo_station_properties', the file should contain
            the following coordinates:
            - elevation: The elevation of the station / grid cell, in meters.
            - latitude: The latitude of the station / grid cell centroid.
            - longitude: The longitude of the station / grid cell centroid.
        data_type : list[str], optional
            The list of types of data provided to Raven in the meteorological file. Only optional if the project files already exist.
            See https://github.com/CSHS-CWRA/RavenPy/blob/master/src/ravenpy/config/conventions.py for the list of available types.
        alt_names_meteo : dict, optional
            A dictionary that allows users to link the names of meteorological variables in their dataset to Raven-compliant names.
            The keys should be the Raven names as listed in the data_type parameter.
        meteo_station_properties : dict, optional
            Additional properties of the weather stations providing the meteorological data. Only required if absent from the 'meteo_file'.
            For single stations, the format is {"ALL": {"elevation": elevation, "latitude": latitude, "longitude": longitude}}.
            This has not been tested for multiple stations or gridded data.
        gridweights : str | Path | None
            If using gridded meteorological data, path to a text file containing the weights linking the grid cells to the HRUs.
            If None, the weights will be computed using ravenpy.extractors.GridWeightExtractor and saved in a 'weights' subdirectory
            of the project folder, using a "{meteo_file}_vs_{hru_file}_weights.txt" pattern.

        Notes
        -----
        If the meteorological data is gridded, new weights will be computed using the HRU file in the RavenpyModel instance and saved
        in a 'weights' subdirectory of the project folder, under the name 'meteo-name_vs_hru-name.txt'.
        """
        if (any(opt is not None for opt in [output_subbasins, minimum_reservoir_area]) and hru is None) or (
            any(opt is not None for opt in [data_type, alt_names_meteo, meteo_station_properties]) and meteo_file is None
        ):
            raise ValueError("All relevant inputs must be provided if updating the HRU properties or meteorological data")

        # Q_obs must be read first, as it might be used to update the subbasins to output
        if qobs_file is not None:
            self._read_qobs(qobs_file, alt_name_flow=alt_name_flow)

        # HRU data must be read before the meteorological data, as it might be used to update the meteorological data weights
        if hru is not None:
            self._read_hru(
                hru,
                output_subbasins=output_subbasins,
                minimum_reservoir_area=minimum_reservoir_area,
            )

        # Meteorological data must be last
        if meteo_file is not None:
            self._read_meteo(
                meteo_file=meteo_file,
                data_type=data_type,
                alt_names_meteo=alt_names_meteo,
                meteo_station_properties=meteo_station_properties,
                gridweights=gridweights,
            )

    def update_config(  # noqa: C901
        self,
        *,
        rvi_dates: bool = False,
        rvi_commands: list[str] | None = None,
        rvt: bool = False,
        rvh: bool = False,
    ) -> None:
        """
        Manually update some aspects of the configuration of the RavenPy model.

        Parameters
        ----------
        rvi_dates : bool
            If True, update the .rvi file with the 'start_date' and 'end_date' defined in the model.
        rvi_commands : list[str] | None
            A list of commands to include in the .rvi file. If None, no additional commands will be added.
            Warning: These commands will be added at the end of the .rvi file, with no checks. Use with caution.
        rvt : bool
            If True, update the .rvt file with the meteorological data and observed streamflow data defined in the model.
        rvh : bool
            If True, update the .rvh file with the list of subbasins to output. Nothing else will be changed in that file.

        Notes
        -----
        Ideally, users should favor using the `update_data` method to update the model configuration, then call the `create_rv`
        method to recreate the project files from scratch. This method assumes that the changes brought to the model configuration
        are minimal, such as wanting to change the meteorological data or the simulation start and end dates.

        Be aware that:
          - The .rvh will be rewritten entirely. If multiple sources of data were mentioned, such as both meteorological and observed streamflow data,
            all of them must be included in the RavenpyModel instance.
          - If the meteorological data is gridded, new weights will be computed using the HRU file in the RavenpyModel instance. If that HRU
            file is different from the one used to create the original .rvh file, it may lead to inconsistencies or errors.
          - Similarly, only the list of subbasins to output will be modified in the new .rvh file. Any additional changes to the HRU or
            other components might also lead to inconsistencies or errors.

        A backup of the original files will be created before any modifications are made.
        """
        # Update the .rvi file
        if rvi_dates:
            # Backup the existing .rvi file
            shutil.copy(
                self.workdir / f"{self.run_name}.rvi",
                self.workdir / f"{self.run_name}_backup_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.rvi",
            )

            # Read the existing .rvi file and update the start date and end date
            with (self.workdir / f"{self.run_name}.rvi").open("r") as file:
                lines = file.readlines()

            start_date = pd.to_datetime(self.start_date).strftime("%Y-%m-%d %H:%M:%S")
            idx = [i for i, line in enumerate(lines) if line.startswith(":StartDate")][0]
            lines[idx] = f":StartDate            {start_date}\n"

            end_date = pd.to_datetime(self.end_date).strftime("%Y-%m-%d %H:%M:%S")
            idx = [i for i, line in enumerate(lines) if line.startswith(":EndDate")][0]
            lines[idx] = f":EndDate              {end_date}\n"

            with (self.workdir / f"{self.run_name}.rvi").open("w") as file:
                file.writelines(lines)

        if rvi_commands is not None:
            if not isinstance(rvi_commands, list):
                raise ValueError("rvi_commands must be a list of strings.")
            with (self.workdir / f"{self.run_name}.rvi").open("a") as file:
                file.write("\n")
                for command in rvi_commands:
                    file.write(command + "\n")

        if rvt:
            if all(data is None for data in [self.meteo, self.qobs]):
                warnings.warn(
                    "Meteorological data and/or observed streamflow data were not provided. The .rvt file will not be updated.", stacklevel=2
                )
            else:
                # Backup the existing .rvt file
                shutil.copy(
                    self.workdir / f"{self.run_name}.rvt",
                    self.workdir / f"{self.run_name}_backup_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.rvt",
                )

                # Update the meteorological data and observed streamflow data in the .rvt file
                rvt = rc.rvs.RVT(**(self.meteo["keys"] | ({"observation_data": self.qobs["rc"]} if self.qobs is not None else {})))
                rvt_lines = rvt.to_rv() + "\n"
                rvt_lines = rvt_lines.split("\n")
                rvt_lines = [line + "\n" for line in rvt_lines]

                # Read the existing .rvt file and replace the GriddedForcing or Gauge section
                with (self.workdir / f"{self.run_name}.rvt").open("r") as file:
                    lines = file.readlines()

                output_lines = []
                for line in lines:
                    # If the line starts with :GriddedForcing, :Gauge, or :ObservationalData, replace everything after with the new rvt_lines
                    if line.startswith(":GriddedForcing") or line.startswith(":Gauge") or line.startswith(":ObservationalData"):
                        output_lines.extend(rvt_lines)
                        break
                    else:
                        output_lines.append(line)

                # Overwrite the .rvt file with the new lines
                with (self.workdir / f"{self.run_name}.rvt").open("w") as file:
                    file.writelines(output_lines)

        if rvh:
            # Get the requested output
            output = self.hru.get("output_subbasins", None)
            if output is None:
                warnings.warn(
                    "Changes to the .rvh file were requested, but no output subbasins were defined in the HRU properties. "
                    "The .rvh file will not be updated.",
                    stacklevel=2,
                )
            else:
                # Backup the existing .rvh file
                shutil.copy(
                    self.workdir / f"{self.run_name}.rvh",
                    self.workdir / f"{self.run_name}_backup_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.rvh",
                )

                # Read the existing .rvh file and update the hydrological response variables
                with (self.workdir / f"{self.run_name}.rvh").open("r") as file:
                    lines = file.readlines()

                output = [int(o) for o in output]

                output_lines = []
                changes_active = False
                for line in lines:
                    if line == ":SubBasins\n":
                        changes_active = True
                        hruid_idx = None
                        gauged_idx = None
                    if line == ":EndSubBasins\n":
                        changes_active = False
                    if changes_active:
                        if "GAUGED" in line:
                            l_split = line.split(",")
                            l_split = [li for li in l_split if li.strip() != ":Attributes"]
                            hruid_idx = [idx for idx, line in enumerate(l_split) if " ID" in line]
                            gauged_idx = [idx for idx, line in enumerate(l_split) if "GAUGED" in line]
                            if len(hruid_idx) != 1 and len(gauged_idx) != 1:
                                raise ValueError("Could not determine unique HRU and GAUGED columns in the .rvh file.")
                            hruid_idx = hruid_idx[0]
                            gauged_idx = gauged_idx[0]
                        if gauged_idx is not None:
                            l_split = line.split()
                            try:
                                idx = int(l_split[hruid_idx])

                                # Replace the gauged element, accounting for the real whitespace between each non-space value
                                whitespaces = np.diff([i for i, c in enumerate(line) if c == " "])
                                actual_gauged_idx = (np.array([i for i, val in enumerate(whitespaces) if val != 1]))[gauged_idx]
                                actual_gauged_idx = np.sum(whitespaces[:actual_gauged_idx]) + 1
                                line = [li for li in line]
                                if idx in output:
                                    line[actual_gauged_idx] = "1"
                                else:
                                    line[actual_gauged_idx] = "0"
                                line = "".join(line)
                            except ValueError as e:
                                if "invalid literal for int()" in str(e):
                                    pass  # Skip lines that do not contain a valid HRU ID
                                else:
                                    raise e

                    output_lines.append(line)

                # Overwrite the .rvh file with the new lines
                with (self.workdir / f"{self.run_name}.rvh").open("w") as file:
                    file.writelines(output_lines)

    def run(self, *, overwrite: bool = False) -> xr.Dataset:
        """
        Run the Raven hydrological model and return simulated streamflow.

        Parameters
        ----------
        overwrite : bool
            If True, overwrite the existing output files. Default is False.

        Returns
        -------
        xr.Dataset
            The simulated streamflow.
        """
        # TODO: Compare the tagged version of the files with the raven-hydro version
        # FIXME: overwrite is currently not working as intended in RavenPy. Remove this once it is fixed.
        if overwrite is False and Path.is_file(self.workdir / "output" / f"{self.run_name}_Hydrographs.nc"):
            raise FileExistsError(f"Output files already exist in {self.workdir / 'output'}. Use 'overwrite=True' to overwrite them.")

        if self.executable is None:
            run(modelname=self.run_name, configdir=self.workdir, overwrite=overwrite)
        else:
            executable = str(Path(self.executable))
            if "raven" not in executable.lower():
                raise ValueError(
                    "The executable command does not seem to be a valid Raven command. "
                    "Please check the 'executable' parameter."
                )

            # Since we bypassed RavenPy, we need to clean up the output directory
            for file in (self.workdir / "output").glob(f"{self.run_name}*.nc"):
                file.unlink()

            subprocess.run(  # noqa: S603
                [
                    executable,
                    self.workdir / f"{self.run_name}",
                    "-o",
                    self.workdir / "output",
                ],
                check=True,
                stdin=subprocess.DEVNULL,
            )

        self._standardise_outputs()

        return self.get_streamflow()

    def get_inputs(self, subset_time: bool = False, **kwargs) -> xr.Dataset:
        r"""
        Return the inputs used to run the Raven model.

        Parameters
        ----------
        subset_time : bool
            If True, only return the weather data for the time period specified in the configuration file.
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_mfdataset`.

        Returns
        -------
        xr.Dataset
            The meteorological data used to run the Raven model simulation.
        """
        rvt_file = self.workdir / f"{self.run_name}.rvt"
        with Path.open(rvt_file) as f:
            lines = f.readlines()
        all_files = {line.split(":FileNameNC")[1].replace("\n", "").replace(" ", "") for line in lines if ":FileNameNC" in line}

        ds = xr.open_mfdataset(*list(all_files), **kwargs)

        if subset_time:
            rvi_file = self.workdir / f"{self.run_name}.rvi"
            with Path.open(rvi_file) as f:
                lines = f.readlines()
            start_date = {line.split(":StartDate")[1].replace("\n", "").replace(" ", "") for line in lines if ":StartDate" in line}
            end_date = {line.split(":EndDate")[1].replace("\n", "").replace(" ", "") for line in lines if ":EndDate" in line}
            if len(start_date) != 1 or len(end_date) != 1:
                raise ValueError("Could not find a valid start or end date in the .rvi file.")
            start_date = dt.datetime.strptime(list(start_date)[0], "%Y-%m-%d%H:%M:%S")
            end_date = dt.datetime.strptime(list(end_date)[0], "%Y-%m-%d%H:%M:%S")
            ds = ds.sel(time=slice(start_date, end_date))

        return ds

    def get_streamflow(self, output: Literal["q", "all", "path"] = "q", **kwargs) -> xr.Dataset | Path:
        r"""
        Return the simulated streamflow from the Raven model.

        Parameters
        ----------
        output : {"q", "all", "path"}
            The type of output to return. If "q", return only the streamflow variable.
            If "all", return the entire hydrograph dataset.
            If "path", return the path to the streamflow file.
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Returns
        -------
        xr.Dataset
            The streamflow file.
        Path
            The path to the streamflow file if output is set to "path".
        """
        outputs = ravenpy.OutputReader(run_name=self.run_name, path=self.workdir / "output")

        if output == "path":
            return Path(outputs.files["hydrograph"])
        else:
            if output == "q":
                return xr.open_dataset(outputs.files["hydrograph"], **kwargs)[["q"]]
            else:
                return xr.open_dataset(outputs.files["hydrograph"], **kwargs)

    def _read_qobs(self, qobs_file: os.PathLike | str, alt_name_flow: str | None = "q") -> None:
        """Read the observed streamflow data from a NetCDF file and update the .qobs properties of the RavenPy model."""
        self.qobs = {
            "file": Path(qobs_file),
            "alt_name_flow": alt_name_flow,
        }

        with xr.open_dataset(qobs_file) as ds_qobs:
            ds_qobs = ds_qobs.squeeze()

            # Try to get the basin name or ID variable as a string
            basin_id = None
            if "timeseries_id" in ds_qobs.cf.cf_roles:
                basin_id = ds_qobs.cf.cf_roles["timeseries_id"][0]
            else:
                basin_name = [a for a in ["basin_id", "basin_name", "subbasin_id"] if a in ds_qobs]
                if len(basin_name) == 1:
                    basin_id = basin_name[0]
                elif len(basin_name) > 1:
                    raise ValueError(
                        "Multiple possible basin ID variables found in the observed streamflow dataset. Please ensure only one of 'basin_id', "
                        "'basin_name', or 'subbasin_id' is present."
                    )
            station_id = "station_id" if "station_id" in ds_qobs else None

            # If the dataset is a single time series, we can manage a missing basin_id
            if basin_id is None and ds_qobs.dims.keys() == {"time"}:
                self.qobs["basin_id"] = [1]
                self.qobs["station_id"] = ["0"] if station_id is None else ds_qobs[station_id].astype(str).values.tolist()
            elif basin_id is None:
                raise ValueError(
                    "The observed streamflow dataset must contain a basin ID variable. Please ensure one of 'basin_id', 'basin_name', "
                    "or 'subbasin_id' is present, or that the dataset contains a time series with an attribute 'cf_role' set to 'timeseries_id'."
                )
            else:
                # Extract the basin and station IDs
                self.qobs["basin_id"] = [int(i) for i in ds_qobs[basin_id].values]
                self.qobs["station_id"] = (
                    [str(i) for i in range(len(self.qobs["basin_id"]))] if station_id is None else ds_qobs[station_id].astype(str).values.tolist()
                )

        self.qobs["rc"] = [
            rc.commands.ObservationData.from_nc(
                qobs_file,
                alt_names=alt_name_flow,
                station_idx=i + 1,
                uid=self.qobs["basin_id"][i],
            )
            for i in range(len(self.qobs["basin_id"]))
        ]

    def _read_hru(  # noqa: C901
        self,
        hru: gpd.GeoDataFrame | dict | os.PathLike | str,
        output_subbasins: Literal["all", "qobs"] | list[int] | None = None,
        minimum_reservoir_area: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Read the HRU properties from a GeoDataFrame and update the .hru properties of the RavenPy model. Might also create a new HRU file."""
        hru = deepcopy(hru)

        self.hru = {"keys": {}}
        has_changed = False
        hru_file = None
        if isinstance(hru, gpd.GeoDataFrame | dict):
            has_changed = True

        # Manage the input to ensure it is a GeoDataFrame
        if isinstance(hru, str | os.PathLike):
            hru_file = Path(hru)
            hru = gpd.read_file(Path(hru))
        elif isinstance(hru, dict):
            gpd_info = {k: [v] for k, v in hru.items() if k not in ["geometry", "crs"]}
            hru = gpd.GeoDataFrame(
                gpd_info,
                geometry=[hru.get("geometry")],
                crs=hru.get("crs"),
            )
        elif not isinstance(hru, gpd.GeoDataFrame):
            raise TypeError("The 'hru' parameter must be a GeoDataFrame, a dictionary, or a path to a shapefile.")

        # Manage the output subbasins
        if output_subbasins is not None:
            has_changed = True
            if output_subbasins == "qobs":
                if self.qobs is None:
                    raise ValueError("The 'qobs' parameter is set to 'qobs', but no observed streamflow data is provided.")
                basin_ids = self.qobs["basin_id"]

                # Also fill in the station names to match the observed streamflow data
                station_ids = self.qobs["station_id"]
                hru["Obs_NM"] = hru["SubId"].map(dict(zip(basin_ids, station_ids, strict=False)))
            elif output_subbasins == "all":
                basin_ids = hru["SubId"].unique()
            elif isinstance(output_subbasins, list):
                basin_ids = output_subbasins
            else:
                raise ValueError(
                    f"The 'output_subbasins' parameter must be either 'all', 'qobs', or a list of basin IDs. Got '{output_subbasins}' instead."
                ) from None

            self.hru["output_subbasins"] = basin_ids
            hru["Has_Gauge"] = hru["SubId"].isin(basin_ids)
            hru["Obs_NM"] = hru["Obs_NM"].fillna("")

        try:
            # Extract the basin properties
            rvh_config = ravenpy.extractors.BasinMakerExtractor(hru).extract()
            self.hru["keys"]["HRUs"] = rvh_config["hrus"]

        except KeyError as err:
            rvh_config = None
            # If the HRU is not compatible with the BasinMakerExtractor, we try to manage it manually
            if len(hru) != 1:
                raise ValueError(
                    "If using multiple HRUs, the HRU GeoDataFrame must contain every property required by the BasinMakerExtractor."
                ) from err
            if any(col not in hru.columns for col in ["HRU_ID", "hru_type", "SubId", "DowSubId"]):
                has_changed = True

            hru["HRU_ID"] = hru.get("HRU_ID", "1")
            hru["hru_type"] = hru.get("hru_type", "land")

            # These two are only required for gridded meteorological data, but we add them anyway for consistency
            hru["SubId"] = hru.get("SubId", 1)
            hru["DowSubId"] = hru.get("DowSubId", -1)

            hru_dict = hru.reset_index().squeeze()
            self.hru["keys"]["HRUs"] = [
                {
                    "area": hru_dict["area"],
                    "elevation": hru_dict["elevation"],
                    "latitude": hru_dict["latitude"],
                    "longitude": hru_dict["longitude"],
                    "hru_type": hru_dict["hru_type"],
                    "hru_id": hru_dict["HRU_ID"],
                }
            ]

        self.hru["hru"] = hru
        self.hru["saved_on_disk"] = False if has_changed else True
        if hru_file is None or has_changed:
            self.hru["file"] = self.workdir / "geospatial" / f"{self.run_name}_hru.gpkg"
        else:
            self.hru["file"] = hru_file

        # Special considerations for HBVEC
        if rvh_config is not None and self.model_name == "HBVEC":
            # Q_REFERENCE is needed for the DIFFUSIVE_ROUTE routing scheme, which is not default but could be used by the user.
            if "sub_basins" in rvh_config and any(q in hru.columns for q in ["Q_Mean", "Q_REFERENCE"]):
                if any(k in self.kwargs for k in ["SubBasinProperties", "sub_basin_properties"]):
                    warnings.warn(
                        "The 'SubBasinProperties' parameter is already set in the model configuration. "
                        "The Q_REFERENCE values from the HRU file will not be added to it.",
                        stacklevel=2,
                    )
                else:
                    col_name = "Q_Mean" if "Q_Mean" in hru.columns else "Q_REFERENCE"

                    # Warning: This entry would completely overwrite the default SubBasinProperties, so we need to copy the existing ones first
                    # and add Q_REFERENCE to it.
                    default_sub_basin_properties = rc.emulators.HBVEC.model_fields["sub_basin_properties"].default
                    sb_parameters = default_sub_basin_properties["parameters"].copy()
                    records_values = default_sub_basin_properties["records"][0]["values"]

                    self.hru["keys"]["sub_basin_properties"] = {
                        "parameters": sb_parameters + ["Q_REFERENCE"],
                        "records": [
                            {
                                "sb_id": subid["subbasin_id"],
                                "values": tuple(
                                    list(records_values)
                                    + [
                                        hru.loc[
                                            hru["SubId"] == subid["subbasin_id"],
                                            col_name,
                                        ].mean()
                                    ]
                                ),
                            }
                            for subid in rvh_config["sub_basins"]
                        ],
                    }

            if "reservoirs" in rvh_config and len(rvh_config["reservoirs"]) > 0:
                # Set the initial storage of implicit lakes at 1000 mm
                if "hru_state_variable_table" in self.kwargs:
                    warnings.warn(
                        "The 'hru_state_variable_table' parameter is already set in the model configuration. "
                        "The initial storage for implicit lakes will not be modified.",
                        stacklevel=2,
                    )
                else:
                    storage_name = (
                        self.kwargs.get("LakeStorage") or self.kwargs.get("lake_storage") or "SOIL[2]"  # Default for HBVEC
                    )
                    self.hru["keys"]["hru_state_variable_table"] = [
                        {
                            "hru_id": res["hru_id"],
                            "data": {storage_name: 1000},
                        }
                        for res in rvh_config["reservoirs"]
                    ]

                # Filter the reservoirs based on the minimum reservoir area, but after setting an initial storage on all lakes
                if minimum_reservoir_area is not None:
                    minimum_reservoir_area = convert_units_to(minimum_reservoir_area, "m2")
                    reservoirs = hru.loc[
                        (hru["LakeArea"] >= minimum_reservoir_area) & (hru["HRU_IsLake"] > 0),
                        "SubId",
                    ].unique()
                    rvh_config["reservoirs"] = [r for r in rvh_config["reservoirs"] if r["subbasin_id"] in reservoirs]

            # Add information from the BasinMakerExtractor
            for key in rvh_config.keys():
                if key != "hrus" and len(rvh_config[key]) > 0:
                    self.hru["keys"][key] = rvh_config[key]

    def _read_meteo(
        self,
        meteo_file: os.PathLike | str,
        data_type: list[str],
        alt_names_meteo: dict | None = None,
        meteo_station_properties: dict | None = None,
        gridweights: os.PathLike | str | None = None,
    ) -> None:
        """Read the meteorological data from a NetCDF file and update the .meteo properties of the RavenPy model."""
        self.meteo = {
            "file": Path(meteo_file),
            "data_type": data_type,
            "alt_names_meteo": alt_names_meteo,
            "station_properties": meteo_station_properties,
            "keys": {},
        }

        # Get some properties from the meteorological file
        with xr.open_dataset(self.meteo["file"]) as ds:
            # Station-based meteorological data (Hard-coding is fine here, since RavenPy only supports station data with a 'station_id' dimension)
            if "station_id" in ds.dims or ds.dims.keys() == {"time"}:
                self.meteo["type"] = "station"
                self.meteo["station_len"] = len(ds.station_id) if "station_id" in ds.dims else 1

                if self.meteo["station_len"] > 5:
                    warnings.warn(
                        "Multiple stations were provided in the meteorological data. Be aware that Raven is very inefficient and will"
                        " open the file for each station and each variable. Try to use gridded data if possible for better performance.",
                        UserWarning,
                        stacklevel=2,
                    )

            elif ds.cf.axes.get("X") is not None:
                if self.hru is None:
                    raise ValueError("The HRU properties must be defined before reading gridded meteorological data.")

                self.meteo["type"] = "grid"

                # Other required properties
                self.meteo["dim_names"] = (ds.cf.axes["X"][0], ds.cf.axes["Y"][0])
                self.meteo["var_names"] = (
                    ds.cf.coordinates["longitude"][0],
                    ds.cf.coordinates["latitude"][0],
                )
                self.meteo["elevation_name"] = ds.cf.coordinates["vertical"][0]

                # Raven requires that the data is in T,Y,X order
                for v in self.meteo["data_type"]:
                    v = alt_names_meteo.get(v, v)
                    if ds[v].dims != (
                        "time",
                        self.meteo["dim_names"][1],
                        self.meteo["dim_names"][0],
                    ):
                        raise ValueError(
                            "All variables in the meteorological dataset must have the dimensions (time, Y, X). "
                            "Please use the 'xhydro.modelling.format_input' function to ensure the data is in the correct format."
                        )

            else:
                raise ValueError(
                    "Could not determine the type of meteorological data. Please use the 'xhydro.modelling.format_input' function "
                    "to ensure the data is in the correct format."
                )

        # Prepare the meteorological data commands
        if self.meteo["type"] == "station":
            self.meteo["keys"]["Gauge"] = [
                rc.commands.Gauge.from_nc(
                    self.meteo["file"],
                    data_type=self.meteo["data_type"],
                    alt_names=self.meteo["alt_names_meteo"],
                    data_kwds=self.meteo["station_properties"],
                    station_idx=i + 1,  # RavenPy uses 1-based indexing for stations
                )
                for i in range(self.meteo["station_len"])
            ]
        else:
            if not self.hru["saved_on_disk"]:
                Path(self.hru["file"].parent).mkdir(parents=True, exist_ok=True)
                self.hru["hru"].to_file(str(self.hru["file"]))
                self.hru["saved_on_disk"] = True

            # Compute the weights
            if gridweights is None:
                gridweights = self.workdir / "weights" / f"{self.meteo['file'].stem}_vs_{self.hru['file'].stem}_weights.txt"

                weights = ravenpy.extractors.GridWeightExtractor(
                    input_file_path=self.meteo["file"],
                    routing_file_path=self.hru["file"],
                    dim_names=self.meteo["dim_names"],
                    var_names=self.meteo["var_names"],
                    routing_id_field="HRU_ID",
                ).extract()
                gw_cmd = rc.commands.GridWeights(**weights)
                gridweights.parent.mkdir(parents=True, exist_ok=True)
                gridweights.write_text(gw_cmd.to_rv() + "\n")

            # Meteo configuration
            self.meteo["keys"]["GriddedForcing"] = [
                rc.commands.GriddedForcing.from_nc(
                    self.meteo["file"],
                    data_type=v,
                    alt_names=(self.meteo["alt_names_meteo"][v] if self.meteo["alt_names_meteo"] is not None else None),
                    data_kwds=self.meteo["station_properties"],
                    station_idx=None,  # FIXME: This can be removed once we have ravenpy >= 0.18.3
                    engine="h5netcdf",
                    GridWeights=rc.commands.RedirectToFile(gridweights),
                    ElevationVarNameNC=self.meteo["elevation_name"],
                    DimNamesNC=list(self.meteo["dim_names"])  # This must always be X, Y, T regardless of the input data
                    + ["time"],
                    # Longitude/Latitude names are only set if they differ from the dimension names (aka. non-regular grids)
                    LongitudeVarNameNC=(self.meteo["var_names"][0] if self.meteo["var_names"][0] != self.meteo["dim_names"][0] else None),
                    LatitudeVarNameNC=(self.meteo["var_names"][1] if self.meteo["var_names"][1] != self.meteo["dim_names"][1] else None),
                )
                for v in data_type
            ]

    def _standardise_outputs(self, **kwargs):
        r"""
        Standardise the outputs of the simulation to be more consistent with CF conventions.

        Parameters
        ----------
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Notes
        -----
        Be aware that since systems such as Windows do not allow to overwrite files that are currently open,
        a temporary file will be created and then renamed to overwrite the original file.
        """
        filename = Path(self.get_streamflow(output="path", **kwargs))

        with self.get_streamflow(output="all", **kwargs) as ds:
            ds = ds.rename({"q_sim": "q"})
            ds["q"] = convert_units_to(ds["q"], "m3 s-1")
            ds["q"].attrs.update(
                {
                    "standard_name": "water_volume_transport_in_river_channel",
                    "cell_methods": "time: mean",
                    "long_name": "Simulated streamflow",
                    "description": "Simulated streamflow at the outlet of the subbasin.",
                }
            )

            ds = ds.swap_dims({"nbasins": "basin_name"}).rename({"basin_name": "subbasin_id"})
            ds = ds.squeeze()
            if "subbasin_id" in ds.dims:
                ds["subbasin_id"].attrs["cf_role"] = "timeseries_id"
                chunks = estimate_chunks(ds, dims=["subbasin_id"], target_mb=5)

                # FIXME: This should be fixed upstream. This is raising a 'OverflowError: can't convert negative value to hsize_t'
                for k, v in chunks.items():
                    if v == -1:
                        chunks[k] = len(ds[k])

            else:
                # Since we squeezed the dataset and renamed basin_name, it is preferable to call xs.io.rechunk_for_saving
                # anyway to clean chunk encoding.
                chunks = {"time": len(ds.time)}

            # Global attributes are already pretty good, but make the Raven version explicit
            # Since the executable used might differ from raven-hydro, we trust the dataset's history
            ds.attrs["Raven_version"] = ds.attrs.get("history", "Raven unknown").split("Raven ")[-1]
            if run is not None:
                ds.attrs["RavenPy_version"] = ravenpy.__version__

            # Overwrite the file
            save_to_netcdf(
                ds,
                filename.parent / "streamflow_tmp.nc",
                rechunk=chunks,
                netcdf_kwargs={"encoding": {"q": {"dtype": "float32", "zlib": True, "complevel": 1}}},
            )

        # Remove the original file and rename the new one
        filename.unlink()
        (filename.parent / "streamflow_tmp.nc").rename(
            filename,
        )
