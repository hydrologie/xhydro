"""Implement the ravenpy handler class for emulating raven models in ravenpy."""

import datetime as dt
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import xarray as xr

try:
    import ravenpy.config.emulators
    from ravenpy import OutputReader
    from ravenpy.config import commands as rc
    from ravenpy.config.commands import GridWeights
    from ravenpy.extractors import GridWeightExtractor
    from ravenpy.ravenpy import run
except ImportError as e:
    run = None

from ._hm import HydrologicalModel

logger = logging.getLogger(__name__)

__all__ = ["RavenpyModel"]


class RavenpyModel(HydrologicalModel):
    r"""Initialize the RavenPy model class.

    Parameters
    ----------
    overwrite : bool
        If True, overwrite the existing project files. Default is False.
    workdir : str | Path | None
        Path to save the .rv files and model outputs. Default is None, which creates a temporary directory.
    run_name : str, optional
        Name of the run, which will be used to name the project files. Defaults to "raven" if not provided.
    model_name : {"Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"}, optional
        The name of the RavenPy model to run. Only optional if the project files already exist.
    hru : gpd.GeoDataFrame | dict | os.PathLike, optional
        A GeoDataFrame, or dictionary containing the HRU properties. Only optional if the project files already exist.
        It should contain the following variables:
        - area: The watershed drainage area, in km².
        - elevation: The elevation of the watershed, in meters.
        - latitude: The latitude of the watershed centroid.
        - longitude: The longitude of the watershed centroid.
        - HRU_ID: The ID of the HRU (required for gridded data, optional for station data).
        If the meteorological data is gridded, the HRU dataset must also contain a SubId, DowSubId, valid geometry and crs. Unless the input is the
        path to a shapefile that already contains all additional properties, a file will be created in the workdir/weights subdirectory.
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
    start_date : dt.datetime | str, optional
        The first date of the simulation. Only optional if the project files already exist.
    end_date : dt.datetime | str, optional
        The last date of the simulation. Only optional if the project files already exist.
    parameters : np.ndarray | list[float], optional
        The model parameters for simulation or calibration. Only optional if the project files already exist.
    global_parameter : dict, optional
        A dictionary of global parameters for the model.
    alt_names_meteo : dict, optional
        A dictionary that allows users to link the names of meteorological variables in their dataset to Raven-compliant names.
        The keys should be the Raven names as listed in the data_type parameter.
    meteo_station_properties : dict, optional
        Additional properties of the weather stations providing the meteorological data. Only required if absent from the 'meteo_file'.
        For single stations, the format is {"ALL": {"elevation": elevation, "latitude": latitude, "longitude": longitude}}.
        This has not been tested for multiple stations or gridded data.
    \*\*kwargs : dict, optional
        Additional parameters to pass to the RavenPy emulator, to modify the default modules used by a given hydrological model.
        Typical entries include RainSnowFraction or Evaporation.
        See https://raven.uwaterloo.ca/Downloads.html for the latest Raven documentation. Currently, model templates are listed in Appendix F.
    """

    def __init__(
        self,
        overwrite: bool = False,
        *,
        workdir: str | os.PathLike | None = None,
        run_name: str | None = None,
        model_name: (
            Literal["Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"]
            | None
        ) = None,
        hru: gpd.GeoDataFrame | os.PathLike | None = None,
        meteo_file: os.PathLike | str | None = None,
        data_type: list[str] | None = None,
        start_date: dt.datetime | str | None = None,
        end_date: dt.datetime | str | None = None,
        parameters: np.ndarray | list[float] | None = None,
        global_parameter: dict | None = None,
        alt_names_meteo: dict | None = None,
        meteo_station_properties: dict | None = None,
        **kwargs,
    ):
        """Initialize the RavenPy model class."""
        # Set a few variables
        self.run_name = run_name or "raven"
        self.emulator_config = (
            {}
        )  # TODO: This is currently not used, but could probably be used to store the emulator configuration.

        if workdir is None:
            self.workdir = Path(tempfile.mkdtemp(prefix="raven-hydro_"))
        else:
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)

        files = [f for f in self.workdir.rglob(f"{self.run_name}.rv*")]
        if len(files) > 0 and not overwrite:
            logger.info("Project already exists and files will not be overwritten.")
            return
        else:
            if len(files) > 0:
                logger.info("Project already exists, but will be overwritten.")

            # Create the project files
            self.create_rv(
                model_name=model_name,
                meteo_file=meteo_file,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters,
                global_parameter=global_parameter,
                hru=hru,
                data_type=data_type,
                alt_names_meteo=alt_names_meteo,
                meteo_station_properties=meteo_station_properties,
                overwrite=overwrite,
                **kwargs,
            )

    def create_rv(  # noqa: C901
        self,
        model_name: Literal[
            "Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"
        ],
        *,
        hru: gpd.GeoDataFrame | os.PathLike,
        meteo_file: os.PathLike | str,
        data_type: list[str],
        start_date: dt.datetime | str,
        end_date: dt.datetime | str,
        parameters: np.ndarray | list[float],
        global_parameter: dict | None = None,
        alt_names_meteo: dict | None = None,
        meteo_station_properties: dict | None = None,
        overwrite: bool = False,
        **kwargs,
    ):
        r"""Write the RavenPy project files.

        Parameters
        ----------
        model_name : {"Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"}
            The name of the RavenPy model to run.
        hru : gpd.GeoDataFrame | dict | os.PathLike
            A GeoDataFrame, or dictionary containing the HRU properties.
            It should contain the following variables:
            - area: The watershed drainage area, in km².
            - elevation: The elevation of the watershed, in meters.
            - latitude: The latitude of the watershed centroid.
            - longitude: The longitude of the watershed centroid.
            - HRU_ID: The ID of the HRU (required for gridded data, optional for station data).
            If the meteorological data is gridded, the HRU dataset must also contain a SubId, DowSubId, valid geometry and crs. Unless the input
            is the path to a shapefile that already contains all additional properties, a file will be created in the workdir/weights subdirectory.
        meteo_file : str | Path
            Path to the file containing the observed meteorological data. Only optional if the project files already exist.
            The meteorological data can be either station or gridded data. Use the 'xhydro.modelling.format_input' function to ensure the data
            is in the correct format. Unless the input is a single station accompanied by 'meteo_station_properties', the file should contain
            the following coordinates:
            - elevation: The elevation of the station / grid cell, in meters.
            - latitude: The latitude of the station / grid cell centroid.
            - longitude: The longitude of the station / grid cell centroid.
        data_type : list[str]
            The list of types of data provided to Raven in the meteorological file.
            See https://github.com/CSHS-CWRA/RavenPy/blob/master/src/ravenpy/config/conventions.py for the list of available types.
        start_date : dt.datetime | str
            The first date of the simulation.
        end_date : dt.datetime | str
            The last date of the simulation.
        parameters : np.ndarray | list[float]
            The model parameters for simulation or calibration.
        global_parameter : dict, optional
            A dictionary of global parameters for the model.
        alt_names_meteo : dict, optional
            A dictionary that allows users to link the names of meteorological variables in their dataset to Raven-compliant names.
            The keys should be the Raven names as listed in the data_type parameter.
        meteo_station_properties : dict, optional
            Additional properties of the weather stations providing the meteorological data. Only required if absent from the 'meteo_file'.
            For single stations, the format is {"ALL": {"elevation": elevation, "latitude": latitude, "longitude": longitude}}.
            This has not been tested for multiple stations or gridded data.
        overwrite : bool
            If True, overwrite the existing project files. Default is False.
            Note that to prevent inconsistencies, all files containing the 'run'name' will be removed, including the output files.
        \*\*kwargs : dict, optional
            Additional parameters to pass to the RavenPy emulator, to modify the default modules used by a given hydrological model.
            Typical entries include RainSnowFraction or Evaporation.
            See https://raven.uwaterloo.ca/Downloads.html for the latest Raven documentation. Currently, model templates are listed in Appendix F.
        """
        if run is None:
            raise ImportError(
                "RavenPy is not installed. Please install it to use this class."
            )

        # Remove any existing files in the project directory
        if len([f for f in self.workdir.rglob(f"{self.run_name}.rv*")]) > 0:
            if overwrite:
                for file in self.workdir.rglob(f"{self.run_name}*.*"):
                    file.unlink()
            else:
                raise FileExistsError(
                    f"Project {self.run_name} in {self.workdir} already exists, but 'overwrite' is set to False."
                )

        global_parameter = global_parameter or {}

        # Get the meteorological data type
        meteo_file = Path(meteo_file)
        with xr.open_dataset(meteo_file) as ds:
            meteo_type = None
            if (
                ds.cf.cf_roles.get("timeseries_id") is not None
                or len(ds.squeeze().dims) == 1
            ):
                meteo_type = "station"

                # Other required properties
                station_len = (
                    len(ds[ds.cf.cf_roles["timeseries_id"][0]])
                    if ds.cf.cf_roles.get("timeseries_id") is not None
                    else 1
                )

            elif ds.cf.axes.get("X") is not None:
                meteo_type = "grid"

                # Other required properties
                dim_names = (ds.cf.axes["X"][0], ds.cf.axes["Y"][0])
                var_names = (
                    ds.cf.coordinates["longitude"][0],
                    ds.cf.coordinates["latitude"][0],
                )
            else:
                raise ValueError("Could not determine the type of meteorological data.")

        # Prepare the HRUs
        hru_file = None
        if isinstance(hru, str | os.PathLike):
            hru = Path(hru)
            hru_file = hru
            hru = gpd.read_file(hru)

        if meteo_type == "grid":
            if isinstance(hru, dict):
                # If the meteo type is grid, we need to convert it to a GeoDataFrame and save it as a shapefile
                hru = gpd.GeoDataFrame(
                    {
                        "area": [hru["area"]],
                        "elevation": [hru["elevation"]],
                        "latitude": [hru["latitude"]],
                        "longitude": [hru["longitude"]],
                        "hru_type": [hru.get("hru_type", "land")],
                        "HRU_ID": [hru.get("HRU_ID", "1")],
                        "SubId": [hru.get("SubId", 1)],
                        "DowSubId": [hru.get("DowSubId", -1)],
                    },
                    geometry=[hru["geometry"]],
                    crs=hru["crs"],
                )
                hru_file = self.workdir / "weights" / f"{self.run_name}_hru.shp"
                Path(hru_file.parent).mkdir(parents=True, exist_ok=True)
                hru.to_file(
                    str(hru_file),
                )
            else:
                # If the HRU is a GeoDataFrame, we need to check if it contains the required additional properties
                if "geometry" not in hru:
                    raise ValueError(
                        "The HRU dataset must contain a geometry when the meteorological data is gridded."
                    )
                if "SubId" not in hru:
                    hru_file = None
                    hru["SubId"] = 1
                if "DowSubId" not in hru:
                    hru_file = None
                    hru["DowSubId"] = -1
                if "HRU_ID" not in hru:
                    hru_file = None
                    hru["HRU_ID"] = "1"
                if hru_file is None:
                    # If None, then we need to create a shapefile
                    hru_file = self.workdir / "weights" / f"{self.run_name}_hru.shp"
                    Path(hru_file.parent).mkdir(parents=True, exist_ok=True)
                    hru.to_file(
                        str(hru_file),
                    )

        if isinstance(hru, gpd.GeoDataFrame):
            if len(hru) != 1:
                raise ValueError("The HRU dataset must contain only one watershed.")
            hru = hru.reset_index()
            hru = hru.squeeze().to_dict()

        # HRU input for the Raven emulator
        hru_info = {
            "area": hru["area"],
            "elevation": hru["elevation"],
            "latitude": hru["latitude"],
            "longitude": hru["longitude"],
            "hru_type": hru.get("hru_type", "land"),
        }
        if "HRU_ID" in hru:
            hru_info["hru_id"] = hru["HRU_ID"]

        # Prepare the meteorological data
        if meteo_type == "station":
            if station_len > 5:
                warnings.warn(
                    "Multiple stations were provided in the meteorological data. Be aware that Raven is very inefficient and will"
                    " open the file for each station and each variable. Try to use gridded data if possible for better performance.",
                    UserWarning,
                )

            meteo_data = [
                rc.Gauge.from_nc(
                    meteo_file,  # File path to the meteorological data
                    data_type=data_type,  # List of all the useful meteorological variables in the file
                    alt_names=alt_names_meteo,  # Mapping between the names of the required variables and those in the file.
                    data_kwds=meteo_station_properties,
                    station_idx=i + 1,  # RavenPy uses 1-based indexing for stations
                )
                for i in range(station_len)
            ]

        else:
            # Compute the weights
            weight_file = (
                self.workdir
                / "weights"
                / f"{meteo_file.stem}_vs_{hru_file.stem}_weights.txt"
            )

            weights = GridWeightExtractor(
                input_file_path=meteo_file,
                routing_file_path=hru_file,
                dim_names=dim_names,
                var_names=var_names,
                routing_id_field="HRU_ID",
            ).extract()
            gw_cmd = GridWeights(**weights)
            Path(weight_file.parent).mkdir(parents=True, exist_ok=True)
            weight_file.write_text(gw_cmd.to_rv() + "\n")

            # Meteo configuration
            meteo_data = [
                rc.GriddedForcing.from_nc(
                    meteo_file,  # Path to the file containing meteorological variables
                    data_type=v,  # List of all the variables
                    alt_names=(
                        alt_names_meteo[v] if alt_names_meteo is not None else None
                    ),  # Mapping between the names of the required variables and those in the file.
                    data_kwds=meteo_station_properties,
                    station_idx=None,  # FIXME: This can be removed once we have ravenpy >= 0.18.3
                    engine="h5netcdf",
                    GridWeights=rc.RedirectToFile(weight_file),
                    ElevationVarNameNC="elevation",
                )
                for v in data_type
            ]

        # Create the emulator configuration
        self.emulator_config = dict(
            RunName=self.run_name,
            HRUs=[hru_info],
            params=parameters,
            global_parameter=global_parameter,
            StartDate=start_date,
            EndDate=end_date,
            Gauge=meteo_data if meteo_type == "station" else None,
            GriddedForcing=meteo_data if meteo_type == "grid" else None,
            **kwargs,
        )
        model = getattr(ravenpy.config.emulators, model_name)(
            **self.emulator_config,
        )
        model.write_rv(workdir=self.workdir, overwrite=overwrite)

        # TODO: Add a tag to the files to identify the raven-hydro version

    def run(self, overwrite=False) -> str | xr.Dataset:
        """Run the Raven hydrological model and return simulated streamflow.

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
        # TODO: Allow running the model through the command line
        # FIXME: overwrite is currently not working as intended in RavenPy. Remove this once it is fixed.
        if overwrite is False and Path.is_file(
            self.workdir / "output" / f"{self.run_name}_Hydrographs.nc"
        ):
            raise FileExistsError(
                f"Output files already exist in {self.workdir / 'output'}. Use 'overwrite=True' to overwrite them."
            )

        run(modelname=self.run_name, configdir=self.workdir, overwrite=overwrite)

        # TODO: Actually reformat the output files
        # TODO: Add metadata to the output files (e.g. the version of Raven used, the emulator used, etc.)
        return self.get_streamflow()

    def get_inputs(self, subset_time: bool = False, **kwargs) -> xr.Dataset:
        r"""Return the inputs used to run the Raven model.

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
        all_files = {
            line.split(":FileNameNC")[1].replace("\n", "").replace(" ", "")
            for line in lines
            if ":FileNameNC" in line
        }

        ds = xr.open_mfdataset(*list(all_files), **kwargs)

        if subset_time:
            rvi_file = self.workdir / f"{self.run_name}.rvi"
            with Path.open(rvi_file) as f:
                lines = f.readlines()
            start_date = {
                line.split(":StartDate")[1].replace("\n", "").replace(" ", "")
                for line in lines
                if ":StartDate" in line
            }
            end_date = {
                line.split(":EndDate")[1].replace("\n", "").replace(" ", "")
                for line in lines
                if ":EndDate" in line
            }
            if len(start_date) != 1 or len(end_date) != 1:
                raise ValueError(
                    "Could not find a valid start or end date in the .rvi file."
                )
            start_date = dt.datetime.strptime(list(start_date)[0], "%Y-%m-%d%H:%M:%S")
            end_date = dt.datetime.strptime(list(end_date)[0], "%Y-%m-%d%H:%M:%S")
            ds = ds.sel(time=slice(start_date, end_date))

        return ds

    def get_streamflow(self):
        """Return the simulated streamflow from the Raven model.

        Returns
        -------
        xr.Dataset
            The simulated streamflow.
        """
        outputs = OutputReader(run_name=self.run_name, path=self.workdir / "output")

        with xr.open_dataset(outputs.files["hydrograph"]) as ds:
            qsim = ds.q_sim.to_dataset(name="qsim").rename({"qsim": "q"}).squeeze()

        return qsim
