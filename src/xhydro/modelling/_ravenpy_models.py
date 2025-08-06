"""Implement the ravenpy handler class for emulating raven models in ravenpy."""

import datetime as dt
import logging
import os
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import xarray as xr
from xclim.core.units import convert_units_to

try:
    import ravenpy.config.emulators
    from ravenpy import OutputReader
    from ravenpy.config import commands as rc
    from ravenpy.config.commands import GridWeights
    from ravenpy.extractors import BasinMakerExtractor, GridWeightExtractor
    from ravenpy.ravenpy import run

    ravenpy_err_msg = None
except (ImportError, RuntimeError) as e:
    run = None
    ravenpy_err_msg = e

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
        For distributed models, it should be readable by ravenpy.extractors.BasinMakerExtractor.
        For lumped models, should contain the following variables:
        - area: The watershed drainage area, in km².
        - elevation: The elevation of the watershed, in meters.
        - latitude: The latitude of the watershed centroid.
        - longitude: The longitude of the watershed centroid.
        - HRU_ID: The ID of the HRU (required for gridded data, optional for station data).
        If the meteorological data is gridded, the HRU dataset must also contain a SubId, DowSubId, valid geometry and crs.
        A file will be created in the workdir/weights subdirectory.
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
    alt_names_meteo : dict, optional
        A dictionary that allows users to link the names of meteorological variables in their dataset to Raven-compliant names.
        The keys should be the Raven names as listed in the data_type parameter.
    meteo_station_properties : dict, optional
        Additional properties of the weather stations providing the meteorological data. Only required if absent from the 'meteo_file'.
        For single stations, the format is {"ALL": {"elevation": elevation, "latitude": latitude, "longitude": longitude}}.
        This has not been tested for multiple stations or gridded data.
    qobs_file : str | Path, optional
        Path to the file containing the observed streamflow data.
    alt_name_flow : str, optional
        Name of the streamflow variable in the observed data file. If not provided, it will be assumed to be "q".
    minimum_reservoir_area : str, optional
        Quantified string (e.g. "20 km2") representing the minimum lake area to consider the lake explicitly as a reservoir.
        If not provided, all lakes with the 'HRU_IsLake' column set to 1 in the HRU file will be considered as reservoirs.
        Note that 'reservoirs' in Raven can also refer to natural lakes with weir-like outflows.
        Only applicable for distributed HBVEC models.
    output_subbasins : {"all", "qobs"}, optional
        If "all", all subbasins will be outputted. If "qobs", only the subbasins with observed flow will be outputted.
        Leave as None to use the value as defined in the HRU file ('Has_Gauge' column). Only applicable for distributed HBVEC models.
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
        alt_names_meteo: dict | None = None,
        meteo_station_properties: dict | None = None,
        qobs_file: str | Path | None = None,
        alt_name_flow: str = "q",
        minimum_reservoir_area: str | None = None,
        output_subbasins: Literal["all", "qobs"] | None = None,
        **kwargs,
    ):
        """Initialize the RavenPy model class."""
        # Set a few variables
        self.run_name = run_name or "raven"

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
                hru=hru,
                data_type=data_type,
                alt_names_meteo=alt_names_meteo,
                meteo_station_properties=meteo_station_properties,
                qobs_file=qobs_file,
                alt_name_flow=alt_name_flow,
                minimum_reservoir_area=minimum_reservoir_area,
                output_subbasins=output_subbasins,
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
        alt_names_meteo: dict | None = None,
        meteo_station_properties: dict | None = None,
        qobs_file: str | Path | None = None,
        alt_name_flow: str = "q",
        minimum_reservoir_area: str | None = None,
        output_subbasins: Literal["all", "qobs"] | None = None,
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
            For distributed models, it should be readable by ravenpy.extractors.BasinMakerExtractor.
            For lumped models, should contain the following variables:
            - area: The watershed drainage area, in km².
            - elevation: The elevation of the watershed, in meters.
            - latitude: The latitude of the watershed centroid.
            - longitude: The longitude of the watershed centroid.
            - HRU_ID: The ID of the HRU (required for gridded data, optional for station data).
            If the meteorological data is gridded, the HRU dataset must also contain a SubId, DowSubId, valid geometry and crs.
            A file will be created in the workdir/weights subdirectory.
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
        alt_names_meteo : dict, optional
            A dictionary that allows users to link the names of meteorological variables in their dataset to Raven-compliant names.
            The keys should be the Raven names as listed in the data_type parameter.
        meteo_station_properties : dict, optional
            Additional properties of the weather stations providing the meteorological data. Only required if absent from the 'meteo_file'.
            For single stations, the format is {"ALL": {"elevation": elevation, "latitude": latitude, "longitude": longitude}}.
            This has not been tested for multiple stations or gridded data.
        qobs_file : str | Path, optional
            Path to the file containing the observed streamflow data.
            If using a distributed model, the file must contain a 'basin_id' dimension that matches the 'SubId' column in the HRU file.
            Additionally, it can contain a 'station_id' coordinate to identify the name of the station, which will be used to fill the 'Obs_NM'
            column in the HRU file.
        alt_name_flow : str, optional
            Name of the streamflow variable in the observed data file. If not provided, it will be assumed to be "q".
        minimum_reservoir_area : str, optional
            Quantified string (e.g. "20 km2") representing the minimum lake area to consider the lake explicitly as a reservoir.
            If not provided, all lakes with the 'HRU_IsLake' column set to 1 in the HRU file will be considered as reservoirs.
            Note that 'reservoirs' in Raven can also refer to natural lakes with weir-like outflows.
            Only applicable for distributed HBVEC models.
        output_subbasins : {"all", "qobs"}, optional
            If "all", all subbasins will be outputted.
            If "qobs", subbasins with observed flow will be outputted, as defined by the basin IDs in the observed streamflow data.
            Leave as None to use the value as defined in the HRU file ('Has_Gauge' column). Only applicable for distributed HBVEC models.
        overwrite : bool
            If True, overwrite the existing project files. Default is False.
            Note that to prevent inconsistencies, all files containing the 'run'name' will be removed, including the output files.
        \*\*kwargs : dict, optional
            Additional parameters to pass to the RavenPy emulator, to modify the default modules used by a given hydrological model.
            Typical entries include RainSnowFraction or Evaporation.
            See https://raven.uwaterloo.ca/Downloads.html for the latest Raven documentation. Currently, model templates are listed in Appendix F.
        """
        if run is None:
            raise RuntimeError(
                "RavenPy is not installed or not properly configured. The RavenpyModel.create_rv method cannot be used without it."
                f" Original error: {ravenpy_err_msg}"
            )
        kwargs = deepcopy(kwargs)
        hru = deepcopy(hru)

        # Remove any existing files in the project directory
        if len([f for f in self.workdir.rglob(f"{self.run_name}.rv*")]) > 0:
            if overwrite:
                for file in self.workdir.rglob(f"{self.run_name}*.*"):
                    file.unlink()
            else:
                raise FileExistsError(
                    f"Project {self.run_name} in {self.workdir} already exists, but 'overwrite' is set to False."
                )

        # Add the observed streamflow data if provided
        obs_basin_ids = None
        obs_station_ids = None
        if qobs_file is not None:
            with xr.open_dataset(qobs_file) as ds_qobs:
                ds_qobs = ds_qobs.squeeze()
                if len(ds_qobs.dims) == 1:
                    # If the dataset is 1D, it is a single station
                    obs_basin_ids = [1]
                    obs_station_ids = ["0"]
                else:
                    # If the dataset is 2D, it contains multiple stations
                    if "basin_id" not in ds_qobs:
                        raise ValueError(
                            "The observed streamflow dataset must contain a 'basin_id' variable."
                        )
                    obs_basin_ids = [int(i) for i in ds_qobs.basin_id]
                    if "station_id" not in ds_qobs:
                        obs_station_ids = [str(i) for i in range(len(obs_basin_ids))]
                    else:
                        obs_station_ids = ds_qobs.station_id.astype(str).values.tolist()

            kwargs["ObservationData"] = [
                rc.ObservationData.from_nc(
                    qobs_file,
                    alt_names=alt_name_flow,
                    station_idx=i + 1,
                    uid=obs_basin_ids[i],
                )
                for i in range(len(obs_basin_ids))
            ]

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
        if isinstance(hru, str | os.PathLike):
            hru = gpd.read_file(Path(hru))
        if isinstance(hru, dict):
            gpd_info = {k: [v] for k, v in hru.items() if k not in ["geometry", "crs"]}
            hru = gpd.GeoDataFrame(
                gpd_info,
                geometry=[hru.get("geometry")],
                crs=hru.get("crs"),
            )
        hru_file = self.workdir / "weights" / f"{self.run_name}_hru.shp"

        # Extract the basin properties
        try:
            # Manage a few things while we still have a GeoDataFrame
            if output_subbasins is not None:
                if output_subbasins == "qobs":
                    hru["Has_Gauge"] = (hru["SubId"].isin(obs_basin_ids)).astype(int)
                    for s in obs_basin_ids:
                        hru.loc[hru["SubId"] == s, "Obs_NM"] = obs_station_ids[
                            obs_basin_ids.index(s)
                        ]
                elif output_subbasins == "all":
                    hru["Has_Gauge"] = 1
                    hru["Obs_NM"] = hru["Obs_NM"].fillna("")
                else:
                    raise ValueError(
                        f"The 'output_subbasins' parameter must be either 'all' or 'qobs'. Got '{output_subbasins}' instead."
                    )

            rvh_config = BasinMakerExtractor(hru).extract()
            kwargs["HRUs"] = rvh_config["hrus"]

            # The GridWeightExtractor requires a file on disk, so we save the HRU as a shapefile.
            Path(hru_file.parent).mkdir(parents=True, exist_ok=True)
            hru.to_file(
                str(hru_file),
            )

        # Manage simplistic HRU inputs that do not contain all the required properties
        except KeyError:
            if len(hru) != 1:
                raise ValueError(
                    "If using multiple HRUs, the HRU GeoDataFrame must contain every property required by the BasinMakerExtractor."
                )

            hru["HRU_ID"] = hru.get("HRU_ID", "1")
            hru["hru_type"] = hru.get("hru_type", "land")
            hru["SubId"] = hru.get(
                "SubId", 1
            )  # These two are only required for gridded meteorological data
            hru["DowSubId"] = hru.get("DowSubId", -1)

            if meteo_type == "grid":
                if hru.get("geometry") is None or hru.crs is None:
                    raise ValueError(
                        "The HRU dataset must contain a geometry and a CRS when the meteorological data is gridded."
                    )
                # The GridWeightExtractor requires a file on disk, so we save the HRU as a shapefile.
                Path(hru_file.parent).mkdir(parents=True, exist_ok=True)
                hru.to_file(
                    str(hru_file),
                )

            hru = hru.reset_index().squeeze()
            kwargs["HRUs"] = [
                {
                    "area": hru["area"],
                    "elevation": hru["elevation"],
                    "latitude": hru["latitude"],
                    "longitude": hru["longitude"],
                    "hru_type": hru["hru_type"],
                    "hru_id": hru["HRU_ID"],
                }
            ]

        # Special considerations for distributed models (currently, only HBVEC)
        if model_name == "HBVEC":
            if "sub_basins" in rvh_config:
                # Add the subbasin reference flow
                if (
                    not any(
                        k in kwargs
                        for k in ["SubBasinProperties", "sub_basin_properties"]
                    )
                    and "Q_Mean" in hru.columns
                ):
                    # Warning: This entry will completely overwrite the default SubBasinProperties, so we need to copy the existing ones
                    # and add REFERENCE_Q to it.
                    default_sub_basin_properties = (
                        ravenpy.config.emulators.HBVEC.model_fields[
                            "sub_basin_properties"
                        ].default
                    )
                    sb_parameters = default_sub_basin_properties["parameters"].copy()
                    records_values = default_sub_basin_properties["records"][0][
                        "values"
                    ]

                    kwargs["sub_basin_properties"] = {
                        "parameters": sb_parameters + ["Q_REFERENCE"],
                        "records": [
                            {
                                "sb_id": subid["subbasin_id"],
                                "values": tuple(
                                    list(records_values)
                                    + [
                                        hru.loc[
                                            hru["SubId"] == subid["subbasin_id"],
                                            "Q_Mean",
                                        ].mean()
                                    ]
                                ),
                            }
                            for subid in rvh_config["sub_basins"]
                        ],
                    }

            if "reservoirs" in rvh_config and len(rvh_config["reservoirs"]) > 0:
                # Initial storage at 1000 mm
                if "hru_state_variable_table" not in kwargs:
                    storage_name = (
                        kwargs.get("LakeStorage")
                        or kwargs.get("lake_storage")
                        or "SOIL[2]"
                    )
                    kwargs["hru_state_variable_table"] = [
                        {
                            "hru_id": res["hru_id"],
                            "data": {storage_name: 1000},
                        }
                        for res in rvh_config["reservoirs"]
                    ]

                # Filter the reservoirs based on the minimum reservoir area, but after setting an initial storage on all lakes
                if minimum_reservoir_area is not None:
                    minimum_reservoir_area = convert_units_to(
                        minimum_reservoir_area, "m2"
                    )
                    reservoirs = hru.loc[
                        (hru["LakeArea"] >= minimum_reservoir_area)
                        & (hru["Lake_Cat"] > 0),
                        "SubId",
                    ].unique()
                    rvh_config["reservoirs"] = [
                        r
                        for r in rvh_config["reservoirs"]
                        if r["subbasin_id"] in reservoirs
                    ]

            # Add information from the BasinMakerExtractor
            for key in rvh_config.keys():
                if key != "hrus" and len(rvh_config[key]) > 0:
                    kwargs[key] = rvh_config[key]

        # Prepare the meteorological data
        if meteo_type == "station":
            if station_len > 5:
                warnings.warn(
                    "Multiple stations were provided in the meteorological data. Be aware that Raven is very inefficient and will"
                    " open the file for each station and each variable. Try to use gridded data if possible for better performance.",
                    UserWarning,
                )

            kwargs["Gauge"] = [
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
            kwargs["GriddedForcing"] = [
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
            params=parameters,
            StartDate=start_date,
            EndDate=end_date,
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
            qsim = ds[["q_sim"]].rename({"q_sim": "q"})
            qsim = qsim.swap_dims({"nbasins": "basin_name"}).rename(
                {"basin_name": "subbasin_id"}
            )
            qsim = qsim.squeeze()
            qsim["subbasin_id"].attrs["cf_role"] = "timeseries_id"

        return qsim
