"""Class to handle Hydrotel simulations."""

import itertools
import os
import re
import subprocess  # noqa: S404
import warnings
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ._hm import HydrologicalModel
from ._model_utils import standardize_output


__all__ = ["Hydrotel"]


class Hydrotel(HydrologicalModel):
    """
    Class to handle HYDROTEL simulations.

    Parameters
    ----------
    project_dir : str or Path
        Path to the project folder.
    project_file : str
        Name of the project file (e.g. 'projet.csv').
    executable : str or Path
        Command to execute HYDROTEL.
        On Windows, this should be the path to hydrotel.exe.
    project_config : dict, optional
        Dictionary of configuration options to overwrite in the project file.
    simulation_config : dict, optional
        Dictionary of configuration options to overwrite in the simulation file. See the Notes section for more details.
    output_config : dict, optional
        Dictionary of configuration options to overwrite in the output file (output.csv).

    Notes
    -----
    The name of the simulation file must match the name of the 'SIMULATION COURANTE' option in the project file.

    This class is designed to handle the execution of HYDROTEL simulations, with the ability to overwrite configuration options,
    but it does not handle the creation of the project folder itself. The project folder must be created beforehand.

    For more information on how to configure the project, refer to the documentation of HYDROTEL:
    https://github.com/INRS-Modelisation-hydrologique/hydrotel
    """

    def __init__(
        self,
        project_dir: str | os.PathLike,
        project_file: str,
        executable: str | os.PathLike,
        *,
        project_config: dict | None = None,
        simulation_config: dict | None = None,
        output_config: dict | None = None,
    ):
        """Initialize the HYDROTEL simulation."""
        project_config = project_config or dict()
        simulation_config = simulation_config or dict()
        output_config = output_config or dict()

        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValueError("The project folder does not exist.")

        self.config_files = dict()
        self.config_files["project"] = Path(self.project_dir / project_file).with_suffix(".csv")

        # Initialize the project, simulation, and output configuration options
        o = dict()
        # Read the configuration files from disk
        o["project_config"] = _read_csv(self.config_files["project"])

        # Get the simulation name
        if len(project_config.get("SIMULATION COURANTE", None) or o["project_config"]["SIMULATION COURANTE"]) == 0:
            raise ValueError(
                "'SIMULATION COURANTE' must be specified in either the project configuration file or as a keyword argument for 'project_config'."
            )
        sim_name = project_config.get("SIMULATION COURANTE", None) or o["project_config"]["SIMULATION COURANTE"]
        self.simulation_dir = self.project_dir / "simulation" / sim_name

        if not self.simulation_dir.is_dir():
            raise ValueError(f"The {self.simulation_dir} folder does not exist in the project directory.")

        # Read the configuration files from disk
        self.config_files["simulation"] = self.simulation_dir / f"{sim_name}.csv"
        self.config_files["output"] = self.simulation_dir / "output.csv"
        for cfg in ["simulation", "output"]:
            o[f"{cfg}_config"] = _read_csv(self.config_files[cfg])

        # Combine the configuration options provided by the user and those read from the files
        self.project_config = o["project_config"] | project_config
        self.simulation_config = o["simulation_config"] | simulation_config
        self.output_config = o["output_config"] | output_config
        # Update the configuration options on disk
        self.update_config(
            project_config=self.project_config,
            simulation_config=self.simulation_config,
            output_config=self.output_config,
        )

        # TODO: Clean up and prepare the 'etat' folder (missing the files)

        self.executable = str(Path(executable))
        self.rhhu = None

    def update_config(
        self,
        *,
        project_config: dict | None = None,
        simulation_config: dict | None = None,
        output_config: dict | None = None,
    ):
        """
        Update the configuration options in the project, simulation, and output files.

        Parameters
        ----------
        project_config : dict, optional
            Dictionary of configuration options to overwrite in the project file.
        simulation_config : dict, optional
            Dictionary of configuration options to overwrite in the simulation file.
        output_config : dict, optional
            Dictionary of configuration options to overwrite in the output file (output.csv).
        """
        if project_config is not None:
            project_config = deepcopy(_fix_os_paths(project_config))
            _overwrite_csv(self.config_files["project"], project_config)

            # Also update class attributes to reflect the changes
            for key, value in project_config.items():
                self.project_config[key] = value
            self.simulation_dir = self.project_dir / "simulation" / self.project_config["SIMULATION COURANTE"]
            self.config_files["simulation"] = self.simulation_dir / f"{self.project_config['SIMULATION COURANTE']}.csv"
            if not self.simulation_dir.is_dir():
                raise ValueError(f"The {self.simulation_dir} folder does not exist in the project directory.")

        if simulation_config is not None:
            simulation_config = deepcopy(_fix_os_paths(_fix_dates(simulation_config)))
            _overwrite_csv(self.config_files["simulation"], simulation_config)

            # Also update class attributes to reflect the changes
            for key, value in simulation_config.items():
                self.simulation_config[key] = value

        if output_config is not None:
            _overwrite_csv(self.config_files["output"], output_config)

            # Also update class attributes to reflect the changes
            for key, value in output_config.items():
                self.output_config[key] = value

    def run(
        self,
        *,
        run_options: list[str] | None = None,
        dry_run: bool = False,
        standardize: bool = True,
        return_streamflow: bool = True,
    ) -> str | xr.Dataset:
        """
        Run the simulation.

        Parameters
        ----------
        run_options : list[str] | None
            Additional options to pass to the HYDROTEL executable.
            Common arguments include:
            - `-t NUM`: Run the simulation using a given number of threads (default is 1).
            - `-c`: Skip the validation of the input files.
            - `-s`: Skip the interpolation of missing values in the input files. Only use this if you are sure that the input files are complete.
            Call the executable without arguments to see the full list of available options.
        dry_run : bool
            If True, returns the command to run the simulation without actually running it.
        standardize : bool
            If True, standardize the output files to ensure they are in a consistent format. Default is True.
        return_streamflow : bool
            If True, return the simulated streamflow. Default is True.

        Returns
        -------
        str
            The command to run the simulation, if 'dry_run' is True.
        xr.Dataset
            The streamflow file, if 'dry_run' is False.
        """
        if os.name == "nt" and Path(self.executable).suffix != ".exe":
            raise ValueError("You must specify the path to hydrotel.exe")
        if "hydrotel" not in self.executable.lower():
            raise ValueError("The executable command does not seem to be a valid HYDROTEL command. Please check the 'executable' parameter.")

        # Make sure that the files reflect the configuration
        self.update_config(
            project_config=self.project_config,
            simulation_config=self.simulation_config,
            output_config=self.output_config,
        )

        # Prepare the input call
        run_options = run_options or []

        # Unwrap elements that contain spaces
        run_options = list(itertools.chain.from_iterable([a.split() if isinstance(a, str) else a for a in run_options]))

        # If the '-t' flag is supplied, merge the next item in the list with it
        if "-t" in run_options:
            t_index = run_options.index("-t")
            try:
                int(run_options[t_index + 1])
            except (IndexError, ValueError) as err:
                raise ValueError("The '-t' flag must be followed by an integer specifying the number of threads to use.") from err
            run_options[t_index : t_index + 2] = [" ".join(run_options[t_index : t_index + 2])]
        else:
            run_options.append("-t 1")

        # HYDROTEL cares about the order of the arguments
        call = [
            self.executable,
            *[r for r in run_options if any(opt in r for opt in ["-i", "-g", "-n", "-u", "-v"])],
            str(self.config_files["project"]),
            *[r for r in run_options if any(opt in r for opt in ["-c", "-d", "-r", "-s"])],
            *[r for r in run_options if any(opt in r for opt in ["-t"])],
            *[r for r in run_options if any(opt in r for opt in ["-l"])],
        ]

        if dry_run:
            return " ".join(call)

        # Run the simulation
        subprocess.run(  # noqa: S603
            call,
            check=True,
            stdin=subprocess.DEVNULL,
        )

        # Standardize the outputs
        if standardize:
            self.standardize_outputs()

        if return_streamflow:
            return self.get_output("q")

    def get_inputs(self, subset_time: bool = False, return_config=False, **kwargs) -> xr.Dataset | tuple[xr.Dataset, dict]:
        r"""
        Get the weather file from the simulation.

        Parameters
        ----------
        subset_time : bool
            If True, only return the weather data for the time period specified in the simulation configuration file.
        return_config : bool
            Whether to return the configuration file as well. If True, returns a tuple of (dataset, configuration).
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Returns
        -------
        xr.Dataset
            If 'return_config' is False, returns the weather file.
        Tuple[xr.Dataset, dict]
            If 'return_config' is True, returns the weather file and its configuration.
        """
        # Find the right weather file
        if all(len(self.simulation_config.get(k, "")) > 0 for k in ["FICHIER GRILLE METEO", "FICHIER STATIONS METEO"]):
            raise ValueError("Both 'FICHIER GRILLE METEO' and 'FICHIER STATIONS METEO' are specified in the simulation configuration file.")
        if len(self.simulation_config.get("FICHIER GRILLE METEO", "")) > 0:
            weather_file = self.simulation_config["FICHIER GRILLE METEO"]
        elif len(self.simulation_config.get("FICHIER STATIONS METEO", "")) > 0:
            weather_file = self.simulation_config["FICHIER STATIONS METEO"]
        else:
            raise ValueError("You must specify either 'FICHIER GRILLE METEO' or 'FICHIER STATIONS METEO' in the simulation configuration file.")

        kwargs = deepcopy(kwargs)
        kwargs.setdefault("chunks", {})
        ds = xr.open_dataset(
            self.project_dir / weather_file,
            **kwargs,
        )

        if subset_time:
            start_date = self.simulation_config["DATE DEBUT"]
            end_date = self.simulation_config["DATE FIN"]
            ds = ds.sel(time=slice(start_date, end_date))

        if return_config is False:
            return ds

        else:
            cfg = (
                pd.read_csv(
                    self.project_dir / f"{weather_file}.config",
                    delimiter=";",
                    header=None,
                    index_col=0,
                )
                .replace([np.nan], [None])
                .squeeze()
                .to_dict()
            )
            # Remove leading and trailing whitespaces
            cfg = {k: v.strip() if isinstance(v, str) else v for k, v in cfg.items()}
            return ds, cfg

    def get_streamflow(self, **kwargs) -> xr.Dataset:
        r"""
        Get the streamflow from the simulation.

        Parameters
        ----------
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Returns
        -------
        xr.Dataset
            The streamflow file.
        """
        warnings.warn(
            "The 'get_streamflow' method is deprecated and will be removed in a future version. Please use the 'get_outputs' method instead.",
            FutureWarning,
            stacklevel=2,
        )
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("chunks", {})
        return xr.open_dataset(
            self.simulation_dir / "resultat" / "debit_aval.nc",
            **kwargs,
        )

    def get_outputs(self, output: str, return_paths: bool = False, **kwargs) -> xr.Dataset:
        r"""
        Get the outputs of the simulation.

        Parameters
        ----------
        output : str
            "path" to return the output directory.
            Otherwise, the name of the output to retrieve, or "q" for the streamflow.
            This should match the name of the output file without the extension (e.g. "neige" for "neige.nc").
        return_paths : bool
            If True, return the path to the output file(s) instead of the dataset. Default is False.
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Returns
        -------
        xr.Dataset
            The requested output variable.
        Path
            The path to the output directory if output is set to "path".
        list[Path]
            The path to the output file(s) if return_path is True.
        """
        outputs = self.simulation_dir / "resultat"
        if output == "path":
            return outputs
        if output == "q":
            output = "debit_aval"
        matching_files = list(Path(outputs).glob(f"{output}*.nc", case_sensitive=False))

        if return_paths:
            return matching_files
        if len(matching_files) == 0:
            raise ValueError(f"No output file matching '{output}' was found in the output directory.")
        else:
            kwargs = deepcopy(kwargs)
            kwargs.setdefault("chunks", {})
            kwargs.setdefault("combine", "by_coords")
            kwargs.setdefault("data_vars", "minimal")
            return xr.open_mfdataset(matching_files, **kwargs)

    def aggregate_outputs(  # noqa: C901
        self, by: Literal["rhhu", "subbasin"], to: Literal["subbasin", "drainage_area"], subset: list[str] | None = None, **kwargs
    ) -> None:
        r"""
        Aggregate the model outputs to a different spatial unit. See the Notes section for more details.

        Parameters
        ----------
        by : {"rhhu", "subbasin"}
            The spatial unit to aggregate from.
        to : {"subbasin", "drainage_area"}
            The spatial unit to aggregate to.
        subset : list[str] | None
            The list of variables to aggregate. If None, all variables will be processed.
            The strings should match the names produced by the HYDROTEL model.
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Notes
        -----
        This method expects that relevant spatial information has been provided to the RavenPy model, either through the initial configuration or
        through the `update_data` method. Furthermore, that spatial information should be consistent with ravenpy.extractors.BasinMakerExtractor
        expectations, as well as the Data Specifications of Basin Maker (https://hydrology.uwaterloo.ca/basinmaker/) and the outputs of
        BasinMaker's `Generate_HRUs` function. In particular, the following variables should be present in the HRU file:

        - Always:
            - SubId: The ID of the subbasins.
            - BasArea: The area of the subbasins.
        - by == "hru":
            - HRU_ID: The ID of the HRUs.
            - HRU_Area: The area of the HRUs, in units consistent with the area of the subbasins.
        - to == "drainage_area":
            - DowSubId: The ID of the downstream subbasin for each HRU.
                        This variable is standard in the RavenPy HRU files, but determining the upstream subbasins from it can be very slow.
            - UpSubId: The ID(s) of the upstream subbasin(s) for each HRU. If multiple, should be separated by a comma and no spaces (e.g. "1,2,3").
                       This variable is not standard, but if present, it will be used for a much faster aggregation to drainage areas.
        """
        if to.lower() == "subbasin" and by.lower() != "hru":
            raise ValueError("Invalid aggregation levels.")

        # clean = {
        #     "hru": "HRU",
        #     "subbasin": "Subbasin",
        #     "drainage_area": "DrainageArea",
        # }

        # # Get the files to aggregate
        # files = self.get_outputs(output=f"_By{by}", return_paths=True)
        # if subset is not None:
        #     files = [file for file in files if any(s in file.name for s in subset)]
        # if len(files) == 0:
        #     raise ValueError(f"No output files matching '{self.run_name}_*_By{clean[by]}*.nc' were found.")

        # kwargs = deepcopy(kwargs)
        # kwargs.setdefault("chunks", {})
        # for file in files:
        #     with xr.open_dataset(file, **kwargs) as ds:
        #         ds_agg = aggregate_output(ds, by=by, to=to)

        #         file_out = file.parent / file.name.replace(f"_By{clean[by]}", f"_By{clean[to]}")
        #         if file_out.exists():
        #             warnings.warn(
        #                 f"The file {file_out} already exists.",
        #                 stacklevel=2,
        #             )
        #             files_exist = list(file_out.parent.glob(file_out.stem.replace("[", "[[]").replace("]_", "[]]_") + "*.nc"))
        #             file_out = Path(str(file_out).replace(".nc", f"_v{len(files_exist) + 1}.nc"))

        #         ds_agg.to_netcdf(file_out)

    def standardize_outputs(self, files: list[str] | None = None, **kwargs):
        r"""
        Standardize the outputs of the simulation to be more consistent with CF conventions.

        Parameters
        ----------
        files : list[str] | None
            Names of the output files to standardize. If None, all output files will be standardized.
            The strings can be part of the file name (e.g. "devil_aval", "neige", "debit*", etc.).
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Notes
        -----
        Be aware that since systems such as Windows do not allow to overwrite files that are currently open,
        a temporary file will be created and then renamed to overwrite the original file.
        """
        if files is None:
            patterns = ["*.nc"]
        else:
            patterns = [f"*{file.replace('.nc', '')}*.nc" for file in files]

        files = []
        for pattern in patterns:
            files.extend(self.get_outputs(output="path").glob(pattern))

        stdout = "HYDROTEL version unspecified"
        if len(files) != 0:
            if self.rhhu is None:
                # Get the RHHU information to add relevant coordinates to the output files if possible.
                self.get_watershed_properties()

            # Get the HYDROTEL version
            if "hydrotel" in self.executable.lower() and Path(self.executable).is_file():
                stdout = subprocess.check_output(  # noqa: S603
                    [self.executable], stdin=subprocess.DEVNULL, text=True
                )

        alt_names = {
            # Dimensions
            # "basin_name": "subbasin_id",
            "idtroncon": "subbasin_id",
            "iduhrh": "rhhu_id",
            # Variables
            "debit_aval": "q",
        }

        kwargs = deepcopy(kwargs)
        kwargs.setdefault("chunks", {})
        for file in files:
            with xr.open_dataset(file, **kwargs) as ds:
                ds = standardize_output(ds, spatial_info=self.rhhu, alt_names=alt_names)

                # Adjust global attributes
                if "initial_simulation_path" in ds.attrs:
                    del ds.attrs["initial_simulation_path"]
                hydrotel_version = re.search(r"HYDROTEL \d\.\d\.\d.\d{4}", stdout)
                if hydrotel_version is not None:
                    ds.attrs["HYDROTEL_version"] = hydrotel_version.group(0).split(" ")[1]
                else:
                    ds.attrs["HYDROTEL_version"] = "unspecified"
                ds.attrs["HYDROTEL_config_version"] = self.simulation_config["SIMULATION HYDROTEL VERSION"]

                # Save the file
                ds.to_netcdf(file.parent / f"{file.stem}_tmp.nc")

            # Remove the original file and rename the new one
            file.unlink()
            (file.parent / f"{file.stem}_tmp.nc").rename(
                file,
            )

    def get_watershed_properties(self):
        """
        Retrieve the properties of the watershed from the input files and store them in the class attributes for later use.

        It is assumed that the properties of the RHHUs are created by Physitel and follow the standard HYDROTEL structure.
        See https://github.com/INRS-Modelisation-hydrologique/hydrotel/tree/main/Docs for more information on the input files.
        """
        df = pd.DataFrame(
            columns=[
                "rhhu_id",
                "subbasin_id",
                "dowsub_id",
                "drainage_area",
                "lon",
                "lat",
                "subbasin_area",
                "subbasin_elevation",
                "station_id",
                "rhhu_centroid_longitude",
                "rhhu_centroid_latitude",
                "rhhu_elevation",
                "rhhu_area",
            ]
        )

        # Get the properties of the RHHUs
        uhrh = pd.read_csv(self.project_dir / "physitel" / "uhrh.csv", delimiter=";", header=1)
        uhrh = uhrh[["UHRH ID", " ALTITUDE MOYENNE (m)", " SUPERFICIE (km2)", " LONGITUDE", " LATITUDE"]]
        uhrh.columns = ["rhhu_id", "rhhu_elevation", "rhhu_area", "rhhu_centroid_longitude", "rhhu_centroid_latitude"]
        df = pd.concat([df, uhrh], axis=0, ignore_index=True)

        # Get the properties of the subbasins
        with (self.project_dir / "physitel" / "troncon.trl").open() as f:
            data = f.readlines()
        data = [line.replace("\n", "").strip().split(" ") for line in data if len(line.replace("\n", "").strip().split(" ")) >= 5]
        for i, line in enumerate(data):
            subbasin_id = line[0]
            troncon_type = line[1]
            node_down = line[2]
            if troncon_type == "1":  # River reach
                nodes_up = [line[3]]
                nb_rhhus = line[7]
                rhhus = line[8 : 8 + int(nb_rhhus)]
            else:  # Lakes and reservoirs
                nb_nodes_up = line[3]
                nodes_up = line[4 : 4 + int(nb_nodes_up)]
                if troncon_type == "2":  # Lake
                    skip = 4
                elif troncon_type == "4":  # Lake without routing
                    skip = 0
                elif troncon_type == "5":  # Reservoir with historical outflow
                    skip = 1
                else:
                    raise ValueError(f"Unknown reach type: {troncon_type}")
                nb_rhhus = line[4 + int(nb_nodes_up) + skip]
                rhhus = line[4 + int(nb_nodes_up) + skip + 1 : 4 + int(nb_nodes_up) + skip + 1 + int(nb_rhhus)]
            data[i] = [subbasin_id, node_down, nodes_up, rhhus]

        # Get the outlet coordinates from the nodes file
        with (self.project_dir / "physitel" / "noeuds.nds").open() as f:
            data_nodes = f.readlines()
        data_nodes = pd.DataFrame(
            [line.replace("\n", "").strip().split(" ")[:3] for line in data_nodes if len(line.replace("\n", "").strip().split(" ")) >= 4],
            columns=["node_id", "longitude", "latitude"],
        )
        crs = gpd.read_file(self.project_dir / "physitel" / "rivieres.shp").crs
        gdf_nodes = gpd.GeoDataFrame(
            data_nodes, geometry=gpd.points_from_xy(data_nodes.longitude.astype(float), data_nodes.latitude.astype(float)), crs=crs
        ).to_crs(epsg=4326)

        # Get the drainage area from the troncon width and depth file
        drain = pd.read_csv(self.project_dir / "physio" / "troncon_width_depth.csv", delimiter=";", header=0)

        # Merge all subbasin information into a single dataframe
        df_sb = pd.DataFrame(data, columns=["subbasin_id", "node_down", "nodes_up", "rhhus"]).assign(
            dowsub_id="", drainage_area=np.nan, subbasin_area=np.nan, subbasin_elevation=np.nan, station_id="", lon=np.nan, lat=np.nan
        )
        for i, row in df_sb.iterrows():
            # Find 'node_down' in 'nodes_up' (which is a list per line) and get the corresponding 'subbasin_id'
            search = df_sb[df_sb["nodes_up"].apply(lambda x, row=row: row["node_down"] in x)]
            df_sb.at[i, "dowsub_id"] = search["subbasin_id"].values[0] if len(search) > 0 else "-1"

            # Add the area and elevation of the subbasin from the uhrh dataframe based on the rhhu_id
            df_sb.at[i, "subbasin_area"] = df[df["rhhu_id"].astype(str).isin(row["rhhus"])]["rhhu_area"].sum()
            df_sb.at[i, "subbasin_elevation"] = np.round(
                (
                    df[df["rhhu_id"].astype(str).isin(row["rhhus"])]["rhhu_elevation"] * df[df["rhhu_id"].astype(str).isin(row["rhhus"])]["rhhu_area"]
                ).sum()
                / df[df["rhhu_id"].astype(str).isin(row["rhhus"])]["rhhu_area"].sum(),
                6,
            )

            # Add the drainage area from the 'troncon_width_depth.csv' file based on the 'subbasin_id'
            df_sb.at[i, "drainage_area"] = drain[drain["ID"].astype(str) == row["subbasin_id"]][" Superficie [km2]"].values[0]

            # Add the coordinates of the outlet of the subbasin
            df_sb.at[i, "lon"] = gdf_nodes[gdf_nodes["node_id"] == row["node_down"]].geometry.x.values[0]
            df_sb.at[i, "lat"] = gdf_nodes[gdf_nodes["node_id"] == row["node_down"]].geometry.y.values[0]

        # Add the station_id from the stats.txt file if available
        if (self.simulation_dir / "stats.txt").is_file():
            with (self.simulation_dir / "stats.txt").open() as f:
                stats = f.readlines()
            stats = pd.DataFrame(
                [line.replace("\n", "").strip().split(" ") for line in stats if "absent" not in line], columns=["subbasin_id", "station_id"]
            )
        else:
            stats = pd.DataFrame(columns=["subbasin_id", "station_id"])
        df_sb.loc[df_sb["subbasin_id"].isin(stats["subbasin_id"]), "station_id"] = df_sb.loc[
            df_sb["subbasin_id"].isin(stats["subbasin_id"]), "subbasin_id"
        ].map(stats.set_index("subbasin_id")["station_id"])

        # Merge the subbasin information with the RHHU information
        for _, row in df_sb.iterrows():
            for col in ["subbasin_id", "dowsub_id", "lon", "lat", "drainage_area", "subbasin_area", "subbasin_elevation", "station_id"]:
                df.loc[df["rhhu_id"].astype(str).isin(row["rhhus"]), col] = row[col]

        self.rhhu = df


def _fix_os_paths(d: dict):
    """Convert paths to fit the OS. Probably not required anymore as of HYDROTEL 4.3.2, but kept in case."""
    return {k: (str(Path(PureWindowsPath(v).as_posix())) if any(slash in str(v) for slash in ["/", "\\"]) else v) for k, v in d.items()}


def _fix_dates(d: dict):
    """Convert dates to the formatting required by HYDROTEL."""
    # Reformat dates
    for key in ["DATE DEBUT", "DATE FIN"]:
        if len(d.get(key, "")) > 0:
            d[key] = pd.to_datetime(d[key]).strftime("%Y-%m-%d %H:%M")
    return d


def _read_csv(file: str | os.PathLike) -> dict:
    """
    Read a CSV file and return the content as a dictionary.

    Parameters
    ----------
    file : str or os.PathLike
        Path to the file to read.

    Returns
    -------
    dict
        Dictionary of options read from the file.

    Notes
    -----
    The CSV files aren't standard, as they contain empty lines and a variable number of separators for each line.
    Therefore, we can't use pandas or a simple dictionary to read or write the files.
    """
    with Path(file).open() as f:
        lines = f.readlines()

    # Manage cases where a semicolon might be part of the value
    lines = [line.replace(";;", ";semicolon") for line in lines]

    output = {line.split(";")[0]: line.split(";")[1] if len(line.split(";")) > 1 else None for line in lines}
    # Remove leading and trailing whitespaces
    output = {k: v.strip() if isinstance(v, str) else v for k, v in output.items()}
    # Remove newlines
    output = {k.replace("\n", ""): v.replace("\n", "") if isinstance(v, str) else v for k, v in output.items()}
    # Remove empty keys
    output = {k: v for k, v in output.items() if len(k) > 0}

    # Manage cases where a semicolon might be part of the value
    output = {k: v.replace("semicolon", ";") if isinstance(v, str) else v for k, v in output.items()}

    return output


def _overwrite_csv(file: str | os.PathLike, d: dict):
    """
    Overwrite a CSV file with new configuration options.

    Older versions of HYDROTEL are very picky about the formatting of the files and need blank lines at specific places
    so we can't use pandas or a simple dictionary to read the files.

    Parameters
    ----------
    file : str or os.PathLike
        Path to the file to write.
    d : dict
        Dictionary of options to write to the file.

    Notes
    -----
    The CSV files aren't standard, as they contain empty lines and a variable number of separators for each line.
    Therefore, we can't use pandas or a simple dictionary to read or write the files.
    """
    # Spaces and underscores are sometimes used interchangeably
    d = {k.replace(" ", "_"): v for k, v in d.items()}

    # Open the file
    with Path(file).open() as f:
        lines = f.readlines()
    lines = [line.replace(";;", ";semicolon") for line in lines]

    overwritten = []
    # clear default values from the template
    for i, line in enumerate(lines):
        if line.split(";")[0].replace(" ", "_") in d:
            overwritten.append(line.split(";")[0])
            lines[i] = f"{line.split(';')[0]};{d[line.split(';')[0].replace(' ', '_')]}\n"

    if len(overwritten) < len(d):
        raise ValueError(f"Could not find the following keys in the file on disk: {set(d.keys()) - {o.replace(' ', '_') for o in overwritten}}")
    lines = [line.replace("semicolon", ";") for line in lines]

    # Save the file
    with Path(file).open("w") as f:
        f.writelines(lines)
