"""Class to handle Hydrotel simulations."""

import os
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

__all__ = ["Hydrotel"]


class Hydrotel:
    """Class to handle Hydrotel simulations.

    Parameters
    ----------
    project_dir : str or os.PathLike
        Path to the project folder.
    project_file : str
        Name of the project file (e.g. 'projet.csv').
    project_config : dict, optional
        Dictionary of options to overwrite in the project file (projet.csv).
    simulation_config : dict, optional
        Dictionary of options to overwrite in the simulation file (simulation.csv).
    output_config : dict, optional
        Dictionary of options to overwrite in the output file (output.csv).
    use_defaults : bool
        If True, base the options on default values loaded from xhydro/modelling/data/hydrotel_defaults.yml.
        If False, read the options directly from the files in the project folder.
    """

    def __init__(
        self,
        project_dir: Union[str, os.PathLike],
        project_file: str,
        *,
        project_config: Optional[dict] = None,
        simulation_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        use_defaults: bool = True,
    ):
        """Class to handle Hydrotel simulations.

        Parameters
        ----------
        project_dir : str or os.PathLike
            Path to the project folder.
        project_file : str
            Name of the project file (e.g. 'projet.csv').
        project_config : dict, optional
            Dictionary of configuration options to overwrite in the project file.
        simulation_config : dict, optional
            Dictionary of configuration options to overwrite in the simulation file (simulation.csv).
        output_config : dict, optional
            Dictionary of configuration options to overwrite in the output file (output.csv).
        use_defaults : bool
            If True, use default configuration options loaded from xhydro/modelling/data/hydrotel_defaults/.
            If False, read the configuration options directly from the files in the project folder.

        Notes
        -----
        At minimum, the project folder must already exist when this function is called
        and either 'default_options' must be True or 'SIMULATION COURANTE' must be specified
        as a keyword argument in 'project_config'.
        """
        project_config = project_config or dict()
        simulation_config = simulation_config or dict()
        output_config = output_config or dict()

        self.project_dir = Path(project_dir)
        if not os.path.isdir(self.project_dir):
            raise ValueError("The project folder does not exist.")

        self.project_file = Path(self.project_dir / project_file).with_suffix(".csv")

        # Initialise the project, simulation, and output configuration options
        template = dict()
        # Get a basic template for the default configuration files
        for cfg in ["project", "simulation", "output"]:
            template[f"{cfg}_config"] = _read_csv(
                Path(__file__).parent / "data" / "hydrotel_defaults" / f"{cfg}.csv"
            )

        if use_defaults is True:
            o = template
            self.simulation_name = o["project_config"]["SIMULATION COURANTE"]

            # If the configuration files are missing, copy the defaults to the project folder
            if not os.path.isfile(self.project_file):
                shutil.copy(
                    Path(__file__).parent / "data" / "hydrotel_defaults" / "projet.csv",
                    self.project_file,
                )
            for cfg in ["simulation", "output"]:
                path = self.project_dir / "simulation" / self.simulation_name
                if not os.path.isfile(path / f"{cfg}.csv"):
                    shutil.copy(
                        Path(__file__).parent
                        / "data"
                        / "hydrotel_defaults"
                        / f"{cfg}.csv",
                        path / f"{cfg}.csv",
                    )

        else:
            o = dict()  # Configuration options from files on disk
            o["project_config"] = _read_csv(self.project_file)

            # Either the file on disk or the keyword argument must specify the current simulation name
            if "SIMULATION COURANTE" in project_config:
                self.simulation_name = project_config["SIMULATION COURANTE"]
            else:
                self.simulation_name = o["project_config"]["SIMULATION COURANTE"]
            if not os.path.isdir(
                self.project_dir / "simulation" / self.simulation_name
            ):
                raise ValueError(
                    f"The 'simulation/{self.simulation_name}/' folder does not exist in the project directory."
                )

            for cfg in ["simulation", "output"]:
                path = self.project_dir / "simulation" / self.simulation_name
                o[f"{cfg}_config"] = _read_csv(path / f"{cfg}.csv")

            # Check that the configuration files on disk have the same number of entries as the default
            for cfg in ["project", "simulation", "output"]:
                if len(o[f"{cfg}_config"]) != len(template[f"{cfg}_config"]):
                    raise ValueError(
                        f"The {cfg} configuration file does not appear to have the same number of entries as the default."
                    )

        # Upon initialisation, call the 'update_config' method to update the configuration options in the class,
        # and fix paths and dates if necessary
        self.project_config = o["project_config"]
        self.simulation_config = o["simulation_config"]
        self.output_config = o["output_config"]
        self.update_config(
            project_config=o["project_config"],
            simulation_config=o["simulation_config"],
            output_config=o["output_config"],
        )

        # Update options with the user-provided ones
        self.update_config(
            project_config=project_config,
            simulation_config=simulation_config,
            output_config=output_config,
        )

        # TODO: Clean up and prepare the 'etat' folder (missing the files)

    def update_config(
        self,
        *,
        project_config: Optional[dict] = None,
        simulation_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
    ):
        """Update the configuration options in the project, simulation, and output files.

        Parameters
        ----------
        project_config : dict, optional
            Dictionary of configuration options to overwrite in the project file.
        simulation_config : dict, optional
            Dictionary of configuration options to overwrite in the simulation file (simulation.csv).
        output_config : dict, optional
            Dictionary of configuration options to overwrite in the output file (output.csv).
        """
        if project_config is not None:
            project_config = deepcopy(_fix_os_paths(project_config))
            file = self.project_file
            _overwrite_csv(file, project_config)

            # Also update class attributes to reflect the changes
            for key, value in project_config.items():
                self.project_config[key] = value
            self.simulation_name = self.project_config["SIMULATION COURANTE"]

        if simulation_config is not None:
            simulation_config = deepcopy(_fix_os_paths(_fix_dates(simulation_config)))
            file = (
                self.project_dir
                / "simulation"
                / self.simulation_name
                / "simulation.csv"
            )
            _overwrite_csv(file, simulation_config)

            # Also update class attributes to reflect the changes
            for key, value in simulation_config.items():
                self.simulation_config[key] = value

        if output_config is not None:
            file = self.project_dir / "simulation" / self.simulation_name / "output.csv"
            _overwrite_csv(file, output_config)

            # Also update class attributes to reflect the changes
            for key, value in output_config.items():
                self.output_config[key] = value

    def run(
        self,
        executable: Union[str, os.PathLike] = "hydrotel",
    ):
        """
        Run the simulation.

        Parameters
        ----------
        executable : str or os.PathLike, optional
            Command to run the simulation.
            On Windows, this should be the path to Hydrotel.exe.
        """
        # Perform basic checkups on the inputs
        self._basic_checks()

        # if dry_run:
        #     command = f"{hydrotel_console} {self.project_dir / self.project_file} -t 1"
        #     return command
        # else:
        # Run the simulation
        subprocess.run(
            [str(executable), str(self.project_dir / self.project_file), "-t", "1"],
            check=True,
        )

        # Standardize the outputs
        if any(
            self.output_config[k] == 1
            for k in self.output_config
            if k not in ["TRONCONS", "DEBITS_AVAL", "OUTPUT_NETCDF"]
        ):
            warnings.warn(
                "The output options are not fully supported yet. Only 'debit_aval.nc' will be reformatted."
            )
        self._standardise_outputs()

    def get_input(
        self, return_config=False, **kwargs
    ) -> Union[xr.Dataset, tuple[xr.Dataset, dict]]:
        r"""Get the weather file from the simulation.

        Parameters
        ----------
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
        # Set the type of weather file
        if all(
            len(self.simulation_config.get(k, "")) > 0
            for k in ["FICHIER GRILLE METEO", "FICHIER STATIONS METEO"]
        ):
            raise ValueError(
                "Both 'FICHIER GRILLE METEO' and 'FICHIER STATIONS METEO' are specified in the simulation configuration file."
            )
        if len(self.simulation_config.get("FICHIER GRILLE METEO", "")) > 0:
            weather_file = self.simulation_config["FICHIER GRILLE METEO"]
        elif len(self.simulation_config.get("FICHIER STATIONS METEO", "")) > 0:
            weather_file = self.simulation_config["FICHIER STATIONS METEO"]
        else:
            raise ValueError(
                "You must specify either 'FICHIER GRILLE METEO' or 'FICHIER STATIONS METEO' in the simulation configuration file."
            )

        if return_config is False:
            return xr.open_dataset(
                self.project_dir / weather_file,
                **kwargs,
            )
        else:
            ds = xr.open_dataset(
                self.project_dir / weather_file,
                **kwargs,
            )
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
        r"""Get the streamflow from the simulation.

        Parameters
        ----------
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Returns
        -------
        xr.Dataset
            The streamflow file.
        """
        return xr.open_dataset(
            self.project_dir
            / "simulation"
            / self.simulation_name
            / "resultat"
            / "debit_aval.nc",
            **kwargs,
        )

    def _basic_checks(self, xr_open_kwargs: Optional[dict] = None):
        """Perform basic checkups on the inputs.

        Parameters
        ----------
        xr_open_kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Notes
        -----
        This function checks that:
            1. All files mentioned in the configuration exist and all expected entries are filled.
            2. The dataset has the TIME and STATION (optional) dimensions, and LONGITUDE, LATITUDE, ELEVATION coordinates.
            3. The dataset has TMIN (degC), TMAX (degC), and PRECIP (mm) variables, named as specified in the configuration.
            4. The dataset has a standard calendar.
            5. The frequency is uniform (i.e. all time steps are equally spaced).
            6. The start and end dates are contained in the dataset.

        The name of the dimensions, coordinates, and variables are checked against the configuration file.
        """
        # Check that the configuration files have no missing entries
        # FIXME: This check is now redundant with the checks in the __init__ method
        # with open(Path(__file__).parent / "data" / "hydrotel_defaults.yml") as f:
        #     defaults = yaml.safe_load(f)
        # options = ["project_options", "simulation_options", "output_options"]
        # for option in options:
        #     for key in defaults[option]:
        #         if (
        #             defaults[option][key] is not None
        #             and key not in self.__dict__[option]
        #         ):
        #             raise ValueError(
        #                 f"The option '{key}' is missing from the {option.split('_')[0]} file."
        #             )

        # Make sure that the files reflect the configuration
        self.update_config(
            project_config=self.project_config,
            simulation_config=self.simulation_config,
            output_config=self.output_config,
        )

        # FIXME: I think that Hydrotel already checks all that when it runs
        # if any(
        #     self.simulation_config.get(k, None) is None
        #     for k in ["DATE DEBUT", "DATE FIN", "PAS DE TEMPS"]
        # ):
        #     raise ValueError(
        #         "You must specify 'DATE DEBUT', 'DATE FIN', and 'PAS DE TEMPS' in the simulation configuration file."
        #     )
        #
        # # Make sure that all the files exist
        # possible_files = [
        #     self.project_config.values(),
        #     self.simulation_config.values(),
        # ]
        # for value in [item for sublist in possible_files for item in sublist]:
        #     if re.match("^.[a-z]", Path(str(value)).suffix):
        #         if Path(value).is_absolute() is False:
        #             # Some paths are relative to the project folder, others to the simulation folder
        #             if str(Path(value).parent) != ".":
        #                 value = self.project_dir / value
        #             else:
        #                 value = self.project_dir / "simulation" / self.simulation_name / value
        #         if not Path(value).is_file():
        #             raise FileNotFoundError(f"The file {value} does not exist.")

        # Open the meteo file and its configuration
        ds, cfg = self.get_input(**(xr_open_kwargs or {}), return_config=True)
        # # Validate the configuration
        # req = [
        #     "TYPE (STATION/GRID/GRID_EXTENT)",
        #     "LATITUDE_NAME",
        #     "LONGITUDE_NAME",
        #     "ELEVATION_NAME",
        #     "TIME_NAME",
        #     "TMIN_NAME",
        #     "TMAX_NAME",
        #     "PRECIP_NAME",
        # ]
        # missing = [k for k in req if cfg.get(k, None) is None]
        # if len(missing) > 0 or cfg.get("STATION_DIM_NAME", None) is None:
        #     raise ValueError(
        #         f"The configuration file is missing some entries: {missing}"
        #     )
        # if cfg["TYPE (STATION/GRID/GRID_EXTENT)"] not in ["STATION", "GRID", "GRID_EXTENT"]:
        #     raise ValueError(
        #         "The configuration file must specify 'STATION', 'GRID', or 'GRID_EXTENT' as the type of weather file."
        #     )
        # if (
        #     cfg["TYPE (STATION/GRID/GRID_EXTENT)"] != "STATION"
        #     and cfg.get("STATION_DIM_NAME", None) is not None
        # ):
        #     raise ValueError(
        #         "STATION_DIM_NAME must be specified if and only if TYPE (STATION/GRID) is 'STATION'."
        #     )

        # Check that the start and end dates are contained in the dataset
        start_date = self.simulation_config["DATE DEBUT"]
        end_date = self.simulation_config["DATE FIN"]

        # Check that the dimensions, coordinates, calendar, and units are correct
        dims = (
            [cfg["TIME_NAME"], cfg["STATION_DIM_NAME"]]
            if cfg["TYPE (STATION/GRID/GRID_EXTENT)"] == "STATION"
            else [cfg["TIME_NAME"], cfg["LATITUDE_NAME"], cfg["LONGITUDE_NAME"]]
        )
        coords = [
            cfg["TIME_NAME"],
            cfg["LATITUDE_NAME"],
            cfg["LONGITUDE_NAME"],
            cfg["ELEVATION_NAME"],
        ]
        if cfg["TYPE (STATION/GRID/GRID_EXTENT)"] == "STATION":
            coords.append(cfg["STATION_DIM_NAME"])
        structure = {
            "dims": dims,
            "coords": coords,
        }
        calendar = "standard"
        variables_and_units = {
            cfg["TMIN_NAME"]: "degC",
            cfg["TMAX_NAME"]: "degC",
            cfg["PRECIP_NAME"]: "mm",
        }

        # Check that the frequency is uniform
        freq = f"{self.simulation_config['PAS DE TEMPS']}H"
        freq = freq.replace("24H", "D")

        health_checks(
            ds,
            structure=structure,
            calendar=calendar,
            start_date=start_date,
            end_date=end_date,
            variables_and_units=variables_and_units,
            freq=freq,
            raise_on=["all"],
        )

    def _standardise_outputs(self):
        """Standardise the outputs of the simulation to be more consistent with CF conventions.

        Parameters
        ----------
        id_as_dim : bool
            Whether to use the 'station_id' coordinate as the dimension, instead of 'station'.
        xr_open_kwargs : dict, optional
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.
        """
        with self.get_streamflow() as ds:
            # station_id as dimension
            ds = ds.swap_dims({"troncon": "idtroncon"})

            # Rename variables to standard names
            ds = ds.assign_coords(idtroncon=ds["idtroncon"])
            ds = ds.rename(
                {
                    "idtroncon": "station_id",
                    "debit_aval": "streamflow",
                }
            )

            # Add standard attributes and fix units
            ds["station_id"].attrs["cf_role"] = "timeseries_id"
            orig_attrs = dict()
            orig_attrs["_original_name"] = "debit_aval"
            for attr in ["standard_name", "long_name", "description"]:
                if attr in ds["streamflow"].attrs:
                    orig_attrs[f"_original_{attr}"] = ds["streamflow"].attrs[attr]
            ds["streamflow"].attrs[
                "standard_name"
            ] = "outgoing_water_volume_transport_along_river_channel"
            ds["streamflow"].attrs["long_name"] = "Streamflow"
            ds["streamflow"].attrs[
                "description"
            ] = "Streamflow at the outlet of the river reach"
            ds["streamflow"] = xc.units.convert_units_to(ds["streamflow"], "m3 s-1")
            for attr in orig_attrs:
                ds["streamflow"].attrs[attr] = orig_attrs[attr]

            # Adjust global attributes
            del ds.attrs["initial_simulation_path"]
            ds.attrs["Hydrotel_version"] = self.simulation_config[
                "SIMULATION HYDROTEL VERSION"
            ]

            # Overwrite the file
            chunks = estimate_chunks(ds, dims=["station_id"], target_mb=5)
            save_to_netcdf(
                ds,
                self.project_dir
                / "simulation"
                / self.simulation_name
                / "resultat"
                / "debit_aval_standard.nc",
                rechunk=chunks,
                netcdf_kwargs={
                    "encoding": {
                        "streamflow": {"dtype": "float32", "zlib": True, "complevel": 1}
                    }
                },
            )

        # Remove the original file and rename the new one
        os.remove(
            self.project_dir
            / "simulation"
            / self.simulation_name
            / "resultat"
            / "debit_aval.nc"
        )
        os.rename(
            self.project_dir
            / "simulation"
            / self.simulation_name
            / "resultat"
            / "debit_aval_standard.nc",
            self.project_dir
            / "simulation"
            / self.simulation_name
            / "resultat"
            / "debit_aval.nc",
        )


def _fix_os_paths(d: dict):
    """Convert paths to fit the OS."""
    return {
        k: (
            str(Path(PureWindowsPath(v).as_posix()))
            if any(slash in str(v) for slash in ["/", "\\"])
            else v
        )
        for k, v in d.items()
    }


def _fix_dates(d: dict):
    """Convert dates to the formatting required by HYDROTEL."""
    # Reformat dates
    for key in ["DATE DEBUT", "DATE FIN"]:
        if key in d and not pd.isnull(d[key]):
            d[key] = pd.to_datetime(d[key]).strftime("%Y-%m-%d %H:%M")

    for key in [
        "LECTURE ETAT FONTE NEIGE",
        "LECTURE ETAT TEMPERATURE DU SOL",
        "LECTURE ETAT BILAN VERTICAL",
        "LECTURE ETAT RUISSELEMENT SURFACE",
        "LECTURE ETAT ACHEMINEMENT RIVIERE",
    ]:
        if len(d.get(key, "")) > 0:
            # If only a date is provided, add the path to the file
            if ".csv" not in d[key]:
                warnings.warn(
                    f"The path to the file was not provided for '{key}'. Assuming it is in the 'etat' folder."
                )
                d[key] = (
                    Path("etat")
                    / f"{'_'.join(key.split(' ')[2:]).lower()}_{d[key]}.csv"
                )
            d[key] = str(d[key]).replace(
                Path(d[key]).stem.split("_")[-1],
                pd.to_datetime(Path(d[key]).stem.split("_")[-1]).strftime("%Y%m%d%H"),
            )
    for key in [
        "ECRITURE ETAT FONTE NEIGE",
        "ECRITURE ETAT TEMPERATURE DU SOL",
        "ECRITURE ETAT BILAN VERTICAL",
        "ECRITURE ETAT RUISSELEMENT SURFACE",
        "ECRITURE ETAT ACHEMINEMENT RIVIERE",
    ]:
        if len(d.get(key, "")) > 0:
            d[key] = pd.to_datetime(d[key]).strftime("%Y-%m-%d %H")

    return d


def _read_csv(file: Union[str, os.PathLike]) -> dict:
    """Read a CSV file and return the content as a dictionary.

    Hydrotel is very picky about the formatting of the files and needs blank lines at specific places
    so we can't use pandas or a simple dictionary to read the files.

    Also, some entries in output.csv are semicolons themselves, which makes it impossible to read
    the file with pandas or other libraries.
    """
    with open(file) as f:
        lines = f.readlines()

    # Manage the case where a semicolon might be part of the value
    lines = [line.replace(";;", ";semicolon") for line in lines]

    output = {
        line.split(";")[0]: line.split(";")[1] if len(line.split(";")) > 1 else None
        for line in lines
    }
    # Remove leading and trailing whitespaces
    output = {k: v.strip() if isinstance(v, str) else v for k, v in output.items()}
    # Remove newlines
    output = {
        k.replace("\n", ""): v.replace("\n", "") if isinstance(v, str) else v
        for k, v in output.items()
    }
    # Remove empty keys
    output = {k: v for k, v in output.items() if len(k) > 0}

    # Manage the case where a semicolon might be part of the value
    output = {
        k: v.replace("semicolon", ";") if isinstance(v, str) else v
        for k, v in output.items()
    }

    return output


def _overwrite_csv(file: Union[str, os.PathLike], d: dict):
    """Overwrite a CSV file with new configuration options.

    Hydrotel is very picky about the formatting of the files and needs blank lines at specific places
    so we can't use pandas or a simple dictionary to read the files.

    Parameters
    ----------
    file : str or os.PathLike
        Path to the file to write.
    d : dict
        Dictionary of options to write to the file.
    """
    # Open the file
    with open(file) as f:
        lines = f.readlines()
    lines = [line.replace(";;", ";separateur") for line in lines]

    overwritten = []
    # clear default values from the template
    for i, line in enumerate(lines):
        if line.split(";")[0] in d:
            overwritten.append(line.split(";")[0])
            lines[i] = f"{line.split(';')[0]};{d[line.split(';')[0]]}\n"

    if len(overwritten) < len(d):
        raise ValueError(
            f"Could not find the following keys in the template file: {set(d.keys()) - set(overwritten)}"
        )
    lines = [line.replace("separateur", ";") for line in lines]

    # Save the file
    with open(file, "w") as f:
        f.writelines(lines)
