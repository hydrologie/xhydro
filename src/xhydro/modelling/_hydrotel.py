"""Class to handle Hydrotel simulations."""

import os
import re
import shutil
import subprocess  # noqa: S404
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

__all__ = ["Hydrotel"]


class Hydrotel(HydrologicalModel):
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
    executable : str or os.PathLike
        Command to run the simulation.
        On Windows, this should be the path to Hydrotel.exe.

    Notes
    -----
    At minimum, the project folder must already exist when this function is called
    and either 'use_defaults' must be True or 'SIMULATION COURANTE' must be specified
    as a keyword argument in 'project_config'.
    """

    def __init__(
        self,
        project_dir: str | os.PathLike,
        project_file: str,
        *,
        project_config: dict | None = None,
        simulation_config: dict | None = None,
        output_config: dict | None = None,
        use_defaults: bool = True,
        executable: str | os.PathLike = "hydrotel",
    ):
        """Initialise the Hydrotel simulation."""
        project_config = project_config or dict()
        simulation_config = simulation_config or dict()
        output_config = output_config or dict()

        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValueError("The project folder does not exist.")

        self.config_files = dict()
        self.config_files["project"] = Path(
            self.project_dir / project_file
        ).with_suffix(".csv")

        # Initialise the project, simulation, and output configuration options
        template = dict()
        # Get a basic template for the default configuration files
        for cfg in ["project", "simulation", "output"]:
            template[f"{cfg}_config"] = _read_csv(
                Path(__file__).parent / "data" / "hydrotel_defaults" / f"{cfg}.csv"
            )

        if use_defaults is True:
            o = template

            # If the keyword argument specifies the current simulation name, it will be used
            self.simulation_dir = (
                self.project_dir
                / "simulation"
                / (
                    project_config.get("SIMULATION COURANTE", None)
                    or o["project_config"]["SIMULATION COURANTE"]
                )
            )

            self.config_files["simulation"] = self.simulation_dir / "simulation.csv"
            self.config_files["output"] = self.simulation_dir / "output.csv"

            # If the configuration files are missing, copy the defaults to the project folder
            for cfg in ["project", "simulation", "output"]:
                if not Path(self.config_files[cfg]).is_file():
                    shutil.copy(
                        Path(__file__).parent
                        / "data"
                        / "hydrotel_defaults"
                        / f"{cfg}.csv",
                        self.config_files[cfg],
                    )

        else:
            o = dict()
            # Read the configuration files from disk
            o["project_config"] = _read_csv(self.config_files["project"])

            if (
                len(
                    project_config.get("SIMULATION COURANTE", None)
                    or o["project_config"]["SIMULATION COURANTE"]
                )
                == 0
            ):
                raise ValueError(
                    "'SIMULATION COURANTE' must be specified in either the project configuration file or as a keyword argument for 'project_config'."
                )

            # If the keyword argument specifies the current simulation name, it will be used
            self.simulation_dir = (
                self.project_dir
                / "simulation"
                / (
                    project_config.get("SIMULATION COURANTE", None)
                    or o["project_config"]["SIMULATION COURANTE"]
                )
            )

            if not self.simulation_dir.is_dir():
                raise ValueError(
                    f"The {self.simulation_dir} folder does not exist in the project directory."
                )

            # Read the configuration files from disk
            self.config_files["simulation"] = self.simulation_dir / "simulation.csv"
            self.config_files["output"] = self.simulation_dir / "output.csv"
            for cfg in ["simulation", "output"]:
                o[f"{cfg}_config"] = _read_csv(self.config_files[cfg])

            # Check that the configuration files on disk have the right entries
            for cfg in ["project", "simulation", "output"]:
                all_keys_in_template = all(
                    key.replace(" ", "_")
                    in (k.replace(" ", "_") for k in template[f"{cfg}_config"])
                    for key in o[f"{cfg}_config"]
                )
                if not all_keys_in_template:
                    warnings.warn(
                        f"The {cfg} configuration file on disk has some entries that might not be valid.",
                        category=UserWarning,
                    )
                nkeys_match = len(o[f"{cfg}_config"]) == len(template[f"{cfg}_config"])
                if not nkeys_match:
                    warnings.warn(
                        f"The {cfg} configuration file on disk has a different number of entries than the template.",
                        category=UserWarning,
                    )

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

    def update_config(
        self,
        *,
        project_config: dict | None = None,
        simulation_config: dict | None = None,
        output_config: dict | None = None,
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
            _overwrite_csv(self.config_files["project"], project_config)

            # Also update class attributes to reflect the changes
            for key, value in project_config.items():
                self.project_config[key] = value
            self.simulation_dir = (
                self.project_dir
                / "simulation"
                / self.project_config["SIMULATION COURANTE"]
            )
            if not self.simulation_dir.is_dir():
                raise ValueError(
                    f"The {self.simulation_dir} folder does not exist in the project directory."
                )

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
        check_missing: bool = False,
        dry_run: bool = False,
        xr_open_kwargs_in: dict | None = None,
        xr_open_kwargs_out: dict | None = None,
    ) -> str | xr.Dataset:
        """Run the simulation.

        Parameters
        ----------
        check_missing : bool
            If True, also checks for missing values in the dataset.
            This can be time-consuming for large datasets, so it is False by default. However, note that Hydrotel
            will not run if there are missing values in the input files.
        dry_run : bool
            If True, returns the command to run the simulation without actually running it.
        xr_open_kwargs_in : dict, optional
            Keyword arguments to pass to :py:func:`xarray.open_dataset` when reading the input files.
        xr_open_kwargs_out : dict, optional
            Keyword arguments to pass to :py:func:`xarray.open_dataset` when reading the raw output files.

        Returns
        -------
        str
            The command to run the simulation, if 'dry_run' is True.
        xr.Dataset
            The streamflow file, if 'dry_run' is False.
        """
        if os.name == "nt" and Path(self.executable).suffix != ".exe":
            raise ValueError("You must specify the path to Hydrotel.exe")

        # Perform basic checkups on the inputs
        self._basic_checks(check_missing=check_missing, **(xr_open_kwargs_in or {}))

        if dry_run:
            return f"{self.executable} {self.config_files['project']} -t 1"

        # Run the simulation
        subprocess.run(  # noqa: S603
            [self.executable, str(self.config_files["project"]), "-t", "1"],
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
        self._standardise_outputs(**(xr_open_kwargs_out or {}))

        return self.get_streamflow()

    def get_inputs(
        self, subset_time: bool = False, return_config=False, **kwargs
    ) -> xr.Dataset | tuple[xr.Dataset, dict]:
        r"""Get the weather file from the simulation.

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
            self.simulation_dir / "resultat" / "debit_aval.nc",
            **kwargs,
        )

    def _basic_checks(self, check_missing: bool = False, **kwargs):
        r"""Perform basic checkups on the inputs before running the simulation.

        Parameters
        ----------
        check_missing : bool
            If True, also checks for missing values in the dataset.
            This can be time-consuming for large datasets, so it is False by default. However, note that Hydrotel
            will not run if there are missing values in the input files.
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Notes
        -----
        This function checks that:
            1. All files mentioned in the configuration exist and all expected entries are filled.
            2. The meteorological dataset has the dimensions, coordinates, and variables named in its configuration file.
            3. The dataset has a standard calendar.
            4. The frequency is uniform (i.e. all time steps are equally spaced).
            5. The start and end dates are contained in the dataset.
            6. The dataset is complete (i.e. no missing values).

        The name of the dimensions, coordinates, and variables are checked against the configuration file.
        """
        # Make sure that the files reflect the configuration
        self.update_config(
            project_config=self.project_config,
            simulation_config=self.simulation_config,
            output_config=self.output_config,
        )

        if any(
            self.simulation_config.get(k, None) is None
            for k in ["DATE DEBUT", "DATE FIN", "PAS DE TEMPS"]
        ):
            raise ValueError(
                "You must specify 'DATE DEBUT', 'DATE FIN', and 'PAS DE TEMPS' in the simulation configuration file."
            )

        # Make sure that all the files exist
        possible_files = [
            self.project_config.values(),
            self.simulation_config.values(),
        ]
        for value in [item for sublist in possible_files for item in sublist]:
            if re.match("^.[a-z]", Path(str(value)).suffix):
                if Path(value).is_absolute() is False:
                    # Some paths are relative to the project folder, others to the simulation folder
                    if str(Path(value).parent) != ".":
                        value = self.project_dir / value
                    else:
                        value = self.simulation_dir / value
                if not Path(value).is_file() and Path(value).suffix != ".sth":
                    raise FileNotFoundError(
                        f"The file {value} is mentioned in the configuration, but does not exist."
                    )

        # Validate the weather file configuration vs. the weather file itself
        ds, cfg = self.get_inputs(return_config=True, **kwargs)
        req = [
            "LATITUDE_NAME",
            "LONGITUDE_NAME",
            "ELEVATION_NAME",
            "TIME_NAME",
            "TMIN_NAME",
            "TMAX_NAME",
            "PRECIP_NAME",
        ]
        gtype = [k for k in cfg if k.startswith("TYPE (")][
            0
        ]  # This entry changed in the latest version of Hydrotel
        gtype_avail = gtype.split("(")[1].split(")")[0].split("/")
        missing = [k for k in req if cfg.get(k, None) is None]
        if len(missing) > 0 or cfg.get("STATION_DIM_NAME", None) is None:
            raise ValueError(
                f"The configuration file is missing some entries: {missing}"
            )
        if cfg[gtype] not in gtype_avail:
            raise ValueError(
                f"The configuration file must specify the type of data as one of {gtype_avail}."
            )
        if cfg[gtype] != "STATION" and cfg.get("STATION_DIM_NAME", None) is not None:
            raise ValueError(
                f"STATION_DIM_NAME must be specified if and only if {gtype} is 'STATION'."
            )

        # Check that the start and end dates are contained in the dataset
        start_date = self.simulation_config["DATE DEBUT"]
        end_date = self.simulation_config["DATE FIN"]

        # Check that the dimensions, coordinates, calendar, and units are correct
        dims = (
            [cfg["TIME_NAME"], cfg["STATION_DIM_NAME"]]
            if cfg[gtype] == "STATION"
            else [cfg["TIME_NAME"], cfg["LATITUDE_NAME"], cfg["LONGITUDE_NAME"]]
        )
        coords = [
            cfg["TIME_NAME"],
            cfg["LATITUDE_NAME"],
            cfg["LONGITUDE_NAME"],
            cfg["ELEVATION_NAME"],
        ]
        if cfg[gtype] == "STATION":
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

        # Check that the dataset is complete
        missing = "missing_any" if check_missing else None

        # Fix badly formatted files that xclim can't handle
        if ds[f"{cfg['TMIN_NAME']}"].attrs["units"] == "DEGC":
            ds[f"{cfg['TMIN_NAME']}"].attrs["units"] = "degC"
        if ds[f"{cfg['TMAX_NAME']}"].attrs["units"] == "DEGC":
            ds[f"{cfg['TMAX_NAME']}"].attrs["units"] = "degC"

        health_checks(
            ds,
            structure=structure,
            calendar=calendar,
            start_date=start_date,
            end_date=end_date,
            variables_and_units=variables_and_units,
            freq=freq,
            missing=missing,
            raise_on=None,
        )

    def _standardise_outputs(self, **kwargs):
        r"""Standardise the outputs of the simulation to be more consistent with CF conventions.

        Parameters
        ----------
        \*\*kwargs : dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Notes
        -----
        Be aware that since systems such as Windows do not allow to overwrite files that are currently open,
        a temporary file will be created and then renamed to overwrite the original file.
        """
        with self.get_streamflow(**kwargs) as ds:
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
            if "initial_simulation_path" in ds.attrs:
                del ds.attrs["initial_simulation_path"]
            ds.attrs["Hydrotel_version"] = self.simulation_config[
                "SIMULATION HYDROTEL VERSION"
            ]

            # Overwrite the file
            chunks = estimate_chunks(ds, dims=["station_id"], target_mb=5)
            save_to_netcdf(
                ds,
                self.simulation_dir / "resultat" / "debit_aval_tmp.nc",
                rechunk=chunks,
                netcdf_kwargs={
                    "encoding": {
                        "streamflow": {"dtype": "float32", "zlib": True, "complevel": 1}
                    }
                },
            )

        # Remove the original file and rename the new one
        Path(self.simulation_dir / "resultat" / "debit_aval.nc").unlink()
        Path(self.simulation_dir / "resultat" / "debit_aval_tmp.nc").rename(
            self.simulation_dir / "resultat" / "debit_aval.nc",
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
        if len(d.get(key, "")) > 0:
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


def _read_csv(file: str | os.PathLike) -> dict:
    """Read a CSV file and return the content as a dictionary.

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
    Hydrotel is very picky about the formatting of the files and needs blank lines at specific places
    so we can't use pandas or a simple dictionary to read the files.

    Also, some entries in output.csv are semicolons themselves, which makes it impossible to read
    the file with pandas or other libraries.
    """
    with Path(file).open() as f:
        lines = f.readlines()

    # Manage cases where a semicolon might be part of the value
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

    # Manage cases where a semicolon might be part of the value
    output = {
        k: v.replace("semicolon", ";") if isinstance(v, str) else v
        for k, v in output.items()
    }

    return output


def _overwrite_csv(file: str | os.PathLike, d: dict):
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
            lines[i] = (
                f"{line.split(';')[0]};{d[line.split(';')[0].replace(' ', '_')]}\n"
            )

    if len(overwritten) < len(d):
        raise ValueError(
            f"Could not find the following keys in the template file: {set(d.keys()) - set(overwritten)}"
        )
    lines = [line.replace("semicolon", ";") for line in lines]

    # Save the file
    with Path(file).open("w") as f:
        f.writelines(lines)
