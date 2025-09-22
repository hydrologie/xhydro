"""Class to handle Hydrotel simulations."""

import itertools
import os
import re
import subprocess  # noqa: S404
import warnings
from copy import deepcopy
from pathlib import Path, PureWindowsPath

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
    project_dir : str or Path
        Path to the project folder.
    project_file : str
        Name of the project file (e.g. 'projet.csv').
    executable : str or Path
        Command to execute Hydrotel.
        On Windows, this should be the path to Hydrotel.exe.
    project_config : dict, optional
        Dictionary of configuration options to overwrite in the project file.
    simulation_config : dict, optional
        Dictionary of configuration options to overwrite in the simulation file. See the Notes section for more details.
    output_config : dict, optional
        Dictionary of configuration options to overwrite in the output file (output.csv).
    use_defaults : bool
        Deprecated, as it caused confusion. Consult the DemoProject in https://github.com/INRS-Modelisation-hydrologique/hydrotel
        to get an idea of the configuration options to use.

    Notes
    -----
    The name of the simulation file must match the name of the 'SIMULATION COURANTE' option in the project file.

    This class is designed to handle the execution of Hydrotel simulations, with the ability to overwrite configuration options,
    but it does not handle the creation of the project folder itself. The project folder must be created beforehand.

    For more information on how to configure the project, refer to the documentation of Hydrotel:
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
        use_defaults=None,
    ):
        if use_defaults is not None:
            warnings.warn(
                "The 'use_defaults' parameter is deprecated and will be ignored. It will be removed in xHydro v0.7.0. "
                "Please refer to the DemoProject in https://github.com/INRS-Modelisation-hydrologique/hydrotel "
                "to get an idea of the configuration options to use.",
                FutureWarning,
            )

        """Initialize the Hydrotel simulation."""
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

        # Initialize the project, simulation, and output configuration options
        o = dict()
        # Read the configuration files from disk
        o["project_config"] = _read_csv(self.config_files["project"])

        # Get the simulation name
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
        sim_name = (
            project_config.get("SIMULATION COURANTE", None)
            or o["project_config"]["SIMULATION COURANTE"]
        )
        self.simulation_dir = self.project_dir / "simulation" / sim_name

        if not self.simulation_dir.is_dir():
            raise ValueError(
                f"The {self.simulation_dir} folder does not exist in the project directory."
            )

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
            self.simulation_dir = (
                self.project_dir
                / "simulation"
                / self.project_config["SIMULATION COURANTE"]
            )
            self.config_files["simulation"] = (
                self.simulation_dir
                / f"{self.project_config['SIMULATION COURANTE']}.csv"
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
        *,
        run_options: list[str] | None = None,
        dry_run: bool = False,
        xr_open_kwargs_out: dict | None = None,
        check_missing=None,
        xr_open_kwargs_in=None,
    ) -> str | xr.Dataset:
        """Run the simulation.

        Parameters
        ----------
        run_options : list[str] | None
            Additional options to pass to the Hydrotel executable.
            Common arguments include:
            - `-t NUM`: Run the simulation using a given number of threads (default is 1).
            - `-c`: Skip the validation of the input files.
            - `-s`: Skip the interpolation of missing values in the input files. Only use this if you are sure that the input files are complete.
            Call the executable without arguments to see the full list of available options.
        dry_run : bool
            If True, returns the command to run the simulation without actually running it.
        xr_open_kwargs_out : dict, optional
            Keyword arguments to pass to :py:func:`xarray.open_dataset` when reading the raw output files.
        check_missing : None
            Deprecated, as it was redundant with checkups performed by the Hydrotel executable.
        xr_open_kwargs_in : None
            Deprecated, as it is not used anymore.

        Returns
        -------
        str
            The command to run the simulation, if 'dry_run' is True.
        xr.Dataset
            The streamflow file, if 'dry_run' is False.
        """
        if check_missing is not None:
            warnings.warn(
                "The 'check_missing' parameter is deprecated and will be ignored. "
                "The Hydrotel executable already performs checks on the input files."
                "This parameter will be removed in xHydro v0.7.0.",
                FutureWarning,
            )
        if xr_open_kwargs_in is not None:
            warnings.warn(
                "The 'xr_open_kwargs_in' parameter is deprecated and will be ignored. "
                "It is not used anymore and will be removed in xHydro v0.7.0.",
                FutureWarning,
            )

        if os.name == "nt" and Path(self.executable).suffix != ".exe":
            raise ValueError("You must specify the path to Hydrotel.exe")
        if "hydrotel" not in self.executable.lower():
            raise ValueError(
                "The executable command does not seem to be a valid Hydrotel command. "
                "Please check the 'executable' parameter."
            )

        # Make sure that the files reflect the configuration
        self.update_config(
            project_config=self.project_config,
            simulation_config=self.simulation_config,
            output_config=self.output_config,
        )

        # Prepare the input call
        run_options = run_options or []
        # Unwrap elements that contain spaces
        run_options = list(
            itertools.chain.from_iterable(
                [a.split() if isinstance(a, str) else a for a in run_options]
            )
        )
        run_options = [*run_options, "-t 1"] if "-t" not in run_options else run_options
        call = [self.executable, str(self.config_files["project"])] + run_options

        if dry_run:
            return " ".join(call)

        # Run the simulation
        subprocess.run(  # noqa: S603
            call,
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
            # subbasin_id as dimension
            ds = ds.swap_dims({"troncon": "idtroncon"})

            # Rename variables to standard names
            ds = ds.assign_coords(idtroncon=ds["idtroncon"])
            ds = ds.rename(
                {
                    "idtroncon": "subbasin_id",
                    "debit_aval": "q",
                }
            )

            # Add standard attributes and fix units
            ds["subbasin_id"].attrs["cf_role"] = "timeseries_id"
            orig_attrs = dict()
            orig_attrs["_original_name"] = "debit_aval"
            for attr in ["standard_name", "long_name", "description"]:
                if attr in ds["q"].attrs:
                    orig_attrs[f"_original_{attr}"] = ds["q"].attrs[attr]
            ds["q"].attrs[
                "standard_name"
            ] = "outgoing_water_volume_transport_along_river_channel"
            ds["q"].attrs["long_name"] = "Streamflow"
            ds["q"].attrs["description"] = "Streamflow at the outlet of the river reach"
            ds["q"] = xc.units.convert_units_to(ds["q"], "m3 s-1")
            for attr in orig_attrs:
                ds["q"].attrs[attr] = orig_attrs[attr]

            # Adjust global attributes
            if "initial_simulation_path" in ds.attrs:
                del ds.attrs["initial_simulation_path"]
            if (
                "hydrotel" not in self.executable.lower()
                or not Path(self.executable).is_file()
            ):
                warnings.warn(
                    "The executable command is suspicious and will not be executed.",
                    UserWarning,
                )
                stdout = "HYDROTEL version unspecified"
            else:
                stdout = subprocess.check_output(  # noqa: S603
                    [self.executable],
                )
            ds.attrs["Hydrotel_version"] = (
                str(stdout).split("HYDROTEL ")[1].split("\\n")[0]
            )
            ds.attrs["Hydrotel_config_version"] = self.simulation_config[
                "SIMULATION HYDROTEL VERSION"
            ]

            # Overwrite the file
            chunks = estimate_chunks(ds, dims=["subbasin_id"], target_mb=5)
            save_to_netcdf(
                ds,
                self.simulation_dir / "resultat" / "debit_aval_tmp.nc",
                rechunk=chunks,
                netcdf_kwargs={
                    "encoding": {
                        "q": {"dtype": "float32", "zlib": True, "complevel": 1}
                    }
                },
            )

        # Remove the original file and rename the new one
        Path(self.simulation_dir / "resultat" / "debit_aval.nc").unlink()
        Path(self.simulation_dir / "resultat" / "debit_aval_tmp.nc").rename(
            self.simulation_dir / "resultat" / "debit_aval.nc",
        )


def _fix_os_paths(d: dict):
    """Convert paths to fit the OS. Probably not required anymore as of Hydrotel 4.3.2, but kept in case."""
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
    The CSV files aren't standard, as they contain empty lines and a variable number of separators for each line.
    Therefore, we can't use pandas or a simple dictionary to read or write the files.
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
            lines[i] = (
                f"{line.split(';')[0]};{d[line.split(';')[0].replace(' ', '_')]}\n"
            )

    if len(overwritten) < len(d):
        raise ValueError(
            f"Could not find the following keys in the file on disk: {set(d.keys()) - {o.replace(' ', '_') for o in overwritten}}"
        )
    lines = [line.replace("semicolon", ";") for line in lines]

    # Save the file
    with Path(file).open("w") as f:
        f.writelines(lines)
