import os
import re
import warnings
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
import xclim as xc
import yaml
from xscen.io import estimate_chunks, save_to_netcdf

from xhydro.utils import health_checks

__all__ = ["Hydrotel"]


class Hydrotel:
    """Class to handle Hydrotel simulations."""

    def __init__(
        self,
        project: Union[str, os.PathLike],
        *,
        default_options: bool = True,
        project_options: dict = None,
        simulation_options: dict = None,
        output_options: dict = None,
    ):
        """Class to handle Hydrotel simulations.

        Parameters
        ----------
        project: str or os.PathLike
            Path to the project folder.
        default_options: bool
            If True, base the options on default values loaded from xhydro/modelling/data/hydrotel_defaults.yml.
            If False, read the options directly from the files in the project folder.
        project_options: dict
            Dictionary of options to overwrite in the project file (projet.csv).
        simulation_options: dict
            Dictionary of options to overwrite in the simulation file (simulation.csv).
        output_options: dict
            Dictionary of options to overwrite in the output file (output.csv).
        """
        project_options = project_options or dict()
        simulation_options = simulation_options or dict()
        output_options = output_options or dict()

        self.project = Path(project)
        if not os.path.isdir(self.project):
            raise ValueError("The project folder does not exist.")

        # Initialise the project, simulation, and output options
        if default_options is True:
            with open(Path(__file__).parent / "data" / "hydrotel_defaults.yml") as f:
                o = yaml.safe_load(f)
            self.simulation_name = o["project_options"]["SIMULATION COURANTE"]
        else:

            def _eval_df(df):
                # convert to int, where possible
                for row in df.itertuples():
                    try:
                        df.loc[row.Index, 1] = eval(row[1])
                    except (TypeError, ValueError):
                        pass
                return df

            o = dict()
            o["project_options"] = (
                _eval_df(
                    pd.read_csv(
                        self.project / "projet.csv",
                        delimiter=";",
                        header=None,
                        index_col=0,
                    ).replace([np.nan], [None])
                )
                .squeeze()
                .to_dict()
            )
            if "SIMULATION COURANTE" in project_options:
                self.simulation_name = project_options["SIMULATION COURANTE"]
            else:
                self.simulation_name = o["project_options"].get(
                    "SIMULATION COURANTE", None
                )
            if self.simulation_name is None:
                raise ValueError(
                    "If not using default options, 'SIMULATION COURANTE' must be specified in the project files "
                    "or as a keyword argument in 'project_options'."
                )

            for option in ["simulation", "output"]:
                path = self.project / "simulation" / self.simulation_name
                o[f"{option}_options"] = (
                    _eval_df(
                        pd.read_csv(
                            path / f"{option}.csv",
                            delimiter=";",
                            header=None,
                            index_col=0,
                        ).replace([np.nan], [None])
                    )
                    .squeeze()
                    .to_dict()
                )
        self.project_options = o["project_options"]
        self.simulation_options = o["simulation_options"]
        self.output_options = o["output_options"]

        # Fix paths and dates
        self.project_options = _fix_os_paths(self.project_options)
        self.simulation_options = _fix_dates(_fix_os_paths(self.simulation_options))

        # Update options with the user-provided ones
        self.update_options(
            project_options=project_options,
            simulation_options=simulation_options,
            output_options=output_options,
        )
        # TODO: Clean up and prepare the 'etat' folder (missing the files)

    def update_options(
        self,
        *,
        project_options: dict = None,
        simulation_options: dict = None,
        output_options: dict = None,
    ):
        """Update the options in the project, simulation, and output files.

        Parameters
        ----------
        project_options: dict
            Dictionary of options to overwrite in the project file (projet.csv).
        simulation_options: dict
            Dictionary of options to overwrite in the simulation file (simulation.csv).
        output_options: dict
            Dictionary of options to overwrite in the output file (output.csv).
        """
        options = ["project_options", "simulation_options", "output_options"]
        for option in options:
            if eval(option) is not None:
                if (
                    option == "project_options"
                    and "SIMULATION COURANTE" in project_options
                ):
                    self.simulation_name = project_options["SIMULATION COURANTE"]
                path = (
                    self.project
                    if option == "project_options"
                    else self.project / "simulation" / self.simulation_name
                )
                options_file = (
                    path / "projet.csv"
                    if option == "project_options"
                    else path / f"{option.split('_')[0]}.csv"
                )
                option_vals = deepcopy(_fix_dates(_fix_os_paths(eval(option))))
                for key, value in option_vals.items():
                    self.__dict__[option][key] = value

                # Update the options file
                df = pd.DataFrame.from_dict(self.__dict__[option], orient="index")
                df = df.replace({None: ""})
                df.to_csv(
                    options_file,
                    sep=";",
                    header=False,
                    columns=[0],
                )

    def run(
        self,
        *,
        hydrotel_console: Union[str, os.PathLike] = None,
        id_as_dim: bool = True,
        xr_open_kwargs_in: dict = None,
        xr_open_kwargs_out: dict = None,
    ):
        """
        Run the simulation.

        Parameters
        ----------
        hydrotel_console: str or os.PathLike
            For Windows only. Path to the Hydrotel.exe file.
        id_as_dim: bool
            Whether to use the 'station_id' coordinate as the dimension, instead of 'station'.
        xr_open_kwargs_in: dict
            Used on the input file. Keyword arguments to pass to :py:func:`xarray.open_dataset`.
        xr_open_kwargs_out: dict
            Used on the output file. Keyword arguments to pass to :py:func:`xarray.open_dataset`.
        """
        # Perform basic checkups on the inputs
        # FIXME: This currently fails because of bad units in the input file
        self._basic_checks(xr_open_kwargs=xr_open_kwargs_in)

        if os.name == "nt":  # Windows
            if hydrotel_console is None:
                raise ValueError("You must specify the path to Hydrotel.exe")
        else:
            hydrotel_console = "hydrotel"

        # FIXME: This needs to be tested once we have access to a working Hydrotel installation
        # subprocess.check_call(hydrotel_console + ' ' + str(self.project) + ' -t 1')

        # TODO: Check that this waits for the simulation to finish
        # Standardize the outputs
        if any(
            self.output_options[k] == 1
            for k in self.output_options
            if k not in ["TRONCONS", "DEBITS_AVAL", "OUTPUT_NETCDF"]
        ):
            warnings.warn(
                "The output options are not fully supported yet. Only 'debit_aval.nc' will be reformatted."
            )
        self._standardise_outputs(
            id_as_dim=id_as_dim,
            xr_open_kwargs=xr_open_kwargs_out,
        )

    def get_input(self, **kwargs) -> xr.Dataset:
        r"""Get the weather file from the simulation.

        Parameters
        ----------
        \*\*kwargs
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        """
        return xr.open_dataset(
            self.project / self.simulation_options["FICHIER STATIONS METEO"],
            **kwargs,
        )

    def get_streamflow(self, **kwargs) -> xr.Dataset:
        r"""Get the streamflow from the simulation.

        Parameters
        ----------
        \*\*kwargs
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        """
        return xr.open_dataset(
            self.project
            / "simulation"
            / self.simulation_name
            / "resultat"
            / "debit_aval.nc",
            **kwargs,
        )

    def _basic_checks(self, xr_open_kwargs: dict = None):
        """Perform basic checkups on the inputs.

        Parameters
        ----------
        xr_open_kwargs: dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Notes
        -----
        This function checks that:
            1. All files mentioned in the configuration exist and all expected entries are filled.
            2. The dataset has "time" and "stations" dimensions.
            3. The dataset has "lat", "lon", "x", "y", "z" coordinates.
            4. The dataset has "tasmin" (degC), "tasmax" (degC), and "pr" (mm) variables.
            5. The dataset has a standard calendar.
            6. The frequency is uniform (i.e. all time steps are equally spaced).
            7. The start and end dates are contained in the dataset.

        """
        # Check that the option files have no missing entries
        with open(Path(__file__).parent / "data" / "hydrotel_defaults.yml") as f:
            defaults = yaml.safe_load(f)
        options = ["project_options", "simulation_options", "output_options"]
        for option in options:
            for key in defaults[option]:
                if (
                    defaults[option][key] is not None
                    and key not in self.__dict__[option]
                ):
                    raise ValueError(
                        f"The option '{key}' is missing from the {option.split('_')[0]} file."
                    )

        # Make sure that all the files exist
        possible_files = [
            self.project_options.values(),
            self.simulation_options.values(),
        ]
        for value in [item for sublist in possible_files for item in sublist]:
            if re.match("^.[a-z]", Path(str(value)).suffix):
                if Path(value).is_absolute() is False:
                    # Some paths are relative to the project folder, others to the simulation folder
                    if str(Path(value).parent) != ".":
                        value = self.project / value
                    else:
                        value = self.project / "simulation" / "simulation" / value
                if not Path(value).is_file():
                    raise FileNotFoundError(f"The file {value} does not exist.")

        # Open the meteo file
        ds = self.get_input(**(xr_open_kwargs or {}))

        # Check that the start and end dates are contained in the dataset
        start_date = self.simulation_options["DATE DEBUT"]
        end_date = self.simulation_options["DATE FIN"]

        # Check that the dimensions, coordinates, calendar, and units are correct
        structure = {
            "dims": ["time", "stations"],
            "coords": ["time", "stations", "lat", "lon", "x", "y", "z"],
        }
        calendar = "standard"
        variables_and_units = {"tasmin": "degC", "tasmax": "degC", "pr": "mm"}

        # Check that the frequency is uniform
        freq = f"{self.simulation_options['PAS DE TEMPS']}H"
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

    def _standardise_outputs(
        self, *, id_as_dim: bool = True, xr_open_kwargs: dict = None
    ):
        """Standardise the outputs of the simulation to be more consistent with CF conventions.

        Parameters
        ----------
        id_as_dim: bool
            Whether to use the 'station_id' coordinate as the dimension, instead of 'station'.
        xr_open_kwargs: dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.
        """
        ds = self.get_streamflow(**(xr_open_kwargs or {}))

        # Rename variables to standard names
        ds = ds.assign_coords(troncon=ds["troncon"], idtroncon=ds["idtroncon"])
        ds = ds.rename(
            {
                "troncon": "station",
                "idtroncon": "station_id",
                "debit_aval": "streamflow",
            }
        )
        # Swap the dimensions, if requested
        if id_as_dim:
            ds = ds.swap_dims({"station": "station_id"})
            ds = ds.drop_vars(["station"])

        # Add standard attributes and fix units
        ds["station_id" if id_as_dim else "station"].attrs["cf_role"] = "timeseries_id"
        orig_attrs = dict()
        orig_attrs["original_name"] = "debit_aval"
        for attr in ["standard_name", "long_name", "description"]:
            if attr in ds["streamflow"].attrs:
                orig_attrs[f"original_{attr}"] = ds["streamflow"].attrs[attr]
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

        # TODO: Adjust global attributes once we have a working Hydrotel installation

        # Overwrite the file
        os.remove(
            self.project / "simulation" / "simulation" / "resultat" / "debit_aval.nc"
        )
        chunks = estimate_chunks(
            ds, dims=["station_id" if id_as_dim else "station"], target_mb=5
        )
        save_to_netcdf(
            ds,
            self.project / "simulation" / "simulation" / "resultat" / "debit_aval.nc",
            rechunk=chunks,
            netcdf_kwargs={
                "encoding": {
                    "streamflow": {"dtype": "float32", "zlib": True, "complevel": 1}
                }
            },
        )


def _fix_os_paths(d: dict):
    """Convert paths to fit the OS."""
    return {
        k: str(Path(PureWindowsPath(v).as_posix()))
        if any(slash in str(v) for slash in ["/", "\\"])
        else v
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
        if key in d and not pd.isnull(d[key]):
            d[key] = d[key].replace(
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
        if key in d and not pd.isnull(d[key]):
            d[key] = pd.to_datetime(d[key]).strftime("%Y-%m-%d %H")

    return d
