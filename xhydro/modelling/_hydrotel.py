import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path, PureWindowsPath

import pandas as pd
import xarray as xr
import xclim as xc
import yaml

from .data_checks import health_checks


class Hydrotel:
    """Class to handle Hydrotel simulations."""

    def __init__(
        self,
        project: str | os.PathLike,
        *,
        default_options: bool = True,
        simulation_options: dict = None,
        output_options: dict = None,
    ):
        """
        Class to handle Hydrotel simulations.

        Parameters
        ----------
        project: str | os.PathLike
            Path to the project folder.
        default_options: bool
            Whether to use the default simulation and output options, or read them from the project folder.
        simulation_options: dict
            Dictionary of options to change in the simulation file.
        output_options: dict
            Dictionary of options to change in the output file.
        """
        self.project = Path(project)

        if default_options is False and (
            not os.path.isfile(
                self.project / "simulation" / "simulation" / "simulation.csv"
            )
            or not os.path.isfile(
                self.project / "simulation" / "simulation" / "output.csv"
            )
        ):
            raise ValueError(
                "The simulation and output files must exist if default_options=False."
            )

        # Load from the files, if they exist. Otherwise use default values
        if (
            os.path.isfile(
                self.project / "simulation" / "simulation" / "simulation.csv"
            )
            and default_options is False
        ):
            self.simulation_options = (
                pd.read_csv(
                    self.project / "simulation" / "simulation" / "simulation.csv",
                    delimiter=";",
                    header=None,
                    index_col=0,
                )
                .squeeze()
                .to_dict()
            )
        else:
            with open(Path(__file__).parent / "hydrotel_defaults.yml") as f:
                self.simulation_options = yaml.safe_load(f)["simulation_options"]
        self.simulation_options = _fix_dates(_fix_os_paths(self.simulation_options))

        if (
            os.path.isfile(self.project / "simulation" / "simulation" / "output.csv")
            and default_options is False
        ):
            self.output_options = (
                pd.read_csv(
                    self.project / "simulation" / "simulation" / "output.csv",
                    delimiter=";",
                    header=None,
                    index_col=0,
                )
                .squeeze()
                .to_dict()
            )
        else:
            with open(Path(__file__).parent / "hydrotel_defaults.yml") as f:
                self.output_options = yaml.safe_load(f)["output_options"]

        # Update options with the user-provided ones
        self.update(
            simulation_options=simulation_options, output_options=output_options
        )

        # TODO: Clean up and prepare the 'etat' folder (missing the files)

    def update(self, *, simulation_options: dict = None, output_options: dict = None):
        """
        Update the simulation and output files.

        Parameters
        ----------
        simulation_options: dict
            Dictionary of options to change in the simulation file.
        output_options: dict
            Dictionary of options to change in the output file.
        """
        if simulation_options is not None:
            # Update options with the user-provided ones
            simulation_options = (
                deepcopy(_fix_dates(_fix_os_paths(simulation_options))) or {}
            )
            for key, value in simulation_options.items():
                self.simulation_options[key] = value

            # Save the simulation options to a file
            df = pd.DataFrame.from_dict(self.simulation_options, orient="index")
            df = df.replace({None: ""})
            df.to_csv(
                self.project / "simulation" / "simulation" / "simulation.csv",
                sep=";",
                header=False,
                columns=[0],
            )

        if output_options is not None:
            output_options = deepcopy(output_options) or {}
            for key, value in output_options.items():
                self.output_options[key] = value

            # Save the output options to a file
            df = pd.DataFrame.from_dict(output_options, orient="index")
            df.to_csv(
                self.project / "simulation" / "simulation" / "output.csv",
                sep=";",
                header=False,
            )

    def basic_checkups(self):
        """
        Perform basic checkups on the inputs.

        1. Checks that the files indicated in simulation.csv exist.
        2. Checks that the start and end dates of the simulation are contained in the meteo file.
        """
        # Make sure that all the files exist
        for key, value in self.simulation_options.items():
            if re.match("^.[a-z]", Path(str(value)).suffix):
                if Path(value).is_absolute() is False:
                    # FIXME: Why are the paths written this way?
                    if str(Path(value).parent) != ".":
                        value = self.project / value
                    else:
                        value = self.project / "simulation" / "simulation" / value
                if not Path(value).is_file():
                    raise FileNotFoundError(f"The file {value} does not exist.")

        # Make sure that the start and end dates of the simulation are contained in the meteo file
        start_date = pd.to_datetime(self.simulation_options.get("DATE DEBUT", None))
        end_date = pd.to_datetime(self.simulation_options.get("DATE FIN", None))
        weather_file = self.project / self.simulation_options.get(
            "FICHIER STATIONS METEO", None
        )
        ds = xr.open_dataset(weather_file, cache=False)
        if not ((ds.time[0] <= start_date) and (ds.time[-1] >= end_date)).item():
            raise ValueError(
                f"The start date ({start_date}) or end date ({end_date}) are outside the bounds of the weather file ({weather_file})."
            )

    def advanced_checkups(self, xr_open_kwargs: dict = None, **kwargs):
        """
        Perform more advanced checkups on the weather input file.

        Parameters
        ----------
        xr_open_kwargs: dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.
        kwargs
            Keyword arguments to pass to :py:func:`xhydro.modelling.health_checks`.
        """
        structure = {
            "dims": ["time", "stations"],
            "coords": ["lat", "lon", "x", "y", "z"],
        }
        variables_and_units = {"tasmin": "degC", "tasmax": "degC", "pr": "mm"}

        ds = xr.open_dataset(
            self.project / self.simulation_options["FICHIER STATIONS METEO"],
            **(xr_open_kwargs or {}),
        )
        health_checks(
            ds,
            structure=structure,
            calendar="standard",
            variables_and_units=variables_and_units,
            **kwargs,
        )

    def run(
        self,
        *,
        hydrotel_console: str | os.PathLike = None,
        id_as_dim: bool = True,
        return_output: bool = False,
        xr_open_kwargs: dict = None,
    ) -> xr.Dataset:
        """
        Run the simulation.

        Parameters
        ----------
        hydrotel_console: str | os.PathLike
            On Windows, path to Hydrotel.exe.
        id_as_dim: bool
            Whether to use the 'idtroncon' coordinate as a dimension.
        return_output: bool
            Whether to return the output as an xarray Dataset.
        xr_open_kwargs: dict
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.
        """
        self.basic_checkups()

        if os.name == "nt":  # Windows
            if hydrotel_console is None:
                raise ValueError("You must specify the path to Hydrotel.exe")
        else:
            hydrotel_console = "hydrotel"

        # TODO: Test it out
        # subprocess.check_call(hydrotel_console + ' ' + str(self.project) + ' -t 1')

        # Standardize the outputs
        ds = self._standardize_outputs(
            id_as_dim=id_as_dim, xr_open_kwargs=xr_open_kwargs
        )

        if return_output:
            return ds
        else:
            # TODO: Save instead of returning
            raise NotImplementedError

    def _standardize_outputs(
        self, *, id_as_dim: bool = True, xr_open_kwargs: dict = None
    ):
        """
        Standardize the outputs of the simulation.

        Parameters
        ----------
        id_as_dim: bool
            Whether to use the 'idtroncon' coordinate as a dimension.
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
        if id_as_dim:
            ds = ds.swap_dims({"station": "station_id"})
        ds = ds.drop_vars(["station"])
        ds["station_id" if id_as_dim else "station"].attrs["cf_role"] = "timeseries_id"
        # FIXME: Do we always want station_id as the dimension?

        # Add standard attributes for streamflow and fix units
        ds["streamflow"].attrs["original_name"] = "debit_aval"
        for attr in ["standard_name", "long_name", "description", "units"]:
            if attr in ds["streamflow"].attrs:
                ds["streamflow"].attrs[f"original_{attr}"] = ds["streamflow"].attrs[
                    attr
                ]
        ds["streamflow"].attrs[
            "standard_name"
        ] = "outgoing_water_volume_transport_along_river_channel"
        ds["streamflow"].attrs["long_name"] = "Streamflow"
        ds["streamflow"].attrs[
            "description"
        ] = "Streamflow at the outlet of the river reach"
        ds["streamflow"] = xc.units.convert_units_to(ds["streamflow"], "m3 s-1")

        # Add standard attributes for SWE and fix units
        # TODO: (currently missing the file)

        return ds

    def get_streamflow(self, **kwargs) -> xr.Dataset:
        """
        Get the streamflow from the simulation.

        Parameters
        ----------
        kwargs
            Keyword arguments to pass to :py:func:`xarray.open_dataset`.

        Returns
        -------
        ds: xr.Dataset
            Dataset containing the streamflow.
        """
        # FIXME: We temporarily work with a clean copy of the output, since we can't run Hydrotel
        import shutil

        if os.path.isfile(
            os.path.join(
                self.project, "simulation", "simulation", "resultat", "debit_aval.nc"
            )
        ):
            os.remove(
                os.path.join(
                    self.project,
                    "simulation",
                    "simulation",
                    "resultat",
                    "debit_aval.nc",
                )
            )
        shutil.copy(
            os.path.join(
                self.project,
                "simulation",
                "simulation",
                "resultat",
                "debit_aval_orig.nc",
            ),
            os.path.join(
                self.project, "simulation", "simulation", "resultat", "debit_aval.nc"
            ),
        )

        return xr.open_dataset(
            self.project / "simulation" / "simulation" / "resultat" / "debit_aval.nc",
            **kwargs,
        )


def _fix_os_paths(d: dict):
    """Convert paths to fit the OS."""
    # FIXME: Ugly fix to switch Windows paths to the right OS. Wouldn't work otherwise.
    return {
        k: str(Path(PureWindowsPath(v).as_posix()))
        if any(slash in str(v) for slash in ["/", "\\"])
        else v
        for k, v in d.items()
    }


def _fix_dates(d: dict):
    """Convert dates to the right format."""
    # Reformat dates
    for key in ["DATE DEBUT", "DATE FIN"]:
        if key in d and not pd.isnull(d[key]):
            d[key] = pd.to_datetime(d[key]).strftime("%Y-%m-%d %H:%M")
    # FIXME: Validate that this part is useful like this
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
            # FIXME: Do we realy need to get rid of the minutes?
            d[key] = pd.to_datetime(d[key]).strftime("%Y-%m-%d %H")

    return d
