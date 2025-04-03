"""Class to handle Hydrotel simulations."""

import os
import re
import shutil
import subprocess  # noqa: S404
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import xclim as xc
from scipy.io import netcdf_file
from xscen.io import estimate_chunks, save_to_netcdf

from xhydro.utils import health_checks

from ._hm import HydrologicalModel

__all__ = ["Hydrobuget"]


class Hydrobudget(HydrologicalModel):
    """Class to handle Hydrobudget simulations.

    Parameters
    ----------
    project_dir : str or os.PathLike
        Path to the project folder (including inputs file, shell script and R script).
    executable : str or os.PathLike
        Command to execute Hydrobudget.
        This should be the path to the shell script launching the R script.
    output_config : dict, optional
        Dictionary of configuration options to overwrite in the output file (output.csv).
    parameters : np.array or list, optional
        Parameters values for calibration.
    parameters_names : np.array or list, optional
        Parameters names for calibration.
    qobs : np.array, optional
        Observed streamflows.
    """

    def __init__(
        self,
        project_dir: str | os.PathLike,
        executable: str | os.PathLike,
        *,
        output_config: dict | None = None,
        parameters: np.ndarray | list[float] | None = None,
        parameters_names: np.ndarray | list[float] | None = None,
        qobs: np.ndarray | None,
    ):
        """Initialize the Hydrobudget simulation."""
        output_config = output_config or dict()

        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValueError("The project folder does not exist.")

        self.executable = str(Path(executable))
        self.qobs = qobs
        self.parameters = parameters
        self.parameters_names = parameters_names

    def run(
        self,
        xr_open_kwargs_out: dict | None = None,
    ) -> str | xr.Dataset:
        """Run the simulation.

        Parameters
        ----------
        xr_open_kwargs_out : dict, optional
            Keyword arguments to pass to :py:func:`xarray.open_dataset` when reading the raw output files.

        Returns
        -------
        xr.Dataset
            The streamflow file, if 'dry_run' is False.
        """
        """Preprocessing path in the bash executable."""

        new_line = 'cd "' + str(self.project_dir) + '\\"'
        new_line = new_line.replace("\\", "/")

        with Path.open(self.executable) as file:
            lines = file.readlines()

        with Path.open(self.executable, "w") as file:
            for line in lines:
                if line.startswith("cd"):
                    file.write(new_line + "\n")
                else:
                    file.write(line)

        # If parameters are given in model_config (for calibration), write .txt file that will be take int accompt by the model
        # FONCTION À ARRANGER !!!!!

        if self.parameters is not None:

            param_file = Path(self.project_dir, "param.txt")
            print(param_file)
            with Path.open(param_file) as file:
                lines = file.readlines()

            with Path.open(param_file, "w") as file:
                for line in lines:
                    if line.startswith(tuple(self.parameters_names)):
                        place_param = [
                            i
                            for i, x in enumerate(self.parameters_names)
                            if x == line.split(" ")[0]
                        ][0]
                        print(
                            self.parameters_names[place_param]
                            + " "
                            + str(self.parameters[place_param])
                        )
                        file.write(
                            self.parameters_names[place_param]
                            + " "
                            + str(self.parameters[place_param])
                            + "\n"
                        )
                    else:
                        file.write(line)

        """Run simulation."""
        subprocess.call(self.executable, shell=True)

        """Standardize the outputs"""
        self._standardise_outputs(**(xr_open_kwargs_out or {}))

        """Get streamflow """
        return self.get_streamflow()

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
        output_dir = Path(self.project_dir, "Output_Hydrobudget")
        self.output_dir = Path(output_dir)

        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            raise ValueError("The project output folder does not exist.")

        list_output_files_stations = [
            f
            for f in pathlib.Path.iterdir(self.output_dir)
            if f.endswith("_debit_sim_obs.csv")
        ]
        list_stations = [
            list_output_files_stations[i].split("_", 1)[0]
            for i in range(len(list_output_files_stations))
        ]

        # Create variables that will be in the netcdf file
        # Times
        columns_to_read = ["year", "month"]
        dates = pd.read_csv(
            Path(self.output_dir, str(list_output_files_stations[1])),
            delimiter=",",
            usecols=columns_to_read,
        )
        reference_date = datetime.strptime("1970-01-01", "%Y-%m-%d")
        times = []
        for d in range(len(dates.index)):
            time = datetime.strptime(
                datetime(dates["year"][d], dates["month"][d], 1).strftime("%Y-%m"),
                "%Y-%m",
            )
            time_month = (
                (time.year - reference_date.year) * 12
                + time.month
                - reference_date.month
            )
            times.append(time_month)

        # Qsim
        # Build variable for netcdf
        qsim_tot = np.zeros((len(list_stations), len(times)))
        qsim_base = np.zeros((len(list_stations), len(times)))
        for st in range(len(list_stations)):
            stat = list_stations[st]
            q_file = pd.read_csv(
                Path(self.output_dir, str(list_output_files_stations[st])),
                delimiter=",",
                usecols=["q", "qbase"],
            )
            qsim_tot[st] = list(q_file["q"])
            qsim_base[st] = list(q_file["qbase"])

        # Qobs
        input_dir = Path(self.project_dir, "Input")
        self.input_dir = Path(input_dir)

        qobs_input_file = [
            f
            for f in pathlib.Path.iterdir(self.input_dir)
            if f.endswith("observed_flow.csv")
        ]
        qobs_file = pd.read_csv(
            Path(self.input_dir, str(qobs_input_file[0])), delimiter=","
        )
        qobs_file["month_number"] = (
            (qobs_file["year"] - reference_date.year) * 12
            + qobs_file["month"]
            - reference_date.month
        )

        # Keep observations only if the corresponding month has been simulated
        unique = set(list(qobs_file["month_number"]))
        qobs_file_select = qobs_file[qobs_file.month_number.isin(times)]
        qobs_sum_month = qobs_file_select.groupby("month_number")[list_stations].sum()

        # Build variable for netcdf
        qobs_tot = np.zeros((len(list_stations), len(times)))
        qobs_base = np.zeros((len(list_stations), len(times)))
        for st in range(len(list_stations)):
            stat = list_stations[st]
            qobs_tot[st] = list(qobs_sum_month[stat])
            # qobs_base[st]=list(q_file['qbase'])

        # Build Netcdf file with the two variables qobs ad qsim
        # Write out data to a new netCDF file with some attributes
        netcdf_obs_path = Path(output_dir, "qobs_netcdf.nc")
        self.netcdf_obs_path = netcdf_obs_path
        filename = netcdf_file(netcdf_obs_path, "w")

        # Dimensions
        filename.createDimension("time", len(times))
        filename.createDimension("station", len(list_stations))

        # Variables
        time = filename.createVariable("time", "i", ("time",))
        station = filename.createVariable("station", "f4", ("station",))
        qobs = filename.createVariable("qobs", "f4", ("station", "time"))
        # qbase_obs = filename.createVariable('qbase_obs', 'f4', ('time', 'station'))

        # Attributes
        time.units = "months since January 1970"
        station.units = "stations names (no unit)"
        qobs.units = "mm/month"
        # qbase_obs.units = 'mm/month'

        # Populate the variables with data
        time[:] = times
        station[:] = list_stations
        qobs[:, :] = qobs_tot

        filename.close()

        # Write out data to a new netCDF file with some attributes
        netcdf_sim_path = Path(output_dir, "qsim_netcdf.nc")
        self.netcdf_sim_path = netcdf_sim_path
        filename = netcdf_file(netcdf_sim_path, "w")

        # Dimensions
        filename.createDimension("time", len(times))
        filename.createDimension("station", len(list_stations))

        # Variables
        time = filename.createVariable("time", "i", ("time",))
        station = filename.createVariable("station", "f4", ("station",))
        streamflow = filename.createVariable("streamflow", "f4", ("station", "time"))
        # qbase_sim = filename.createVariable('qbase_sim', 'f4', ('time', 'station'))

        # Attributes
        time.units = "months since January 1970"
        station.units = "stations names (no unit)"
        streamflow.units = "mm/month"
        # qbase_sim.units = 'mm/month'

        # Populate the variables with data
        time[:] = times
        station[:] = list_stations
        streamflow[:, :] = qsim_tot

        filename.close()

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
        qsim = xr.open_dataset(self.netcdf_sim_path, decode_times=False)
        # qsim = qsim["qsim"]

        return qsim
