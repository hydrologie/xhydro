# numpydoc ignore=EX01,SA01,ES01
"""Class to handle Hydrotel simulations."""

import os
import subprocess  # noqa: S404
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import netcdf_file

from ._hm import HydrologicalModel

__all__ = ["Hydrobuget"]


class Hydrobudget(HydrologicalModel):
    # numpydoc ignore=EX01,SA01,ES01
    """
    Class to handle Hydrobudget simulations.

    Parameters
    ----------
    project_dir : str or os.PathLike
        Path to the project folder (including inputs file, shell script and R script).
    executable : str or os.PathLike
        Command to execute Hydrobudget.
        This should be the path to the shell script launching the R script.
    output_config : dict, optional
        Dictionary of configuration options to overwrite in the output file (output.csv).
    simulation_config : dict, optional
        Begin and end dates of the simulation, format "%Y-%m-%d".
    parameters : np.array or list, optional
        Parameters values for calibration.
    parameters_names : np.array or list, optional
        Parameters names for calibration.
    qobs : np.array, optional
        Observed streamflows.
    """

    def __init__(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        project_dir: str | os.PathLike,
        executable: str | os.PathLike,
        *,
        output_config: dict | None = None,
        simulation_config: dict | None = None,
        parameters: np.ndarray | list[float] | None = None,
        parameters_names: np.ndarray | list[float] | None = None,
        qobs: np.ndarray | None,
    ):
        """
        Initialize the Hydrobudget simulation.

        Parameters
        ----------
        project_dir : str or os.PathLike
            Path to the project folder (including inputs file, shell script and R script).
        executable : str or os.PathLike
            Command to execute Hydrobudget.
            This should be the path to the shell script launching the R script.
        output_config : dict, optional
            Dictionary of configuration options to overwrite in the output file (output.csv).
        simulation_config : dict, optional
            Begin and end dates of the simulation, format "%Y-%m-%d".
        parameters : np.array or list, optional
            Parameters values for calibration.
        parameters_names : np.array or list, optional
            Parameters names for calibration.
        qobs : np.array, optional
            Observed streamflows.
        """
        output_config = output_config or dict()

        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValueError("The project folder does not exist.")

        self.executable = str(Path(executable))
        self.qobs = qobs or None
        self.parameters = parameters
        self.parameters_names = parameters_names
        self.simu_begin = simulation_config["DATE DEBUT"]
        self.simu_end = simulation_config["DATE FIN"]

    def run(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        xr_open_kwargs_out: dict | None = None,
    ) -> str | xr.Dataset:
        """
        Run the simulation.

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

        # If parameters are given in model_config (for calibration), write .txt file that will be take int account by the model
        param_file = Path(self.project_dir, "param.txt")
        with Path.open(param_file) as file:
            lines = file.readlines()

        with Path.open(param_file, "w") as file:
            for line in lines:
                if line.startswith("Debut"):
                    file.write(
                        "Debut"
                        + " "
                        + str(datetime.strptime(self.simu_begin, "%Y-%m-%d").year)
                        + "\n"
                    )
                elif line.startswith("Fin"):
                    file.write(
                        "Fin"
                        + " "
                        + str(datetime.strptime(self.simu_end, "%Y-%m-%d").year)
                        + "\n"
                    )
                else:
                    file.write(line)

        if self.parameters is not None:

            with Path.open(param_file, "w") as file:
                for line in lines:
                    if line.startswith(tuple(self.parameters_names)):
                        place_param = [
                            i
                            for i, x in enumerate(self.parameters_names)
                            if x == line.split(" ")[0]
                        ][0]
                        file.write(
                            self.parameters_names[place_param]
                            + " "
                            + str(self.parameters[place_param])
                            + "\n"
                        )
                    else:
                        file.write(line)

        """Run simulation."""
        # subprocess.call(self.executable, shell=True)

        """Standardize the outputs"""
        self._standardise_outputs(**(xr_open_kwargs_out or {}))

        """Get streamflow """
        return self.get_streamflow()

    def _standardise_outputs(self, **kwargs):
        # numpydoc ignore=EX01,SA01,ES01
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
        output_dir = Path(self.project_dir, "Output_Hydrobudget")
        self.output_dir = Path(output_dir)

        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            raise ValueError("The project output folder does not exist.")

        list_output_files_stations_tot = [
            str(f) for f in self.output_dir.iterdir() if f.is_file()
        ]

        list_output_files_stations = [
            file
            for file in list_output_files_stations_tot
            if file.endswith("debit_sim.csv")
        ]

        list_stations_interm = [
            list_output_files_stations[i].split("\\")[-1]
            for i in range(len(list_output_files_stations))
        ]
        list_stations = [
            list_stations_interm[i].split("_")[-3]
            for i in range(len(list_stations_interm))
        ]

        # Create variables that will be in the netcdf file
        # Times
        columns_to_read = ["year", "month"]
        dates = pd.read_csv(
            Path(self.output_dir, str(list_output_files_stations[1])),
            delimiter=",",
            usecols=columns_to_read,
        )
        times = [
            datetime(dates["year"][d], dates["month"][d], 1).strftime("%Y-%m-%d")
            for d in range(len(dates.index))
        ]

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

        list_input_files_tot = [str(f) for f in self.input_dir.iterdir() if f.is_file()]
        qobs_input_file = [
            file for file in list_input_files_tot if file.endswith("observed_flow.csv")
        ]

        qobs_file = pd.read_csv(Path(qobs_input_file[0]), delimiter=",")
        qobs_file["time"] = [
            datetime(qobs_file["year"][d], qobs_file["month"][d], 1).strftime(
                "%Y-%m-%d"
            )
            for d in range(len(qobs_file.index))
        ]

        # Keep observations only if the corresponding month has been simulated
        unique = set(list(qobs_file["time"]))
        qobs_file_select = qobs_file[qobs_file.time.isin(times)]
        qobs_sum_month = qobs_file_select.groupby("time")[list_stations].sum()

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
        time.units = "days since 1970-01-01 0:0:0"
        station.units = "stations names (no unit)"
        qobs.units = "mm/month"
        # qbase_obs.units = 'mm/month'

        # Populate the variables with data
        time_day = [
            (
                datetime.strptime(times[d], "%Y-%m-%d")
                - datetime.strptime("1970-01-01", "%Y-%m-%d")
            ).days
            for d in range(len(times))
        ]
        time[:] = time_day
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
        time.units = "days since 1970-01-01 0:0:0"
        station.units = "stations names (no unit)"
        streamflow.units = "mm/month"
        # qbase_sim.units = 'mm/month'

        # Populate the variables with data
        time[:] = time_day
        station[:] = list_stations
        streamflow[:, :] = qsim_tot

        filename.close()

    def get_streamflow(self, **kwargs) -> xr.Dataset:
        # numpydoc ignore=EX01,SA01,ES01
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
        qsim = xr.open_dataset(self.netcdf_sim_path, decode_times=False)
        # qsim = qsim["qsim"]

        return qsim

    def get_inputs(self, **kwargs) -> xr.Dataset:
        # numpydoc ignore=EX01,SA01,ES01
        r"""
        Get the input data for the hydrological model.

        Parameters
        ----------
        \*\*kwargs : dict
            Additional keyword arguments for the hydrological model.

        Returns
        -------
        xr.Dataset
            Input data for the hydrological model, in xarray Dataset format.
        """
        pass
