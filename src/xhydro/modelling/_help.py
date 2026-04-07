# numpydoc ignore=EX01,SA01,ES01
"""Class to handle Hydrotel simulations."""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr
from matplotlib.dates import DateFormatter
from pyhelp.managers import HelpManager
from rasterio.transform import from_origin
from scipy.io import netcdf_file

from ._hm import HydrologicalModel


__all__ = ["HELP"]


class HELP(HydrologicalModel):
    # numpydoc ignore=EX01,SA01,ES01
    """
    Class to handle HELP simulations.

    Parameters
    ----------
    project_dir : str or os.PathLike
        Path to the project folder (including inputs file, shell script and R script).
    gauging_cells : str
        Name of the file (in project_dir) describing which cells are contained in which gauging stations.
    simulation_config : dict, optional
        Begin and end dates of the simulation, format "%Y-%m-%d".
    parameters_base : np.array or list, optional
        Parameters values for modifying default parameters.
    parameters_base_names : np.array or list, optional
        Parameters names for for modifying default parameters.
    parameters : np.array or list, optional
        Parameters values for calibration.
    parameters_names : np.array or list, optional
        Parameters names for calibration.
    qobs : np.array, optional
        Observed streamflows.
    start_date : str, optional
        Calibration start date.
    end_date : str, optional
        Calibration end date.
    frequency : str, optional
        Frequency of output data : month, season, year, month&year, all. If None, default is season.
    graph_outputs : int, optional
        Level of diversity in graph outputs : 0 = no graph, 1 = 1:1 obs sim graph, 2 = all graphs.
    """

    def __init__(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        project_dir: str | os.PathLike,
        gauging_cells: str,
        simulation_config: dict | None = None,
        parameters_base: np.ndarray | list[float] | None = None,
        parameters_base_names: np.ndarray | list[float] | None = None,
        parameters: np.ndarray | list[float] | None = None,
        parameters_names: np.ndarray | list[float] | None = None,
        qobs: np.ndarray | xr.Dataset | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str | None = None,
        graph_outputs: int | None = 0,
    ):
        """
        Initialize the HELP simulation.

        Parameters
        ----------
        project_dir : str or os.PathLike
            Path to the project folder (including inputs file, shell script and R script).
        gauging_cells : str
            Name of the file (in project_dir) describing which cells are contained in which gauging stations.
        simulation_config : dict, optional
            Begin and end dates of the simulation, format "%Y-%m-%d".
        parameters_base : np.array or list, optional
            Parameters values for modifying default parameters.
        parameters_base_names : np.array or list, optional
            Parameters names for for modifying default parameters.
        parameters : np.array or list, optional
            Parameters values for calibration.
        parameters_names : np.array or list, optional
            Parameters names for calibration.
        qobs : np.array, optional
            Observed streamflows.
        start_date : str, optional
            Calibration start date.
        end_date : str, optional
            Calibration end date.
        frequency : str, optional
            Frequency of output data : month, season, year, month&year, all. If None, default is season.
        graph_outputs : int, optional
            Level of diversity in graph outputs : 0 = no graph, 1 = 1:1 obs sim graph, 2 = all graphs.
        """
        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValueError("The project folder does not exist.")

        self.path_gauging_cells = self.project_dir / gauging_cells
        self.qobs = qobs
        self.simu_begin = simulation_config["DATE DEBUT"]
        self.simu_end = simulation_config["DATE FIN"]
        self.cal_start_date = start_date
        self.cal_end_date = end_date
        self.frequency = frequency
        self.graph_outputs = graph_outputs

        if parameters_base is None and parameters is None:
            parameters_base = [1, 1, 1, 0, 1]
            parameters_base_names = ["sf_edepth", "sf_ulai", "sf_cn", "tfsoil", "sfrad"]
            param_base_dict = dict(zip(parameters_base_names, parameters_base, strict=True))
            self.param_dict = param_base_dict

        if parameters_base is not None and parameters is None:
            param_base_dict = dict(zip(parameters_base_names, parameters_base, strict=True))
            self.param_dict = param_base_dict

        if parameters_base is None and parameters is not None:
            parameters_base = [1, 1, 1, 0, 1]
            parameters_base_names = ["sf_edepth", "sf_ulai", "sf_cn", "tfsoil", "sfrad"]
            param_base_dict = dict(zip(parameters_base_names, parameters_base, strict=True))
            param_dict = dict(zip(parameters_names, parameters, strict=True))
            # Fusion of the two dict, with priority to calibration parameters when present
            self.param_dict = param_base_dict | param_dict

        if parameters_base is not None and parameters is not None:
            param_base_dict = dict(zip(parameters_base_names, parameters_base, strict=True))
            param_dict = dict(zip(parameters_names, parameters, strict=True))
            # Fusion of the two dict, with priority to calibration parameters when present
            self.param_dict = param_base_dict | param_dict

        self.param_base_grid = [elem for elem in self.param_dict if elem not in ["sf_edepth", "sf_ulai", "sf_cn", "tfsoil", "sfrad"]]
        if len(self.param_base_grid) > 0:
            self.modify_input_file()

        # Enregistrement des jeux de param generes dans un fichier csv
        file_exists = Path(self.project_dir / "param_.csv").exists()
        with Path(self.project_dir / "param_.csv").open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.param_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.param_dict)

        self.start_year = str(datetime.strptime(self.simu_begin, "%Y-%m-%d").year)

        return

    def get_inputs(
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
        p_to_grid = self.project_dir / "input_grid.csv"
        p_to_precip = self.project_dir / "precip_input_data.csv"
        p_to_airtemp = self.project_dir / "airtemp_input_data.csv"
        p_to_solrad = self.project_dir / "solrad_input_data.csv"

        return p_to_grid, p_to_precip, p_to_airtemp, p_to_solrad

    def modify_input_file(
        # numpydoc ignore=EX01,SA01,ES01
        self,
    ):
        """Modify HELP input files with new parameters."""
        df_rad = pd.read_csv(self.project_dir / "solrad_input_data_origin.csv", sep=",", decimal=".", header=None)
        firstlignes = df_rad.iloc[0:2].copy()
        rest = df_rad.iloc[2:].copy()
        colonnesmod = rest.columns[1:]
        rest[colonnesmod] = rest[colonnesmod] * self.param_dict["sfrad"]
        df_final_rad = pd.concat([firstlignes, rest], ignore_index=True)
        df_final_rad.to_csv(self.project_dir / "solrad_input_data.csv", sep=",", decimal=".", index=False, header=None, encoding="utf-8")
        self.p_to_solrad = self.project_dir / "solrad_input_data.csv"

        grid_data = pd.read_csv(self.project_dir / "input_grid.csv", delimiter=",", dtype={"cell_ID": int, "gauging_stat": str})

        grid_data["thick1"][grid_data["thick1"] < 100] = 100
        grid_data["thick1"][grid_data["context"] == 0] = 0
        grid_data["run"][grid_data["context"] == 0] = 0
        # grid_data["slope1"] = grid_data["slope1"]/10
        # grid_data["slope2"] = grid_data["slope2"]/10
        # grid_data["slope3"] = grid_data["slope3"]/10
        grid_data["CN"][grid_data["geol"] == 5] = 95
        grid_data["LAI"][grid_data["LAI"] == 1] = 3
        grid_data["LAI"][grid_data["LAI"] == 0] = 3
        grid_data["LAI"][grid_data["context"] == 0] = 0

        self.param_grid = [elem for elem in list(self.param_dict.keys()) if elem not in ["sf_edepth", "sf_ulai", "sf_cn", "tfsoil"]]

        param_ksat = [m for m in self.param_grid if m.startswith("ksat")]
        param_poro = [m for m in self.param_grid if m.startswith("poro")]
        param_fc = [m for m in self.param_grid if m.startswith("fc")]
        param_difffcwp = [m for m in self.param_grid if m.startswith("diff")]

        if len(param_ksat) > 0:
            g = [m[-1] for m in param_ksat]
            for ge in range(len(g)):
                grid_data["ksat1"][grid_data["geol"] == ge + 1] = self.param_dict[param_ksat[ge]]
                # activate if 3 layers
                grid_data["ksat2"][grid_data["geol"] == ge + 1] = self.param_dict[param_ksat[ge]]
                grid_data["ksat3"][grid_data["geol"] == ge + 1] = self.param_dict[param_ksat[ge]]

        if len(param_poro) > 0:
            g = [m[-1] for m in param_poro]
            for ge in range(len(g)):
                grid_data["poro1"][grid_data["geol"] == ge + 1] = self.param_dict[param_poro[ge]]
                # activate if 3 layers
                grid_data["poro2"][grid_data["geol"] == ge + 1] = self.param_dict[param_poro[ge]]
                grid_data["poro3"][grid_data["geol"] == ge + 1] = self.param_dict[param_poro[ge]]

        if len(param_fc) > 0:
            g = [m[-1] for m in param_fc]
            for ge in range(len(g)):
                grid_data["fc1"][grid_data["geol"] == ge + 1] = self.param_dict[param_fc[ge]]
                # activate if 3 layers
                grid_data["fc2"][grid_data["geol"] == ge + 1] = self.param_dict[param_fc[ge]]
                grid_data["fc3"][grid_data["geol"] == ge + 1] = self.param_dict[param_fc[ge]]

        if len(param_difffcwp) > 0:
            g = [m[-1] for m in param_difffcwp]
            for ge in range(len(g)):
                grid_data["wp1"][grid_data["geol"] == ge + 1] = self.param_dict[param_fc[ge]] - self.param_dict[param_difffcwp[ge]]
                # activate if 3 layers
                grid_data["wp2"][grid_data["geol"] == ge + 1] = self.param_dict[param_fc[ge]] - self.param_dict[param_difffcwp[ge]]
                grid_data["wp3"][grid_data["geol"] == ge + 1] = self.param_dict[param_fc[ge]] - self.param_dict[param_difffcwp[ge]]

        grid_data.to_csv(self.project_dir / "input_grid.csv", index=False, header=True, sep=",")

        return

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
        p_to_grid, p_to_precip, p_to_airtemp, p_to_solrad = self.get_inputs(**(xr_open_kwargs_out or {}))

        helpm = HelpManager(
            self.project_dir,
            path_to_grid=p_to_grid,
            path_to_precip=p_to_precip,
            path_to_airtemp=p_to_airtemp,
            path_to_solrad=p_to_solrad,
        )

        cellnames = helpm.grid.index[helpm.grid["run"] == 1]

        output_help = helpm.calc_help_cells(
            path_to_hdf5=self.project_dir / "help_outputhdf5.out",
            cellnames=cellnames,
            tfsoil=self.param_dict["tfsoil"],
            sf_edepth=self.param_dict["sf_edepth"],
            sf_ulai=self.param_dict["sf_ulai"],
            sf_cn=self.param_dict["sf_cn"],
        )
        self.output_help = output_help

        output_help.save_to_csv(
            self.project_dir / "Results" / "gwr_annuel_spat.csv",
            datetime.strptime(self.simu_begin, "%Y-%m-%d").year + 1,
            datetime.strptime(self.simu_end, "%Y-%m-%d").year,
        )

        varnames = ["precip", "rechg", "runoff", "evapo", "subrun1", "subrun2", "perco"]

        gaug_cells = pd.read_csv(self.path_gauging_cells, delimiter=",", dtype={"cell_ID": int, "gauging_stat": str})

        list_stations = list(set(gaug_cells["gauging_stat"]))

        stat_cells = {}
        for stat in list_stations:
            subset = gaug_cells[gaug_cells["gauging_stat"] == stat]
            stat_cells[stat] = list(subset["cell_ID"])

        hdf5 = h5py.File(self.project_dir / "help_outputhdf5.out", mode="r+")
        try:
            # Load the data.
            self.data = {}
            for key in list(hdf5["data"].keys()):
                values = np.array(hdf5["data"][key])
                if key == "cid":
                    values = values.astype(str)
                self.data[key] = values
        finally:
            hdf5.close()

        for stat in list_stations:
            list_cells_stat = stat_cells[stat]
            cell_rep = np.where(np.isin(self.data["cid"], list_cells_stat))[0]

            data_month = pd.DataFrame(
                columns=["year", "month", "trim", "station", "precip", "runoff", "evapo", "rechg", "subrun1", "subrun2", "perco"]
            )

            years_abs = [int(self.start_year) + a for a in range(self.data["evapo"].shape[1])]

            data_month["year"] = [i for i in years_abs for _ in range(12)]
            data_month["month"] = [m + 1 for m in range(12)] * len(years_abs)
            data_month["trim"][np.isin(data_month["month"], [12, 1, 2])] = 1
            data_month["trim"][np.isin(data_month["month"], [3, 4, 5])] = 2
            data_month["trim"][np.isin(data_month["month"], [6, 7, 8])] = 3
            data_month["trim"][np.isin(data_month["month"], [9, 10, 11])] = 4
            data_month["station"] = stat

            for varname in varnames:
                var = []
                var_mat = np.nansum(self.data[varname][cell_rep], axis=0) / len(list_cells_stat)
                for y in range(self.data[varname].shape[1]):
                    var.extend(var_mat[y])
                    # for m in range(self.data[varname].shape[2]):
                    #    var.append(var_mat[y][m])
                data_month[varname] = var

            data_month.columns = ["year", "month", "trim", "station", "precip", "runoff", "aet", "gwr", "subrun1", "subrun2", "perco"]
            data_month = data_month.rename(columns={"rechg": "gwr"})

            # enregistrement en csv
            output_dir = self.project_dir / "Results"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir
            file_path = self.project_dir / f"Results/{stat}_monthly.csv"
            data_month.to_csv(file_path, sep=",", index=False, header=True, encoding="utf-8")

        # """Standardize the outputs"""
        self._standardise_outputs(**(xr_open_kwargs_out or {}))

        """Get streamflow """
        return self.get_streamflow()

    def get_streamflow(
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
        qsim = xr.open_dataset(self.netcdf_sim_path, decode_times=False)

        return qsim

    def _standardise_outputs(self, **kwargs):  # noqa: C901
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
        frequency = self.frequency
        output_dir = Path(self.project_dir, "Results")
        self.output_dir = Path(output_dir)

        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            raise ValueError("The project output folder does not exist.")

        # Create variables that will be in the netcdf file
        # Qobs et Qsim

        obs_qtot = pd.read_csv(Path(self.output_dir, "observed_flow.csv"))
        obs_qbase = pd.read_csv(Path(self.output_dir, "eckhardt_baseflow.csv"))
        stations = obs_qtot.keys()[3:]

        obs_qtot.loc[obs_qtot.months.isin([12, 1, 2]), "trim"] = "1"
        obs_qtot.loc[obs_qtot.months.isin([3, 4, 5]), "trim"] = "2"
        obs_qtot.loc[obs_qtot.months.isin([6, 7, 8]), "trim"] = "3"
        obs_qtot.loc[obs_qtot.months.isin([9, 10, 11]), "trim"] = "4"

        obs_qbase.loc[obs_qbase.months.isin([12, 1, 2]), "trim"] = "1"
        obs_qbase.loc[obs_qbase.months.isin([3, 4, 5]), "trim"] = "2"
        obs_qbase.loc[obs_qbase.months.isin([6, 7, 8]), "trim"] = "3"
        obs_qbase.loc[obs_qbase.months.isin([9, 10, 11]), "trim"] = "4"

        obs_qtot["trim_year"] = " "
        obs_qtot["trim_year"] = [obs_qtot["trim"][i] + "_" + str(obs_qtot["year"][i]) for i in range(len(obs_qtot["trim_year"]))]
        obs_qbase["trim_year"] = " "
        obs_qbase["trim_year"] = [obs_qbase["trim"][i] + "_" + str(obs_qbase["year"][i]) for i in range(len(obs_qbase["trim_year"]))]

        obs_qtot["month_year"] = " "
        obs_qtot["month_year"] = [str(obs_qtot["months"][i]) + "_" + str(obs_qtot["year"][i]) for i in range(len(obs_qtot["trim_year"]))]
        obs_qbase["month_year"] = " "
        obs_qbase["month_year"] = [str(obs_qbase["months"][i]) + "_" + str(obs_qbase["year"][i]) for i in range(len(obs_qbase["trim_year"]))]

        qtotsim_totalite = []
        qbasesim_totalite = []
        qtotobs_totalite = []
        qbaseobs_totalite = []

        year_list = []
        # trim_list=[]
        stat_list = []
        temps_list = []
        temps1_list = []
        frequen_list = []

        for stat in stations:
            # reperer les dates de NaN dans les obs pour les appliquer aux sim
            a = np.where(np.isnan(obs_qbase[stat]))[0]
            # liste_dates_nan = [[obs_qbase["year"][b], obs_qbase["months"][b], obs_qbase["day"][b]] for b in a]

            resul_sim0 = pd.read_csv(Path(self.output_dir, stat + "_monthly.csv"))

            # retirer la 1ere année de simu considérée comme l'année de chauffe
            y = list(dict.fromkeys(resul_sim0.year))
            resul_sim = resul_sim0[resul_sim0["year"] != y[0]]
            resul_sim.reset_index(drop=True, inplace=True)

            convert_dict = {"trim": str}
            resul_sim = resul_sim.astype(convert_dict)
            resul_sim["trim_year"] = " "

            resul_sim["trim_year"] = [resul_sim["trim"][i] + "_" + str(resul_sim["year"][i]) for i in range(len(resul_sim["trim_year"]))]
            resul_sim["month_year"] = " "
            resul_sim["month_year"] = [str(resul_sim["month"][i]) + "_" + str(resul_sim["year"][i]) for i in range(len(resul_sim["month_year"]))]

            trimestres_an = list(dict.fromkeys(resul_sim.trim_year))
            mois_an = list(dict.fromkeys(resul_sim.month_year))
            years_an = list(dict.fromkeys(resul_sim.year))

            # Computing simulated total and base river flow.
            if frequency in ("year", "month&year", "all"):
                yearly_qflow = pd.DataFrame(index=years_an, columns=["qtot_sim"])
                yearly_qflow.index.name = "year"

                for y in years_an:
                    subset_sim = resul_sim[resul_sim["year"] == y]
                    somme_qtot = np.sum(np.sum(subset_sim["runoff"]) + np.sum(subset_sim["subrun1"]) + np.sum(subset_sim["gwr"]))

                    somme_qbase = np.sum(np.sum(subset_sim["gwr"]))

                    qtotsim_totalite.append(somme_qtot)
                    qbasesim_totalite.append(somme_qbase)

                    qtotobs_totalite.append(np.sum(obs_qtot[obs_qtot["year"] == y][stat]))
                    qbaseobs_totalite.append(np.sum(obs_qbase[obs_qbase["year"] == y][stat]))
                    year_list.append(y)
                    stat_list.append(stat)
                    temps_list.append(y)
                    frequen_list.append("year")

            if frequency == "season" or frequency == "all":
                yearly_qflow = pd.DataFrame(index=trimestres_an, columns=["qtot_sim"])
                yearly_qflow.index.name = "trim_year"

                for y in trimestres_an:
                    subset_sim = resul_sim[resul_sim["trim_year"] == y]
                    somme_qtot = np.sum(np.sum(subset_sim["runoff"]) + np.sum(subset_sim["subrun1"]) + np.sum(subset_sim["gwr"]))

                    somme_qbase = np.sum(np.sum(subset_sim["gwr"]))

                    qtotsim_totalite.append(somme_qtot)
                    qbasesim_totalite.append(somme_qbase)

                    qtotobs_totalite.append(np.sum(obs_qtot[obs_qtot["trim_year"] == y][stat]))
                    qbaseobs_totalite.append(np.sum(obs_qbase[obs_qbase["trim_year"] == y][stat]))
                    temps_list.append(y)
                    stat_list.append(stat)
                    temps1_list.append(y.split("_")[0])
                    year_list.append(y.split("_")[1])
                    frequen_list.append("season")

            if frequency in ("month", "month&year", "all"):
                yearly_qflow = pd.DataFrame(index=mois_an, columns=["qtot_sim"])
                yearly_qflow.index.name = "month_year"

                for y in mois_an:
                    subset_sim = resul_sim[resul_sim["month_year"] == y]
                    somme_qtot = np.sum(np.sum(subset_sim["runoff"]) + np.sum(subset_sim["subrun1"]) + np.sum(subset_sim["gwr"]))

                    somme_qbase = np.sum(np.sum(subset_sim["gwr"]))

                    qtotsim_totalite.append(somme_qtot)
                    qbasesim_totalite.append(somme_qbase)

                    qtotobs_totalite.append(np.sum(obs_qtot[obs_qtot["month_year"] == y][stat]))
                    qbaseobs_totalite.append(np.sum(obs_qbase[obs_qbase["month_year"] == y][stat]))
                    temps_list.append(y)
                    stat_list.append(stat)
                    temps1_list.append(y.split("_")[0])
                    year_list.append(y.split("_")[1])
                    frequen_list.append("month")

        qflow_sim_an = pd.DataFrame()
        qflow_sim_an["qtot_sim"] = qtotsim_totalite
        qflow_sim_an["qbase_sim"] = qbasesim_totalite
        qflow_sim_an["temps_list"] = temps_list
        qflow_sim_an["station"] = stat_list
        qflow_sim_an["frequen"] = frequen_list

        # doubler le poids des débits de base pour réhausser leur importance dans la calibration
        qbasesim_totalite_double = [(qbasesim_totalite[a] * 2) for a in range(len(qbasesim_totalite))]

        qflow_sim_unique = pd.DataFrame()
        qflow_sim_unique["q"] = np.concatenate((qtotsim_totalite, qbasesim_totalite_double), axis=None)
        qflow_sim_unique["temps_list"] = np.concatenate((temps_list, temps_list), axis=None)
        qflow_sim_unique["station"] = np.concatenate((stat_list, [stat + "000" for stat in stat_list]), axis=None)
        qflow_sim_unique["frequen"] = np.concatenate((frequen_list, frequen_list), axis=None)

        qflow_obs_an = pd.DataFrame()
        qflow_obs_an["qtot_obs"] = qtotobs_totalite
        qflow_obs_an["qbase_obs"] = qbaseobs_totalite
        qflow_obs_an["temps_list"] = temps_list
        qflow_obs_an["station"] = stat_list
        qflow_obs_an["frequen"] = frequen_list

        # doubler le poids des débits de base pour réhausser leur importance dans la calibration
        qbaseobs_totalite_double = [(qbaseobs_totalite[a] * 2) for a in range(len(qbaseobs_totalite))]

        qflow_obs_unique = pd.DataFrame()
        qflow_obs_unique["q"] = np.concatenate((qtotobs_totalite, qbaseobs_totalite_double), axis=None)
        qflow_obs_unique["temps_list"] = np.concatenate((temps_list, temps_list), axis=None)
        qflow_obs_unique["station"] = np.concatenate((stat_list, [stat + "000" for stat in stat_list]), axis=None)
        qflow_obs_unique["frequen"] = np.concatenate((frequen_list, frequen_list), axis=None)

        # Application of weights to measurements to prioritize annual, then seasonal, then monthly reports.
        if frequency == "all":
            qflow_sim_unique[qflow_sim_unique["frequen"] == "year"]["q"] = qflow_sim_unique[qflow_sim_unique["frequen"] == "year"]["q"] * 5
            qflow_sim_unique[qflow_sim_unique["frequen"] == "season"]["q"] = qflow_sim_unique[qflow_sim_unique["frequen"] == "season"]["q"] * 3.3
            qflow_sim_unique[qflow_sim_unique["frequen"] == "month"]["q"] = qflow_sim_unique[qflow_sim_unique["frequen"] == "month"]["q"] * 1.7
            qflow_obs_unique[qflow_obs_unique["frequen"] == "year"]["q"] = qflow_obs_unique[qflow_obs_unique["frequen"] == "year"]["q"] * 5
            qflow_obs_unique[qflow_obs_unique["frequen"] == "season"]["q"] = qflow_obs_unique[qflow_obs_unique["frequen"] == "season"]["q"] * 3.3
            qflow_obs_unique[qflow_obs_unique["frequen"] == "month"]["q"] = qflow_obs_unique[qflow_obs_unique["frequen"] == "month"]["q"] * 1.7
        if frequency == "month&year":
            qflow_sim_unique[qflow_sim_unique["frequen"] == "year"]["q"] = qflow_sim_unique[qflow_sim_unique["frequen"] == "year"]["q"] * 4
            qflow_sim_unique[qflow_sim_unique["frequen"] == "month"]["q"] = qflow_sim_unique[qflow_sim_unique["frequen"] == "month"]["q"] * 1
            qflow_obs_unique[qflow_obs_unique["frequen"] == "year"]["q"] = qflow_obs_unique[qflow_obs_unique["frequen"] == "year"]["q"] * 4
            qflow_obs_unique[qflow_obs_unique["frequen"] == "month"]["q"] = qflow_obs_unique[qflow_obs_unique["frequen"] == "month"]["q"] * 1

        # Afficher les kge et rmse avec les poids
        qsim = qflow_sim_unique["q"]
        qobs = qflow_obs_unique["q"]
        qsim_mean = np.mean(qsim)
        qobs_mean = np.mean(qobs)
        # Calculate the components of KGE
        r_num = np.sum((qsim - qsim_mean) * (qobs - qobs_mean))
        r_den = np.sqrt(np.sum((qsim - qsim_mean) ** 2) * np.sum((qobs - qobs_mean) ** 2))
        r = r_num / r_den
        a = np.std(qsim) / np.std(qobs)
        b = np.sum(qsim) / np.sum(qobs)
        # Calculate the KGE
        kge = 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
        print(f"KGE : {kge}")
        # RMSE
        rmse = np.sqrt(np.mean((qobs - qsim) ** 2))
        print(f"RMSE : {rmse}")

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.graph_outputs > 0:
            self.plot_streamflow_scatter_color(
                self.frequency,
                sim_qflow=qflow_sim_an,
                obs_qflow=qflow_obs_an,
                output_dir=self.output_dir,
                timestamp=self.timestamp,
            )

            if self.graph_outputs > 1:
                self.plot_streamflow_scatter_color_bystation(
                    self.frequency,
                    sim_qflow=qflow_sim_an,
                    obs_qflow=qflow_obs_an,
                    output_dir=self.output_dir,
                    timestamp=self.timestamp,
                )

                self.plot_sim_vs_obs_yearly_streamflow(
                    self.frequency,
                    sim_qflow=qflow_sim_an,
                    obs_qflow=qflow_obs_an,
                    output_dir=self.output_dir,
                    timestamp=self.timestamp,
                )

        self.create_raster("precip")
        self.create_raster("gwr")
        self.create_raster("runoff")
        self.create_raster("evapo")

        self.create_netcdf(qflow_obs_unique=qflow_obs_unique, qflow_sim_unique=qflow_sim_unique, suffix="month", frequency="month")
        self.create_netcdf(qflow_obs_unique=qflow_obs_unique, qflow_sim_unique=qflow_sim_unique, suffix="", frequency=self.frequency)

    def create_netcdf(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        qflow_obs_unique,
        qflow_sim_unique,
        suffix: str or None,
        frequency: str,
    ):
        """
        Create the qobs and qsim netcdf files.

        Parameters
        ----------
        qflow_obs_unique : pd.DataFrame
            Dataframe of observed flow (base and total flow).
        qflow_sim_unique : pd.DataFrame
            Dataframe of simulated flow (base and total flow).
        suffix : str
            Tag the netcdf file to differentiate it form other netcdf files.
        frequency : str, optional
            Frequency of output data : month, season, year, month&year, all. If None, default is season.
        """
        output_dir = self.output_dir
        # Build Netcdf file with the two variables qobs ad qsim
        # Write out data to a new netCDF file with some attributes
        netcdf_obs_path = Path(output_dir, f"qobs{suffix}_netcdf.nc")
        # sys.exit()
        self.netcdf_obs_path = netcdf_obs_path
        filename = netcdf_file(netcdf_obs_path, "w")

        if frequency == "all":
            subsety_year = qflow_obs_unique[qflow_obs_unique["frequen"] == "year"]
            subsety_season = qflow_obs_unique[qflow_obs_unique["frequen"] == "season"]
            subsety_month = qflow_obs_unique[qflow_obs_unique["frequen"] == "month"]
            times = []
            year_unique = list(set(subsety_year["temps_list"]))
            times.append([datetime.strptime(str(year_unique[a]), "%Y") for a in range(len(year_unique))])
            yeartrim_unique = list(set(subsety_season["temps_list"]))
            year_unique = [yeartrim_unique[a].split("_")[1] for a in range(len(yeartrim_unique))]
            trim_unique = [yeartrim_unique[a].split("_")[0] for a in range(len(yeartrim_unique))]
            times.append([datetime.strptime(year_unique[a], "%Y") + timedelta(days=int(trim_unique[a]) * 91.25) for a in range(len(yeartrim_unique))])
            yeartrim_unique = list(set(subsety_month["temps_list"]))
            year_unique = [yeartrim_unique[a].split("_")[1] for a in range(len(yeartrim_unique))]
            trim_unique = [yeartrim_unique[a].split("_")[0] for a in range(len(yeartrim_unique))]
            times.append([datetime.strptime(year_unique[a], "%Y") + timedelta(days=int(trim_unique[a]) * 30.42) for a in range(len(yeartrim_unique))])
            times = np.concatenate(times)
        if frequency == "month&year":
            subsety_year = qflow_obs_unique[qflow_obs_unique["frequen"] == "year"]
            subsety_month = qflow_obs_unique[qflow_obs_unique["frequen"] == "month"]
            times = []
            year_unique = list(set(subsety_year["temps_list"]))
            times.append([datetime.strptime(str(year_unique[a]), "%Y") for a in range(len(year_unique))])
            yeartrim_unique = list(set(subsety_month["temps_list"]))
            year_unique = [yeartrim_unique[a].split("_")[1] for a in range(len(yeartrim_unique))]
            trim_unique = [yeartrim_unique[a].split("_")[0] for a in range(len(yeartrim_unique))]
            times.append([datetime.strptime(year_unique[a], "%Y") + timedelta(days=int(trim_unique[a]) * 30.42) for a in range(len(yeartrim_unique))])
            times = np.concatenate(times)
        if frequency == "year":
            qflow_sim_unique = qflow_sim_unique[qflow_sim_unique["frequen"] == frequency]
            qflow_obs_unique = qflow_obs_unique[qflow_obs_unique["frequen"] == frequency]
            year_unique = list(set(qflow_obs_unique["temps_list"]))
            times = [datetime.strptime(str(year_unique[a]), "%Y") for a in range(len(year_unique))]
        if frequency == "season":
            qflow_sim_unique = qflow_sim_unique[qflow_sim_unique["frequen"] == frequency]
            qflow_obs_unique = qflow_obs_unique[qflow_obs_unique["frequen"] == frequency]
            yeartrim_unique = list(set(qflow_obs_unique["temps_list"]))
            year_unique = [yeartrim_unique[a].split("_")[1] for a in range(len(yeartrim_unique))]
            trim_unique = [yeartrim_unique[a].split("_")[0] for a in range(len(yeartrim_unique))]
            times = [datetime.strptime(year_unique[a], "%Y") + timedelta(days=int(trim_unique[a]) * 91.25) for a in range(len(yeartrim_unique))]
        if frequency == "month":
            qflow_sim_unique = qflow_sim_unique[qflow_sim_unique["frequen"] == frequency]
            qflow_obs_unique = qflow_obs_unique[qflow_obs_unique["frequen"] == frequency]
            yeartrim_unique = list(set(qflow_obs_unique["temps_list"]))
            year_unique = [yeartrim_unique[a].split("_")[1] for a in range(len(yeartrim_unique))]
            trim_unique = [yeartrim_unique[a].split("_")[0] for a in range(len(yeartrim_unique))]
            times = [datetime.strptime(year_unique[a], "%Y") + timedelta(days=int(trim_unique[a]) * 30.42) for a in range(len(yeartrim_unique))]

        stations_unique = list(set(qflow_obs_unique["station"]))
        # Dimensions
        filename.createDimension("time", len(times))
        filename.createDimension("station", len(stations_unique))

        # Variables
        time = filename.createVariable("time", "i", ("time",))
        station = filename.createVariable("station", "f4", ("station",))
        streamflow = filename.createVariable("streamflow", "f4", ("station", "time"))

        # Attributes
        time.units = "days since 1970-01-01 0:0:0"
        station.units = "stations names (no unit)"
        streamflow.units = "mm/year"

        # Populate the variables with data
        time_day = [(times[d] - datetime.strptime("1970-01-01", "%Y-%m-%d")).days for d in range(len(times))]
        time[:] = time_day
        station[:] = stations_unique
        for stat in range(len(stations_unique)):
            station = stations_unique[stat]
            subset = qflow_obs_unique[qflow_obs_unique["station"] == station]
            streamflow[stat, :] = subset["q"]
        filename.close()

        # Write out data to a new netCDF file with some attributes
        # netcdf_sim_path = Path(output_dir, f"qsim{suffix}_netcdf_{self.timestamp}.nc")
        netcdf_sim_path = Path(output_dir, f"qsim{suffix}_netcdf.nc")
        # if os.path.exists(netcdf_sim_path):
        #    os.remove(netcdf_sim_path)
        self.netcdf_sim_path = netcdf_sim_path
        filename = netcdf_file(netcdf_sim_path, "w")
        # Dimensions
        filename.createDimension("time", len(times))
        filename.createDimension("station", len(stations_unique))

        # Variables
        time = filename.createVariable("time", "i", ("time",))
        station = filename.createVariable("station", "f4", ("station",))
        streamflow = filename.createVariable("streamflow", "f4", ("station", "time"))
        # qbase_sim = filename.createVariable('qbase_sim', 'f4', ('time', 'station'))

        # Attributes
        time.units = "days since 1970-01-01 0:0:0"
        station.units = "stations names (no unit)"
        streamflow.units = "mm/month"
        # Populate the variables with data
        time[:] = time_day
        station[:] = stations_unique
        for stat in range(len(stations_unique)):
            station = stations_unique[stat]
            subset = qflow_sim_unique[qflow_sim_unique["station"] == station]
            streamflow[stat, :] = subset["q"]

        filename.close()

    def plot_streamflow_scatter_color(  # noqa: C901
        # numpydoc ignore=EX01,SA01,ES01,C901
        self,
        frequency: str,
        sim_qflow: pd.DataFrame,
        obs_qflow: pd.DataFrame,
        output_dir: Path,
        timestamp: str,
    ):
        """
        Create a scatter plot comparing simulated total and base streamflow yearly values with observed values.

        Parameters
        ----------
        frequency : str, optional
            Frequency of output data : month, season, year, month&year, all. If None, default is season.
        sim_qflow : pd.DataFrame
            Dataframe of simulated flow (base and total flow).
        obs_qflow : pd.DataFrame
            Dataframe of observed flow (base and total flow).
        output_dir : Path
            Path to output directiry for graphs.
        timestamp : str
            Timestamp to identify output times.
        """
        self.frequency = frequency
        self.output_dir = output_dir

        # déterminantion des conditions pour réaliser les figures dans tous les cas de fréquence
        if frequency == "all":
            fin_decompte = 3
            freq = ["year", "season", "month"]
        if frequency == "month&year":
            fin_decompte = 2
            freq = ["year", "month"]
        if frequency == "year":
            fin_decompte = 1
            freq = ["year"]
        if frequency == "season":
            fin_decompte = 1
            freq = ["season"]
        if frequency == "month":
            fin_decompte = 1
            freq = ["month"]

        decompte = 0
        while decompte < fin_decompte:
            subset_obs_qflow = obs_qflow[obs_qflow["frequen"] == freq[decompte]]
            subset_sim_qflow = sim_qflow[sim_qflow["frequen"] == freq[decompte]]

            # Join both dataframe in a single dataframe.
            yearly_qflow = subset_sim_qflow.copy()
            yearly_qflow = yearly_qflow.reindex(index=subset_obs_qflow.index)
            yearly_qflow.loc[subset_obs_qflow.index, "qtot_obs"] = subset_obs_qflow["qtot_obs"]
            yearly_qflow.loc[subset_obs_qflow.index, "qbase_obs"] = subset_obs_qflow["qbase_obs"]
            yearly_qflow.loc[subset_obs_qflow.index, "temps_list"] = subset_obs_qflow["temps_list"]

            if freq[decompte] == "year":
                yearly_qflow["temps_list1"] = [yearly_qflow["temps_list"][a] for a in yearly_qflow.index]

            else:
                yearly_qflow["temps_list1"] = [yearly_qflow["temps_list"][a].split("_")[0] for a in yearly_qflow.index]

            septemporelle = list(dict.fromkeys(yearly_qflow.temps_list1))

            def generate_colors(x, colormap="Set1"):
                cmap = plt.get_cmap(colormap)
                colors = [cmap(i / (x - 1)) for i in range(x)]
                return colors

            if freq[decompte] == "year":
                years_nb = len(list(dict.fromkeys(yearly_qflow.temps_list1)))
                colorslist = generate_colors(years_nb)
                qmin = 200
                qmax = 1400
                qbasemax = 800
                frequen = "année"
                anchor_y = 1.3
            if freq[decompte] == "season":
                colorslist = ["mediumturquoise", "green", "gold", "firebrick"]
                qmin = 0
                qmax = 700
                qbasemax = 250
                frequen = "saison"
                anchor_y = 1.35
            if freq[decompte] == "month":
                colorslist = generate_colors(12)
                qmin = 0
                qmax = 600
                qbasemax = 150
                anchor_y = 1.35
                frequen = "mois"

            # Figure Qtot
            fwidth, fheight = 6, 5
            fig, ax = plt.subplots()
            fig.set_size_inches(fwidth, fheight)

            left_margin = 1 / fwidth
            right_margin = 1.3 / fwidth
            top_margin = 0.5 / fheight
            bot_margin = 1 / fheight
            ax.set_position(
                [
                    left_margin,
                    bot_margin,
                    1 - left_margin - right_margin,
                    1 - top_margin - bot_margin,
                ]
            )

            xymin, xymax = qmin, qmax
            ax.axis([xymin, xymax, xymin, xymax])
            ax.set_ylabel("Débits Simulés (mm/an)", fontsize=16, labelpad=15)
            ax.set_xlabel("Débits Observés (mm/an)", fontsize=16, labelpad=15)

            ax.tick_params(axis="both", direction="out", labelsize=12)

            (l1,) = ax.plot([xymin, xymax], [xymin, xymax], "--", color="black", lw=1)

            for tr in range(len(septemporelle)):
                subset = yearly_qflow[yearly_qflow["temps_list1"] == septemporelle[tr]]
                (l2,) = ax.plot(subset["qtot_obs"], subset["qtot_sim"], ".", color=colorslist[tr])

            # Plot the model fit stats.
            rmse_qtot = np.nanmean((yearly_qflow["qtot_sim"] - yearly_qflow["qtot_obs"]) ** 2) ** 0.5
            me_qtot = np.nanmean(yearly_qflow["qtot_sim"] - yearly_qflow["qtot_obs"])

            dx, dy = 3, -3
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"RMSE débit total = {rmse_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )
            dy += -12
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"ME débit total = {me_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )

            # Add a graph title.
            fig_title1 = f"Débits total par {frequen}"
            offset = transforms.ScaledTranslation(0 / 72, 12 / 72, fig.dpi_scale_trans)
            ax.text(
                0.5,
                1,
                fig_title1,
                fontsize=16,
                ha="center",
                va="bottom",
                transform=ax.transAxes + offset,
            )
            # Add a legend.
            if freq[decompte] == "year":
                septemporelle_str = [str(septemporelle[tr]) for tr in range(len(septemporelle))]
                color = ["1:1"]
                colors = color + septemporelle_str
            if freq[decompte] == "season":
                colors = ["1:1", "Hiver", "Printemps", "Été", "Automne"]
            if freq[decompte] == "month":
                colors = [
                    "1:1",
                    "Janvier",
                    "Février",
                    "Mars",
                    "Avril",
                    "Mai",
                    "Juin",
                    "Juillet",
                    "Août",
                    "Septembre",
                    "Octrobre",
                    "Novembre",
                    "Décembre",
                ]
            legend = ax.legend(
                colors,
                numpoints=1,
                fontsize=10,
                borderaxespad=0,
                loc="lower right",
                borderpad=0.5,
                bbox_to_anchor=(anchor_y, 0),
                ncol=1,
            )
            legend.draw_frame(False)
            fig_path1 = Path(
                self.output_dir,
                f"SP_{freq[decompte]}_tot_allstations_{self.timestamp}.png",
            )
            fig.savefig(fig_path1, dpi=200)

            # Figure Qbase
            fwidth, fheight = 6, 5
            fig, ax = plt.subplots()
            fig.set_size_inches(fwidth, fheight)

            left_margin = 1 / fwidth
            right_margin = 1.3 / fwidth
            top_margin = 0.5 / fheight
            bot_margin = 1 / fheight
            ax.set_position(
                [
                    left_margin,
                    bot_margin,
                    1 - left_margin - right_margin,
                    1 - top_margin - bot_margin,
                ]
            )

            xymin, xymax = 0, qbasemax
            ax.axis([xymin, xymax, xymin, xymax])
            ax.set_ylabel("Débits Simulés (mm/an)", fontsize=16, labelpad=15)
            ax.set_xlabel("Débits Observés (mm/an)", fontsize=16, labelpad=15)

            ax.tick_params(axis="both", direction="out", labelsize=12)

            (l1,) = ax.plot([xymin, xymax], [xymin, xymax], "--", color="black", lw=1)

            for tr in range(len(septemporelle)):
                subset = yearly_qflow[yearly_qflow["temps_list1"] == septemporelle[tr]]
                (l2,) = ax.plot(subset["qbase_obs"], subset["qbase_sim"], ".", color=colorslist[tr])

            # Plot the model fit stats.
            rmse_qtot = np.nanmean((yearly_qflow["qbase_sim"] - yearly_qflow["qbase_obs"]) ** 2) ** 0.5
            me_qtot = np.nanmean(yearly_qflow["qbase_sim"] - yearly_qflow["qbase_obs"])

            dx, dy = 3, -3
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"RMSE débit total = {rmse_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )
            dy += -12
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"ME débit total = {me_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )

            # Add a graph title.
            fig_title2 = f"Débits de base par {frequen}"
            offset = transforms.ScaledTranslation(0 / 72, 12 / 72, fig.dpi_scale_trans)
            ax.text(
                0.5,
                1,
                fig_title2,
                fontsize=16,
                ha="center",
                va="bottom",
                transform=ax.transAxes + offset,
            )

            # Add a legend.
            if freq[decompte] == "year":
                septemporelle_str = [str(septemporelle[tr]) for tr in range(len(septemporelle))]
                color = ["1:1"]
                colors = color + septemporelle_str
            if freq[decompte] == "season":
                colors = ["1:1", "Hiver", "Printemps", "Été", "Automne"]
            if freq[decompte] == "month":
                colors = [
                    "1:1",
                    "Janvier",
                    "Février",
                    "Mars",
                    "Avril",
                    "Mai",
                    "Juin",
                    "Juillet",
                    "Août",
                    "Septembre",
                    "Octrobre",
                    "Novembre",
                    "Décembre",
                ]

            legend = ax.legend(
                colors,
                numpoints=1,
                fontsize=10,
                borderaxespad=0,
                loc="lower right",
                borderpad=0.5,
                bbox_to_anchor=(anchor_y, 0),
                ncol=1,
            )
            legend.draw_frame(False)
            fig_path2 = Path(
                self.output_dir,
                f"SP_{freq[decompte]}_base_allstations_{self.timestamp}.png",
            )
            fig.savefig(fig_path2, dpi=200)

            decompte = decompte + 1

        return

    def plot_streamflow_scatter_color_bystation(
        # numpydoc ignore=EX01,SA01,ES01,C901
        self,
        frequency: str,
        sim_qflow: pd.DataFrame,
        obs_qflow: pd.DataFrame,
        output_dir: Path,
        timestamp: str,
    ):
        """
        Create a scatter plot comparing simulated total and base streamflow yearly values with observed values.

        Parameters
        ----------
        frequency : str, optional
            Frequency of output data : month, season, year, month&year, all. If None, default is season.
        sim_qflow : pd.DataFrame
            Dataframe of simulated flow (base and total flow).
        obs_qflow : pd.DataFrame
            Dataframe of observed flow (base and total flow).
        output_dir : Path
            Path to output directiry for graphs.
        timestamp : str
            Timestamp to identify output times.
        """
        self.frequency = frequency
        self.output_dir = output_dir

        # déterminantion des conditions pour réaliser les figures dans tous les cas de fréquence
        if frequency == "all":
            fin_decompte = 3
            freq = ["year", "season", "month"]
        if frequency == "month&year":
            fin_decompte = 3
            freq = ["year", "month"]
        if frequency == "year":
            fin_decompte = 1
            freq = ["year"]
        if frequency == "season":
            fin_decompte = 1
            freq = ["season"]
        if frequency == "month":
            fin_decompte = 1
            freq = ["month"]

        decompte = 0
        while decompte < fin_decompte:
            subset_obs_qflow = obs_qflow[obs_qflow["frequen"] == freq[decompte]]
            subset_sim_qflow = sim_qflow[sim_qflow["frequen"] == freq[decompte]]

            # Join both dataframe in a single dataframe.
            yearly_qflow = subset_sim_qflow.copy()
            yearly_qflow = yearly_qflow.reindex(index=subset_obs_qflow.index)
            yearly_qflow.loc[subset_obs_qflow.index, "qtot_obs"] = subset_obs_qflow["qtot_obs"]
            yearly_qflow.loc[subset_obs_qflow.index, "qbase_obs"] = subset_obs_qflow["qbase_obs"]
            yearly_qflow.loc[subset_obs_qflow.index, "temps_list"] = subset_obs_qflow["temps_list"]

            if freq[decompte] == "year":
                yearly_qflow["temps_list1"] = [yearly_qflow["temps_list"][a] for a in yearly_qflow.index]

            else:
                yearly_qflow["temps_list1"] = [yearly_qflow["temps_list"][a].split("_")[0] for a in yearly_qflow.index]

            def generate_colors(x, colormap="Set1"):
                cmap = plt.get_cmap(colormap)
                colors = [cmap(i / (x - 1)) for i in range(x)]
                return colors

            station_ = list(set(yearly_qflow.station))
            colorslist = generate_colors(len(station_))

            if freq[decompte] == "year":
                qmin = 200
                qmax = 1400
                qbasemax = 800
                frequen = "année"
                anchor_y = 1.3
            if freq[decompte] == "season":
                qmin = 0
                qmax = 700
                qbasemax = 250
                frequen = "saison"
                anchor_y = 1.35
            if freq[decompte] == "month":
                # colorslist=generate_colors(12)
                qmin = 0
                qmax = 600
                qbasemax = 150
                anchor_y = 1.35
                frequen = "mois"

            # Figure Qtot
            fwidth, fheight = 6, 5
            fig, ax = plt.subplots()
            fig.set_size_inches(fwidth, fheight)

            left_margin = 1 / fwidth
            right_margin = 1.3 / fwidth
            top_margin = 0.5 / fheight
            bot_margin = 1 / fheight
            ax.set_position(
                [
                    left_margin,
                    bot_margin,
                    1 - left_margin - right_margin,
                    1 - top_margin - bot_margin,
                ]
            )

            xymin, xymax = qmin, qmax
            ax.axis([xymin, xymax, xymin, xymax])
            ax.set_ylabel("Débits Simulés (mm/an)", fontsize=16, labelpad=15)
            ax.set_xlabel("Débits Observés (mm/an)", fontsize=16, labelpad=15)

            ax.tick_params(axis="both", direction="out", labelsize=12)

            (l1,) = ax.plot([xymin, xymax], [xymin, xymax], "--", color="black", lw=1)

            for tr in range(len(station_)):
                subset = yearly_qflow[yearly_qflow["station"] == station_[tr]]
                (l2,) = ax.plot(subset["qtot_obs"], subset["qtot_sim"], ".", color=colorslist[tr])

            # Plot the model fit stats.
            rmse_qtot = np.nanmean((yearly_qflow["qtot_sim"] - yearly_qflow["qtot_obs"]) ** 2) ** 0.5
            me_qtot = np.nanmean(yearly_qflow["qtot_sim"] - yearly_qflow["qtot_obs"])

            dx, dy = 3, -3
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"RMSE débit total = {rmse_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )
            dy += -12
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"ME débit total = {me_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )

            # Add a graph title.
            fig_title1 = f"Débits total par {frequen}"
            offset = transforms.ScaledTranslation(0 / 72, 12 / 72, fig.dpi_scale_trans)
            ax.text(
                0.5,
                1,
                fig_title1,
                fontsize=16,
                ha="center",
                va="bottom",
                transform=ax.transAxes + offset,
            )
            # Add a legend.
            color = ["1:1"]
            colors = color + station_

            legend = ax.legend(
                colors,
                numpoints=1,
                fontsize=10,
                borderaxespad=0,
                loc="lower right",
                borderpad=0.5,
                bbox_to_anchor=(anchor_y, 0),
                ncol=1,
            )
            legend.draw_frame(False)
            fig_path1 = Path(
                self.output_dir,
                f"SP_{freq[decompte]}_tot_allstations_bystation_{self.timestamp}.png",
            )
            fig.savefig(fig_path1, dpi=200)

            # Figure Qbase
            fwidth, fheight = 6, 5
            fig, ax = plt.subplots()
            fig.set_size_inches(fwidth, fheight)

            left_margin = 1 / fwidth
            right_margin = 1.3 / fwidth
            top_margin = 0.5 / fheight
            bot_margin = 1 / fheight
            ax.set_position(
                [
                    left_margin,
                    bot_margin,
                    1 - left_margin - right_margin,
                    1 - top_margin - bot_margin,
                ]
            )

            xymin, xymax = 0, qbasemax
            ax.axis([xymin, xymax, xymin, xymax])
            ax.set_ylabel("Débits Simulés (mm/an)", fontsize=16, labelpad=15)
            ax.set_xlabel("Débits Observés (mm/an)", fontsize=16, labelpad=15)

            ax.tick_params(axis="both", direction="out", labelsize=12)

            (l1,) = ax.plot([xymin, xymax], [xymin, xymax], "--", color="black", lw=1)

            for tr in range(len(station_)):
                subset = yearly_qflow[yearly_qflow["station"] == station_[tr]]
                (l2,) = ax.plot(subset["qbase_obs"], subset["qbase_sim"], ".", color=colorslist[tr])

            # Plot the model fit stats.
            rmse_qtot = np.nanmean((yearly_qflow["qbase_sim"] - yearly_qflow["qbase_obs"]) ** 2) ** 0.5
            me_qtot = np.nanmean(yearly_qflow["qbase_sim"] - yearly_qflow["qbase_obs"])

            dx, dy = 3, -3
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"RMSE débit total = {rmse_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )
            dy += -12
            offset = transforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
            ax.text(
                0,
                1,
                f"ME débit total = {me_qtot:0.1f} mm/an",
                transform=ax.transAxes + offset,
                ha="left",
                va="top",
            )

            # Add a graph title.
            fig_title2 = f"Débits de base par {frequen}"
            offset = transforms.ScaledTranslation(0 / 72, 12 / 72, fig.dpi_scale_trans)
            ax.text(
                0.5,
                1,
                fig_title2,
                fontsize=16,
                ha="center",
                va="bottom",
                transform=ax.transAxes + offset,
            )

            # Add a legend.
            color = ["1:1"]
            colors = color + station_

            legend = ax.legend(
                colors,
                numpoints=1,
                fontsize=10,
                borderaxespad=0,
                loc="lower right",
                borderpad=0.5,
                bbox_to_anchor=(anchor_y, 0),
                ncol=1,
            )
            legend.draw_frame(False)
            fig_path2 = Path(
                self.output_dir,
                f"SP_{freq[decompte]}_base_allstations_bystation_{self.timestamp}.png",
            )
            fig.savefig(fig_path2, dpi=200)

            decompte = decompte + 1

        return

    def plot_sim_vs_obs_yearly_streamflow(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        frequency: str,
        sim_qflow: pd.DataFrame,
        obs_qflow: pd.DataFrame,
        output_dir: Path,
        timestamp: str,
    ):
        """
        Plot simulated vs observed yearly total and base streamflow.

        Parameters
        ----------
        frequency : str, optional
            Frequency of output data : month, season, year, month&year, all. If None, default is season.
        sim_qflow : pd.DataFrame
            Dataframe of simulated flow (base and total flow).
        obs_qflow : pd.DataFrame
            Dataframe of observed flow (base and total flow).
        output_dir : Path
            Path to output directiry for graphs.
        timestamp : str
            Timestamp to identify output times.
        """
        self.timestamp = timestamp
        self.output_dir = output_dir
        self.frequency = frequency

        # déterminantion des conditions pour réaliser les figures dans tous les cas de fréquence
        if frequency == "all":
            fin_decompte = 3
            freq = ["year", "season", "month"]
        if frequency == "month&year":
            fin_decompte = 3
            freq = ["year", "month"]
        if frequency == "year":
            fin_decompte = 1
            freq = ["year"]
        if frequency == "season":
            fin_decompte = 1
            freq = ["season"]
        if frequency == "month":
            fin_decompte = 1
            freq = ["month"]

        decompte = 0
        while decompte < fin_decompte:
            subset_obs_qflow = obs_qflow[obs_qflow["frequen"] == freq[decompte]]
            subset_sim_qflow = sim_qflow[sim_qflow["frequen"] == freq[decompte]]

            # Join both dataframe in a single dataframe.
            yearly_qflow = subset_sim_qflow.copy()
            yearly_qflow = yearly_qflow.reindex(index=subset_obs_qflow.index)
            yearly_qflow.loc[subset_obs_qflow.index, "qtot_obs"] = subset_obs_qflow["qtot_obs"]
            yearly_qflow.loc[subset_obs_qflow.index, "qbase_obs"] = subset_obs_qflow["qbase_obs"]
            yearly_qflow.loc[subset_obs_qflow.index, "temps_list"] = subset_obs_qflow["temps_list"]

            if freq[decompte] == "year":
                yearly_qflow["year"] = [yearly_qflow["temps_list"][a] for a in yearly_qflow.index]
                yearly_qflow["date"] = [datetime.strptime(str(yearly_qflow.loc[a, "year"]) + "/01/01", "%Y/%m/%d") for a in yearly_qflow.index]
                yearly_qflow["temps_list1"] = yearly_qflow["date"]

            if freq[decompte] == "season":
                mois_deb_saison = [1, 4, 7, 10]
                nom_saison = ["Hiver", "Printemps", "Été", "Automne"]
                yearly_qflow["year"] = [int(yearly_qflow["temps_list"][a].split("_")[1]) for a in yearly_qflow.index]
                yearly_qflow["month"] = [int(yearly_qflow["temps_list"][a].split("_")[0]) for a in yearly_qflow.index]
                yearly_qflow["temps_list_ecrite"] = [
                    nom_saison[yearly_qflow.loc[a, "month"] - 1] + " " + str(yearly_qflow.loc[a, "year"]) for a in yearly_qflow.index
                ]
                yearly_qflow["month"] = [mois_deb_saison[yearly_qflow.loc[a, "month"] - 1] for a in yearly_qflow.index]

                # # repérage des mois de décembre pour enlever 1 an à l'année correspondante
                # rep_dec=np.where(yearly_qflow['month']==12)
                # yearly_qflow.loc[rep_dec[0],'year']=yearly_qflow.loc[rep_dec[0],'year']-1

                yearly_qflow["date"] = [
                    datetime.strptime(
                        str(yearly_qflow.loc[a, "year"]) + "/" + str(yearly_qflow.loc[a, "month"]) + "/01",
                        "%Y/%m/%d",
                    )
                    for a in yearly_qflow.index
                ]
                yearly_qflow["temps_list1"] = yearly_qflow["date"]

            if freq[decompte] == "month":
                yearly_qflow["year"] = [yearly_qflow["temps_list"][a].split("_")[1] for a in yearly_qflow.index]
                yearly_qflow["month"] = [yearly_qflow["temps_list"][a].split("_")[0] for a in yearly_qflow.index]
                yearly_qflow["date"] = [
                    datetime.strptime(
                        yearly_qflow.loc[a, "year"] + "/" + yearly_qflow.loc[a, "month"] + "/01",
                        "%Y/%m/%d",
                    )
                    for a in yearly_qflow.index
                ]
                yearly_qflow["temps_list1"] = yearly_qflow["date"]

            stations = list(set(obs_qflow["station"]))

            for stat in range(len(stations)):
                subset_yearly_qflow = yearly_qflow[yearly_qflow.station == stations[stat]]

                if freq[decompte] == "year":
                    qmin = 0
                    qmax = 1400
                    frequen_tit = "annuels"
                    large = 7
                if freq[decompte] == "season":
                    qmin = 0
                    qmax = 700
                    frequen_tit = "saisonniers"
                    large = 10
                if freq[decompte] == "month":
                    qmin = 0
                    qmax = 600
                    frequen_tit = "mensuels"
                    large = 15

                fwidth, fheight = large, 5.5
                fig, ax = plt.subplots()
                fig.set_size_inches(fwidth, fheight)

                left_margin = 1.5 / fwidth
                right_margin = 0.25 / fwidth
                top_margin = 0.5 / fheight
                bot_margin = 0.7 / fheight
                ax.set_position(
                    [
                        left_margin,
                        bot_margin,
                        1 - left_margin - right_margin,
                        1 - top_margin - bot_margin,
                    ]
                )

                (l1,) = ax.plot(
                    subset_yearly_qflow["temps_list1"],
                    subset_yearly_qflow["qtot_obs"],
                    clip_on=True,
                    lw=2,
                    linestyle="--",
                    color="#3690c0",
                )

                (l2,) = ax.plot(
                    subset_yearly_qflow["temps_list1"],
                    subset_yearly_qflow["qtot_sim"],
                    clip_on=True,
                    lw=2,
                    linestyle="-",
                    color="#034e7b",
                )

                # Streamflow base
                (l3,) = ax.plot(
                    subset_yearly_qflow["temps_list1"],
                    subset_yearly_qflow["qbase_obs"],
                    lw=2,
                    linestyle="--",
                    clip_on=True,
                    color="#ef6548",
                )

                (l4,) = ax.plot(
                    subset_yearly_qflow["temps_list1"],
                    subset_yearly_qflow["qbase_sim"],
                    clip_on=True,
                    lw=2,
                    linestyle="-",
                    color="#990000",
                )

                ax.tick_params(axis="both", direction="out", labelsize=12)
                ax.set_ylabel("Débit par unité de surface\n(mm/an)", fontsize=16, labelpad=10)
                ax.set_xlabel("Date", fontsize=16, labelpad=10)
                ax.axis(ymin=qmin, ymax=qmax)
                ax.grid(axis="y", color=[0.35, 0.35, 0.35], ls="-", lw=0.5)
                ax.grid(axis="x", color=[0.35, 0.35, 0.35], ls="-", lw=0.5)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
                myfmt = DateFormatter("%Y")
                ax.xaxis.set_major_formatter(myfmt)

                # ax.autofmt_xdate()
                # ax.axis(axis_)

                lines = [l1, l2, l3, l4]
                labels = [
                    "Débit total observé",
                    "Débit total simulé",
                    "Débit base observé",
                    "Débit base simulé",
                ]
                legend = ax.legend(
                    lines,
                    labels,
                    numpoints=1,
                    fontsize=12,
                    borderaxespad=0,
                    loc="upper left",
                    borderpad=0.5,
                    bbox_to_anchor=(0, 1),
                    ncol=2,
                )
                legend.draw_frame(False)

                # Add a graph title.
                fig_title = f"Débits {frequen_tit} à la station {stations[stat]}"
                offset = transforms.ScaledTranslation(0 / 72, 12 / 72, fig.dpi_scale_trans)
                ax.text(
                    0.5,
                    1,
                    fig_title,
                    fontsize=16,
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes + offset,
                )

                folder_path = Path(self.output_dir, f"{stations[stat]}")
                folder_path.mkdir(parents=True, exist_ok=True)

                chemin = Path(
                    folder_path,
                    f"obsVSsim_{frequen_tit}_station{stations[stat]}_{self.timestamp}.png",
                )
                # Save figure to file.
                fig.savefig(chemin)

            decompte = decompte + 1

        return

    def create_raster(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        variable,
    ):
        """
        Create a raster of average yearly groundwater recharge.

        Parameters
        ----------
        variable : str
            Set variable for the raster : precip or gwr.
        """
        columns_to_read = ["cid", "lat_dd", "lon_dd", "rechg", "precip", "runoff", "evapo"]

        gwr_an_spat = pd.read_csv(
            Path(self.output_dir, "gwr_annuel_spat.csv"),
            delimiter=",",
            usecols=columns_to_read,
        )

        df = gwr_an_spat.rename(columns={"lon_dd": "lon", "lat_dd": "lat", "rechg": "gwr"})
        pixel_size = 1000  # 1000 m taille des mailles du modèle

        # 1. Définir les CRS

        crs_in = pyproj.CRS("EPSG:8237")  # lat/lon GRS80
        crs_out = pyproj.CRS("EPSG:6622")  # Lambert 93
        transformer = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)

        # 2. Conversion en Lambert 93

        df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)

        # 3. Définir l'étendue de la grille

        min_x = df["x"].min() - pixel_size / 2
        max_x = df["x"].max() + pixel_size / 2
        min_y = df["y"].min() - pixel_size / 2
        max_y = df["y"].max() + pixel_size / 2

        # 4. Construire les vecteurs de grille

        x_vals = np.arange(min_x, max_x, pixel_size)
        y_vals = np.arange(max_y, min_y, -pixel_size)  # du nord vers le sud

        # 5. Créer une matrice vide

        raster = np.full((len(y_vals), len(x_vals)), np.nan)

        # 6. Remplir le raster
        if variable == "precip":
            for _, row in df.iterrows():
                ix = int((row["x"] - min_x) // pixel_size)
                iy = int((max_y - row["y"]) // pixel_size)
                raster[iy, ix] = row["precip"]
        if variable == "runoff":
            for _, row in df.iterrows():
                ix = int((row["x"] - min_x) // pixel_size)
                iy = int((max_y - row["y"]) // pixel_size)
                raster[iy, ix] = row["runoff"]
        if variable == "evapo":
            for _, row in df.iterrows():
                ix = int((row["x"] - min_x) // pixel_size)
                iy = int((max_y - row["y"]) // pixel_size)
                raster[iy, ix] = row["evapo"]
        if variable == "gwr":
            for _, row in df.iterrows():
                ix = int((row["x"] - min_x) // pixel_size)
                iy = int((max_y - row["y"]) // pixel_size)
                raster[iy, ix] = row["gwr"]

        # 7. Définir la transformation géographique

        transform = from_origin(x_vals[0], y_vals[0], pixel_size, pixel_size)

        # 8. Écrire le raster GeoTIFF

        with rasterio.open(
            Path(self.output_dir, f"{variable}_raster.tiff"),
            "w",
            driver="GTiff",
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs=crs_out,
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(raster, 1)

        return
