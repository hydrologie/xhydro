# numpydoc ignore=EX01,SA01,ES01
"""Class to handle Hydrotel simulations."""

import os
import sys
import subprocess  # noqa: S404
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import netcdf_file

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import csv
import shutil 
from matplotlib import colormaps
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

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
    start_date : str, optional
        Calibration start date.
    end_date : str, optional
        Calibration end date.
    frequency : str, optional
        frequency of output data : month, season, year. If None, default is season.
    """

    def __init__(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        project_dir: str | os.PathLike,
        executable: str | os.PathLike,
        output_config: dict | None = None,
        simulation_config: dict | None = None,
        parameters: np.ndarray | list[float] | None = None,
        parameters_names: np.ndarray | list[float] | None = None,
        qobs: np.ndarray | xr.Dataset | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str | None = None,
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
        start_date : str, optional
            Calibration start date.
        end_date : str, optional
            Calibration end date.
        frequency : str, optional
            frequency of output data : month, season, year. If None, default is season.
        """
        output_config = output_config or dict()
        qobs = qobs or None
        start_date = start_date or None
        end_date = end_date or None

        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValueError("The project folder does not exist.")

        self.executable = str(Path(executable))
        self.qobs = qobs
        self.parameters = parameters
        self.parameters_names = parameters_names
        self.simu_begin = simulation_config["DATE DEBUT"]
        self.simu_end = simulation_config["DATE FIN"]
        self.cal_start_date = start_date
        self.cal_end_date = end_date
        self.frequency = frequency

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
        # Copy param file with data 
        source_path = param_file
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        destination_path = Path(self.project_dir, f"param_{self.timestamp}.txt")
        shutil.copy(source_path, destination_path)
        
        """Run simulation."""
        subprocess.call(self.executable, shell=True)

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
        frequency = self.frequency
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
            if file.endswith("monthly.csv")
        ]

        list_stations_interm = [
            list_output_files_stations[i].split("\\")[-1]
            for i in range(len(list_output_files_stations))
        ]
        list_stations = [
            list_stations_interm[i].split("_")[0]
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
        # Qobs et Qsim
        # Get scatter figures
                   
        obs_qtot = pd.read_csv(Path(self.output_dir, 'observed_flow.csv'))
        obs_qbase = pd.read_csv(Path(self.output_dir,'eckhardt_baseflow.csv'))
        stations=obs_qtot.keys()[3:]
         
        obs_qtot.loc[obs_qtot.months.isin([12,1,2]),"trim"]="1"
        obs_qtot.loc[obs_qtot.months.isin([3,4,5]),"trim"]="2"
        obs_qtot.loc[obs_qtot.months.isin([6,7,8]),"trim"]="3"
        obs_qtot.loc[obs_qtot.months.isin([9,10,11]),"trim"]="4"
        
        obs_qbase.loc[obs_qbase.months.isin([12,1,2]),"trim"]="1"
        obs_qbase.loc[obs_qbase.months.isin([3,4,5]),"trim"]="2"
        obs_qbase.loc[obs_qbase.months.isin([6,7,8]),"trim"]="3"
        obs_qbase.loc[obs_qbase.months.isin([9,10,11]),"trim"]="4"
        
        obs_qtot["trim_year"]=" "
        obs_qtot["trim_year"]=[obs_qtot["trim"][i]+"_"+str(obs_qtot["year"][i]) for i in range(len(obs_qtot["trim_year"]))]
        obs_qbase["trim_year"]=" "
        obs_qbase["trim_year"]=[obs_qbase["trim"][i]+"_"+str(obs_qbase["year"][i]) for i in range(len(obs_qbase["trim_year"]))]

        obs_qtot["month_year"]=" "
        obs_qtot["month_year"]=[str(obs_qtot["months"][i])+"_"+str(obs_qtot["year"][i]) for i in range(len(obs_qtot["trim_year"]))]
        obs_qbase["month_year"]=" "
        obs_qbase["month_year"]=[str(obs_qbase["months"][i])+"_"+str(obs_qbase["year"][i]) for i in range(len(obs_qbase["trim_year"]))]
        
        qtotsim_totalite=[]
        qbasesim_totalite=[]
        qtotobs_totalite=[]
        qbaseobs_totalite=[]
        
        yeartrim_list=[]
        year_list=[]
        #trim_list=[]
        stat_list=[]
        temps_list=[]
        temps1_list=[]
        
        for stat in stations:
            resul_sim = pd.read_csv(Path(self.output_dir,stat+"_monthly.csv"))
            convert_dict = {'trim': str}
            resul_sim = resul_sim.astype(convert_dict)
            resul_sim["trim_year"]=" "
            resul_sim["trim_year"]=[resul_sim["trim"][i]+"_"+str(resul_sim["year"][i]) for i in range(len(resul_sim["trim_year"]))]
            resul_sim["month_year"]=" "
            resul_sim["month_year"]=[str(resul_sim["month"][i])+"_"+str(resul_sim["year"][i]) for i in range(len(resul_sim["month_year"]))]
            
            trimestres_an=list(dict.fromkeys(resul_sim.trim_year))
            mois_an=list(dict.fromkeys(resul_sim.month_year))
            years_an=list(dict.fromkeys(resul_sim.year))
            
            # Computing simulated total and base river flow.
            if frequency=="year":
                yearly_qflow = pd.DataFrame(index=years_an, columns=['qtot_sim'])
                yearly_qflow.index.name = 'year'
                
                for y in years_an: 
                    subset_sim=resul_sim[resul_sim["year"]==y]
                    somme_qtot=np.sum(np.sum(subset_sim["runoff"])+
                                      np.sum(subset_sim["runoff_2"])+
                                      np.sum(subset_sim["gwr"]))
                    
                    somme_qbase=np.sum(np.sum(subset_sim["gwr"]))
                    
                    qtotsim_totalite.append(somme_qtot)
                    qbasesim_totalite.append(somme_qbase)
    
                    qtotobs_totalite.append(np.sum(obs_qtot[obs_qtot["year"]==y][stat]))
                    qbaseobs_totalite.append(np.sum(obs_qbase[obs_qbase["year"]==y][stat]))
                    year_list.append(y)
                    stat_list.append(stat)
                    temps_list.append(y)
                    
            if frequency=="season":
                yearly_qflow = pd.DataFrame(index=trimestres_an, columns=['qtot_sim'])
                yearly_qflow.index.name = 'trim_year'
                
                for y in trimestres_an: 
                    subset_sim=resul_sim[resul_sim["trim_year"]==y]
                    somme_qtot=np.sum(np.sum(subset_sim["runoff"])+
                                      np.sum(subset_sim["runoff_2"])+
                                      np.sum(subset_sim["gwr"]))
                    
                    somme_qbase=np.sum(np.sum(subset_sim["gwr"]))
                    
                    qtotsim_totalite.append(somme_qtot)
                    qbasesim_totalite.append(somme_qbase)
    
                    qtotobs_totalite.append(np.sum(obs_qtot[obs_qtot["trim_year"]==y][stat]))
                    qbaseobs_totalite.append(np.sum(obs_qbase[obs_qbase["trim_year"]==y][stat]))
                    temps_list.append(y)
                    stat_list.append(stat)
                    temps1_list.append(y.split('_')[0])
                    year_list.append(y.split('_')[1])
             
            if frequency=="month":
                yearly_qflow = pd.DataFrame(index=mois_an, columns=['qtot_sim'])
                yearly_qflow.index.name = 'month_year'
                
                for y in mois_an: 
                    subset_sim=resul_sim[resul_sim["month_year"]==y]
                    somme_qtot=np.sum(np.sum(subset_sim["runoff"])+
                                      np.sum(subset_sim["runoff_2"])+
                                      np.sum(subset_sim["gwr"]))
                    
                    somme_qbase=np.sum(np.sum(subset_sim["gwr"]))
                    
                    qtotsim_totalite.append(somme_qtot)
                    qbasesim_totalite.append(somme_qbase)
        
                    qtotobs_totalite.append(np.sum(obs_qtot[obs_qtot["month_year"]==y][stat]))
                    qbaseobs_totalite.append(np.sum(obs_qbase[obs_qbase["month_year"]==y][stat]))
                    temps_list.append(y)
                    stat_list.append(stat)
                    temps1_list.append(y.split('_')[0])
                    year_list.append(y.split('_')[1])
                    
                    
                 
        qflow_sim_an = pd.DataFrame()
        qflow_sim_an["qtot_sim"]=qtotsim_totalite
        qflow_sim_an["qbase_sim"]=qbasesim_totalite
        qflow_sim_an["temps_list"]=temps_list
        qflow_sim_an["station"]=stat_list

        qflow_sim_unique = pd.DataFrame()
        qflow_sim_unique["q"]=np.concatenate((qtotsim_totalite,qbasesim_totalite), axis=None)  
        qflow_sim_unique["temps_list"]=np.concatenate((temps_list,temps_list), axis=None)  
        qflow_sim_unique["station"]=np.concatenate((stat_list,[stat + "999" for stat in stat_list]), axis=None)  

        qflow_obs_an = pd.DataFrame()
        qflow_obs_an["qtot_obs"]=qtotobs_totalite
        qflow_obs_an["qbase_obs"]=qbaseobs_totalite
        qflow_obs_an["temps_list"]=temps_list
        qflow_obs_an["station"]=stat_list
        
        qflow_obs_unique = pd.DataFrame()
        qflow_obs_unique["q"]=np.concatenate((qtotobs_totalite,qbaseobs_totalite), axis=None)  
        qflow_obs_unique["temps_list"]=np.concatenate((temps_list,temps_list), axis=None)  
        qflow_obs_unique["station"]=np.concatenate((stat_list,[stat + "999" for stat in stat_list]), axis=None)  
        
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        chemin1=Path(self.output_dir,f"SP_{frequency}_tot_allstations_{self.timestamp}.png")
        chemin2=Path(self.output_dir,f"SP_{frequency}_base_allstations_{self.timestamp}.png")
        self.plot_streamflow_scatter_color(self.frequency,sim_qflow=qflow_sim_an, 
                                           obs_qflow=qflow_obs_an, 
                                           fig_path1=chemin1, 
                                           fig_path2=chemin2)
        
        self.plot_sim_vs_obs_yearly_streamflow(self.frequency,sim_qflow=qflow_sim_an, 
                                           obs_qflow=qflow_obs_an, 
                                           output_dir=self.output_dir, timestamp=self.timestamp)
                
        # Build Netcdf file with the two variables qobs ad qsim
        # Write out data to a new netCDF file with some attributes
        netcdf_obs_path = Path(output_dir, "qobs_netcdf.nc")
                     
        #sys.exit()
        self.netcdf_obs_path = netcdf_obs_path
        filename = netcdf_file(netcdf_obs_path, "w")
        
        if frequency=="year":
           year_unique=list(set(qflow_obs_unique["temps_list"]))
           times=[datetime.strptime(str(year_unique[a]), "%Y") for a in range(len(year_unique))]
        if frequency=="season":
            yeartrim_unique=list(set(qflow_obs_unique["temps_list"]))
            year_unique=[yeartrim_unique[a].split("_")[1] for a in range(len(yeartrim_unique))]
            trim_unique=[yeartrim_unique[a].split("_")[0] for a in range(len(yeartrim_unique))]
            times=[datetime.strptime(year_unique[a], "%Y") + timedelta(days=int(trim_unique[a])*91.25) for a in range(len(yeartrim_unique))]
        if frequency=="month":
            yeartrim_unique=list(set(qflow_obs_unique["temps_list"]))
            year_unique=[yeartrim_unique[a].split("_")[1] for a in range(len(yeartrim_unique))]
            trim_unique=[yeartrim_unique[a].split("_")[0] for a in range(len(yeartrim_unique))]
            times=[datetime.strptime(year_unique[a], "%Y") + timedelta(days=int(trim_unique[a])*30.42) for a in range(len(yeartrim_unique))]

        stations_unique=list(set(qflow_obs_unique["station"]))
        
        # Dimensions
        filename.createDimension("time", len(times))
        filename.createDimension("station", len(stations_unique))

        # Variables
        time = filename.createVariable("time", "i", ("time",))
        station = filename.createVariable("station", "f4", ("station",))
        streamflow = filename.createVariable(
            "streamflow", "f4", ("station", "time")
        )

        # Attributes
        time.units = "days since 1970-01-01 0:0:0"
        station.units = "stations names (no unit)"
        streamflow.units = "mm/year"

        # Populate the variables with data
        time_day = [
            (
                times[d]
                - datetime.strptime("1970-01-01", "%Y-%m-%d")
            ).days
            for d in range(len(times))
        ]
        time[:] = time_day
        station[:] = stations_unique
        for stat in range(len(stations_unique)):
            station=stations_unique[stat]
            subset=qflow_obs_unique[qflow_obs_unique["station"]==station]
            streamflow[stat, :] = subset['q']
        #print(time_day)
        filename.close()

        # Write out data to a new netCDF file with some attributes
        netcdf_sim_path = Path(output_dir,  f"qsim_netcdf_{self.timestamp}.nc")
        
        #if os.path.exists(netcdf_sim_path): 
        #    os.remove(netcdf_sim_path)
        
        self.netcdf_sim_path = netcdf_sim_path
        filename = netcdf_file(netcdf_sim_path, "w")
        
        # Dimensions
        filename.createDimension("time", len(times))
        filename.createDimension("station", len(stations_unique))

        # Variables
        time = filename.createVariable("time", "i", ("time",))
        station = filename.createVariable("station", "f4", ("station",))
        streamflow = filename.createVariable(
            "streamflow", "f4", ("station", "time")
        )
        # qbase_sim = filename.createVariable('qbase_sim', 'f4', ('time', 'station'))

        # Attributes
        time.units = "days since 1970-01-01 0:0:0"
        station.units = "stations names (no unit)"
        streamflow.units = "mm/month"
        
        # Populate the variables with data
        time[:] = time_day
        station[:] = stations_unique
        for stat in range(len(stations_unique)):
            station=stations_unique[stat]
            subset=qflow_sim_unique[qflow_sim_unique["station"]==station]
            streamflow[stat, :] = subset['q']

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

        
    def plot_streamflow_scatter_color(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        frequency: str(),
        sim_qflow: pd.DataFrame,
        obs_qflow: pd.DataFrame,
        fig_path1: Path,
        fig_path2: Path,
    ) :
        
        """
        Create a scatter plot comparing simulated total and base
        streamflow yearly values with observed values.
    
        """   
        self.frequency = frequency  
        #sim_qflow=qflow_sim_an
        #obs_qflow=qflow_obs_an
        
        # Join both dataframe in a single dataframe.
        yearly_qflow = sim_qflow.copy()
        yearly_qflow = yearly_qflow.reindex(index=obs_qflow.index)
        yearly_qflow.loc[obs_qflow.index, 'qtot_obs'] = obs_qflow['qtot_obs']
        yearly_qflow.loc[obs_qflow.index, 'qbase_obs'] = obs_qflow['qbase_obs']
        yearly_qflow.loc[obs_qflow.index, 'temps_list'] = obs_qflow['temps_list']
        if frequency=="year":
            yearly_qflow['temps_list1']=[yearly_qflow['temps_list'][a] for a in yearly_qflow.index]
           
        else : 
            yearly_qflow['temps_list1']=[yearly_qflow['temps_list'][a].split('_')[0] for a in yearly_qflow.index]
        
        septemporelle=list(dict.fromkeys(yearly_qflow.temps_list1))
        
        def generate_colors(x, colormap='Set1'):
            cmap = plt.get_cmap(colormap)
            colors = [cmap(i / (x - 1)) for i in range(x)]
            return colors
        
        if frequency=="year" : 
            years_nb=len(list(dict.fromkeys(yearly_qflow.temps_list1)))
            colorslist=generate_colors(years_nb)
            qmin=200
            qmax=1400
            qbasemax=500
            frequence="année"
            anchor_y=1.3
        if frequency=="season" : 
            colorslist=["mediumturquoise","green","gold","firebrick"]
            qmin=0
            qmax=700
            qbasemax=250
            frequence="saison"
            anchor_y=1.35
        if frequency=="month" : 
            colorslist=generate_colors(12)
            qmin=0
            qmax=600
            qbasemax=150
            anchor_y=1.35
            frequence="mois"

        # Figure Qtot
        fwidth, fheight = 6, 5
        fig, ax = plt.subplots()
        fig.set_size_inches(fwidth, fheight)
    
        left_margin = 1/fwidth
        right_margin = 1.3/fwidth
        top_margin = 0.5/fheight
        bot_margin = 1/fheight
        ax.set_position([left_margin, bot_margin,
                         1 - left_margin - right_margin,
                         1 - top_margin - bot_margin])
    
        xymin, xymax = qmin, qmax
        ax.axis([xymin, xymax, xymin, xymax])
        ax.set_ylabel('Débits Simulés (mm/an)', fontsize=16, labelpad=15)
        ax.set_xlabel('Débits Observés (mm/an)', fontsize=16, labelpad=15)
    
        ax.tick_params(axis='both', direction='out', labelsize=12)
    
        l1, = ax.plot([xymin, xymax], [xymin, xymax], '--', color='black', lw=1)
        
        for tr in range(len(septemporelle)):
            subset=yearly_qflow[yearly_qflow["temps_list1"]==septemporelle[tr]]
            l2, = ax.plot(subset['qtot_obs'],
                          subset['qtot_sim'],
                          '.',
                          color=colorslist[tr])
                
        # Plot the model fit stats.
        rmse_qtot = np.nanmean(
            (yearly_qflow['qtot_sim'] - yearly_qflow['qtot_obs'])**2)**0.5
        me_qtot = np.nanmean(
            yearly_qflow['qtot_sim'] - yearly_qflow['qtot_obs'])
    
        dx, dy = 3, -3
        offset = transforms.ScaledTranslation(dx/72, dy/72, fig.dpi_scale_trans)
        ax.text(0, 1, "RMSE débit total = %0.1f mm/an" % rmse_qtot,
                transform=ax.transAxes+offset, ha='left', va='top')
        dy += -12
        offset = transforms.ScaledTranslation(dx/72, dy/72, fig.dpi_scale_trans)
        ax.text(0, 1, "ME débit total = %0.1f mm/an" % me_qtot,
                transform=ax.transAxes+offset, ha='left', va='top')
    
        # Add a graph title.
        fig_title1=f"Débits total par {frequence}"
        offset = transforms.ScaledTranslation(0/72, 12/72, fig.dpi_scale_trans)
        ax.text(0.5, 1, fig_title1, fontsize=16, ha='center', va='bottom',
                transform=ax.transAxes+offset)
        # Add a legend.
        if frequency=="year" : 
            septemporelle_str=[str(septemporelle[tr]) for tr in range(len(septemporelle))]
            color = ["1:1"]
            colors=color + septemporelle_str
        if frequency=="season" : 
            colors = ["1:1", "Hiver","Printemps","Été","Automne"]
        if frequency=="month" : 
            colors = ["1:1", "Janvier","Février","Mars","Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octrobre", "Novembre", "Décembre"]
        legend = ax.legend(colors, numpoints=1, fontsize=10,
                           borderaxespad=0, loc='lower right', borderpad=0.5,
                           bbox_to_anchor=(anchor_y, 0), ncol=1)
        legend.draw_frame(False)
        fig.savefig(fig_path1,dpi=200)
        
        
        # Figure Qbase
        fwidth, fheight = 6, 5
        fig, ax = plt.subplots()
        fig.set_size_inches(fwidth, fheight)
    
        left_margin = 1/fwidth
        right_margin = 1.3/fwidth
        top_margin = 0.5/fheight
        bot_margin = 1/fheight
        ax.set_position([left_margin, bot_margin,
                         1 - left_margin - right_margin,
                         1 - top_margin - bot_margin])
    
        xymin, xymax = 0, qbasemax
        ax.axis([xymin, xymax, xymin, xymax])
        ax.set_ylabel('Débits Simulés (mm/an)', fontsize=16, labelpad=15)
        ax.set_xlabel('Débits Observés (mm/an)', fontsize=16, labelpad=15)
    
        ax.tick_params(axis='both', direction='out', labelsize=12)
    
        l1, = ax.plot([xymin, xymax], [xymin, xymax], '--', color='black', lw=1)
        
        for tr in range(len(septemporelle)):
            subset=yearly_qflow[yearly_qflow["temps_list1"]==septemporelle[tr]]
            l2, = ax.plot(subset['qbase_obs'],
                          subset['qbase_sim'],
                          '.',
                          color=colorslist[tr])
                
        # Plot the model fit stats.
        rmse_qtot = np.nanmean(
            (yearly_qflow['qbase_sim'] - yearly_qflow['qbase_obs'])**2)**0.5
        me_qtot = np.nanmean(
            yearly_qflow['qbase_sim'] - yearly_qflow['qbase_obs'])
    
        dx, dy = 3, -3
        offset = transforms.ScaledTranslation(dx/72, dy/72, fig.dpi_scale_trans)
        ax.text(0, 1, "RMSE débit total = %0.1f mm/an" % rmse_qtot,
                transform=ax.transAxes+offset, ha='left', va='top')
        dy += -12
        offset = transforms.ScaledTranslation(dx/72, dy/72, fig.dpi_scale_trans)
        ax.text(0, 1, "ME débit total = %0.1f mm/an" % me_qtot,
                transform=ax.transAxes+offset, ha='left', va='top')
    
        # Add a graph title.
        fig_title2=f"Débits de base par {frequence}"
        offset = transforms.ScaledTranslation(0/72, 12/72, fig.dpi_scale_trans)
        ax.text(0.5, 1, fig_title2, fontsize=16, ha='center', va='bottom',
                transform=ax.transAxes+offset)
        
        # Add a legend.
        if frequency=="year" : 
            septemporelle_str=[str(septemporelle[tr]) for tr in range(len(septemporelle))]
            color = ["1:1"]
            colors=color + septemporelle_str
        if frequency=="season" : 
            colors = ["1:1", "Hiver","Printemps","Été","Automne"]
        if frequency=="month" : 
            colors = ["1:1", "Janvier","Février","Mars","Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octrobre", "Novembre", "Décembre"]
            
        legend = ax.legend(colors, numpoints=1, fontsize=10,
                           borderaxespad=0, loc='lower right', borderpad=0.5,
                           bbox_to_anchor=(anchor_y, 0), ncol=1)
        legend.draw_frame(False)
    
        fig.savefig(fig_path2,dpi=200)
    
        return
          
    def plot_sim_vs_obs_yearly_streamflow(
            # numpydoc ignore=EX01,SA01,ES01
            self,
            frequency: str(),
            sim_qflow: pd.DataFrame,
            obs_qflow: pd.DataFrame,
            output_dir: Path,
            timestamp: str()
        ) :
               
        """
        Plot simulated vs observed yearly total and base streamflow.
    
        Parameters
        ----------
        figname : str, optional
            The abolute path of the file where to save the figure to disk.
            Note that the format of the file is inferred from the extension of
            "figname".
        """
        self.timestamp = timestamp 
        self.output_dir = output_dir 
        self.frequency = frequency  
        # Join both dataframe in a single dataframe.
        yearly_qflow = sim_qflow.copy()
        yearly_qflow = yearly_qflow.reindex(index=obs_qflow.index)
        yearly_qflow.loc[obs_qflow.index, 'qtot_obs'] = obs_qflow['qtot_obs']
        yearly_qflow.loc[obs_qflow.index, 'qbase_obs'] = obs_qflow['qbase_obs']
        yearly_qflow.loc[obs_qflow.index, 'temps_list'] = obs_qflow['temps_list']
        
        if frequency=="year":
            yearly_qflow['year']=[yearly_qflow['temps_list'][a] for a in yearly_qflow.index]
            yearly_qflow['date']=[datetime.strptime(str(yearly_qflow.loc[a,'year'])+'/01/01', '%Y/%m/%d') for a in yearly_qflow.index]
            yearly_qflow['temps_list1']=yearly_qflow['date']

        if frequency=="season":
            mois_deb_saison=[1,4,7,10]
            nom_saison=['Hiver', 'Printemps', 'Été', 'Automne']
            yearly_qflow['year']=[int(yearly_qflow['temps_list'][a].split('_')[1]) for a in yearly_qflow.index]
            yearly_qflow['month']=[int(yearly_qflow['temps_list'][a].split('_')[0]) for a in yearly_qflow.index]
            yearly_qflow['temps_list_ecrite']=[ nom_saison[yearly_qflow.loc[a,'month']-1] + ' ' + str(yearly_qflow.loc[a,'year']) for a in yearly_qflow.index]
            yearly_qflow['month']=[mois_deb_saison[yearly_qflow.loc[a,'month']-1] for a in yearly_qflow.index]
            
            # # repérage des mois de décembre pour enlever 1 an à l'année correspondante
            # rep_dec=np.where(yearly_qflow['month']==12)
            # yearly_qflow.loc[rep_dec[0],'year']=yearly_qflow.loc[rep_dec[0],'year']-1

            yearly_qflow['date']=[datetime.strptime(str(yearly_qflow.loc[a,'year'])+'/'+str(yearly_qflow.loc[a,'month'])+'/01', '%Y/%m/%d') for a in yearly_qflow.index]
            yearly_qflow['temps_list1']=yearly_qflow['date']

        if frequency=="month":
            yearly_qflow['year']=[yearly_qflow['temps_list'][a].split('_')[1] for a in yearly_qflow.index]
            yearly_qflow['month']=[yearly_qflow['temps_list'][a].split('_')[0] for a in yearly_qflow.index]
            yearly_qflow['date']=[datetime.strptime(yearly_qflow.loc[a,'year']+'/'+yearly_qflow.loc[a,'month']+'/01', '%Y/%m/%d') for a in yearly_qflow.index]
            yearly_qflow['temps_list1']=yearly_qflow['date']
        
        
        septemporelle=list(dict.fromkeys(yearly_qflow.temps_list1))
        stations=list(set(obs_qflow["station"]))
        

        for stat in range(len(stations)) :
            
            subset_yearly_qflow=yearly_qflow[yearly_qflow.station==stations[stat]]
            
            if frequency=="year" : 
                septemporelle_str=[septemporelle[tr] for tr in range(len(septemporelle))]
                qmin=0
                qmax=1400
                frequence_tit="annuels"
                larg=7
            if frequency=="season" : 
                qmin=0
                qmax=700
                frequence_tit="saisonniers"
                larg=10
            if frequency=="month" : 
                qmin=0
                qmax=600
                frequence_tit="mensuels"
                larg=15
    
            fwidth, fheight = larg, 5.5
            fig, ax = plt.subplots()
            fig.set_size_inches(fwidth, fheight)
        
            left_margin = 1.5/fwidth
            right_margin = 0.25/fwidth
            top_margin = 0.5/fheight
            bot_margin = 0.7/fheight
            ax.set_position([left_margin, bot_margin,
                             1 - left_margin - right_margin,
                             1 - top_margin - bot_margin])
        
            l1, = ax.plot(subset_yearly_qflow['temps_list1'],subset_yearly_qflow['qtot_obs'],  
                          clip_on=True, lw=2, linestyle='--',  color='#3690c0')
        
            l2, = ax.plot(subset_yearly_qflow['temps_list1'],subset_yearly_qflow['qtot_sim'], 
                          clip_on=True, lw=2, linestyle='-', color='#034e7b')
        
            # Streamflow base
            l3, = ax.plot(subset_yearly_qflow['temps_list1'],subset_yearly_qflow['qbase_obs'], 
                          lw=2, linestyle='--', clip_on=True, color='#ef6548')
        
            l4, = ax.plot(subset_yearly_qflow['temps_list1'],subset_yearly_qflow['qbase_sim'],  
                          clip_on=True, lw=2, linestyle='-', color='#990000')
            
        
            ax.tick_params(axis='both', direction='out', labelsize=12)
            ax.set_ylabel('Débit par unité de surface\n(mm/an)',
                          fontsize=16, labelpad=10)
            ax.set_xlabel("Date", fontsize=16, labelpad=10)
            ax.axis(ymin=qmin, ymax=qmax)
            ax.grid(axis='y', color=[0.35, 0.35, 0.35], ls='-', lw=0.5)
            ax.grid(axis='x', color=[0.35, 0.35, 0.35], ls='-', lw=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
            myFmt = DateFormatter("%Y")
            ax.xaxis.set_major_formatter(myFmt)
            
            #ax.autofmt_xdate()
            #ax.axis(axis_)
        
            lines = [l1, l2, l3, l4]
            labels = ["Débit total observé", "Débit total simulé",
                      "Débit base observé", "Débit base simulé"]
            legend = ax.legend(lines, labels, numpoints=1, fontsize=12,
                                borderaxespad=0, loc='upper left', borderpad=0.5,
                                bbox_to_anchor=(0, 1), ncol=2)
            legend.draw_frame(False)
        
            # Add a graph title.
            fig_title=f"Débits {frequence_tit} à la station {stations[stat]}"
            offset = transforms.ScaledTranslation(0/72, 12/72, fig.dpi_scale_trans)
            ax.text(0.5, 1, fig_title, fontsize=16, ha='center', va='bottom',
                    transform=ax.transAxes+offset)
            
            folder_path = Path(self.output_dir,f'{stations[stat]}')
            folder_path.mkdir(parents=True, exist_ok=True)
            
            chemin=Path(folder_path,f"obsVSsim_{frequence_tit}_station{stations[stat]}_{self.timestamp}.png")
            # Save figure to file.
            fig.savefig(chemin)
    
        return fig