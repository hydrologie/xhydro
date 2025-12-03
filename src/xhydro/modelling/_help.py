# numpydoc ignore=EX01,SA01,ES01
"""Class to handle Hydrotel simulations."""

import os
from pathlib import Path

import xarray as xr
from pyhelp.managers import HelpManager

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
    """

    def __init__(
        # numpydoc ignore=EX01,SA01,ES01
        self,
        project_dir: str | os.PathLike,
    ):
        """
        Initialize the HELP simulation.

        Parameters
        ----------
        project_dir : str or os.PathLike
            Path to the project folder (including inputs file, shell script and R script).
        """
        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValueError("The project folder does not exist.")

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
            path_to_hdf5=self.project_dir / "help_example.out",
            cellnames=cellnames,
            tfsoil=-3,
            sf_edepth=0.15,
            sf_ulai=1,
            sf_cn=1.15,
        )
        self.output_help = output_help

        # """Standardize the outputs"""
        # self._standardise_outputs(**(xr_open_kwargs_out or {}))

        """Get streamflow """
        self.get_streamflow()

        return

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
        self.output_help.save_to_csv(str(self.project_dir) + "/help_example_yearly.csv")

        return

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
        # Plot some results.
        self.output_help.plot_area_monthly_avg(fig_title="PyHELP Example")
        self.output_help.plot_area_yearly_avg(fig_title="PyHELP Example")
        self.output_help.plot_area_yearly_series(fig_title="PyHELP Example")
        return
