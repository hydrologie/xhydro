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

        helpm = HelpManager(
            self.project_dir,
            path_to_grid=str(self.project_dir) + "/input_grid_ex.csv",
            path_to_precip=str(self.project_dir) + "/precip_input_data.csv",
            path_to_airtemp=str(self.project_dir) + "/airtemp_input_data.csv",
            path_to_solrad=str(self.project_dir) + "/solrad_input_data.csv",
        )

        cellnames = helpm.grid.index[helpm.grid["Bassin"] == 1]

        helpm.calc_help_cells(
            path_to_hdf5=str(self.project_dir) + "/help_example.out",
            cellnames=cellnames,
            tfsoil=-3,
            sf_edepth=0.15,
            sf_ulai=1,
            sf_cn=1.15,
        )

        return
