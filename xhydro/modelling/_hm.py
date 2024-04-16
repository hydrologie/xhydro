"""Hydrological model class."""

from abc import ABC, abstractmethod

import xarray as xr


class HydrologicalModel(ABC):
    """Hydrological model class.

    This class is a wrapper for the different hydrological models that can be used in xhydro.
    """

    @abstractmethod
    def run(self, **kwargs) -> xr.Dataset:
        r"""Run the hydrological model.

        Parameters
        ----------
        \*\*kwargs : dict
            Additional keyword arguments for the hydrological model.

        Returns
        -------
        xr.Dataset
            Simulated streamflow from the hydrological model, in xarray Dataset format.
        """
        pass

    @abstractmethod
    def get_inputs(self, **kwargs) -> xr.Dataset:
        r"""Get the input data for the hydrological model.

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

    @abstractmethod
    def get_streamflow(self, **kwargs) -> xr.Dataset:
        r"""Get the simulated streamflow data from the hydrological model.

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
