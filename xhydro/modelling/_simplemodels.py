"""Simple hydrological models."""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ._hm import HydrologicalModel


class DummyModel(HydrologicalModel):
    """Dummy model.

    Dummy model to use as a placeholder for testing purposes.

    Parameters
    ----------
    precip : xr.DataArray
        Daily precipitation in mm.
    temperature : xr.DataArray
        Daily average air temperature in °C.
    drainage_area : float
        Drainage area of the catchment.
    parameters : np.ndarray
        Model parameters, length 3.
    qobs : np.ndarray, optional
        Observed streamflow in m3/s.
    """

    def __init__(
        self,
        precip: xr.DataArray,
        temperature: xr.DataArray,
        drainage_area: float,
        parameters: np.ndarray,
        qobs: np.ndarray = None,
    ):
        """Initialize the DummyModel object."""
        warnings.warn(
            "This is a dummy model. Do not use for real applications.", UserWarning
        )
        self.precip = precip
        self.temperature = temperature
        self.drainage_area = drainage_area
        self.parameters = parameters
        self.qsim = None
        self.qobs = qobs

    def run(self) -> xr.Dataset:
        """Run the Dummy model.

        Returns
        -------
        xr.Dataset
            Simulated streamflow from the Dummy model, in xarray Dataset format.
        """
        # Make sure that the input data is in the correct dtype
        self.precip = self.precip.astype("float64")
        self.temperature = self.temperature.astype("float64")
        self.drainage_area = float(self.drainage_area)
        self.parameters = np.array(self.parameters, dtype="float64")

        out = (
            (
                self.precip * self.parameters[0]
                + abs(self.temperature) * self.parameters[1]
            )
            * self.parameters[2]
            * self.drainage_area
        )
        time = pd.date_range("2024-01-01", periods=len(self.precip))

        qsim = xr.DataArray(
            out,
            coords=[time],
            dims=["time"],
            attrs={
                "description": "Streamflow simulated by the Dummy Model",
                "standard_name": "outgoing_water_volume_transport_along_river_channel",
                "long_name": "Streamflow",
                "units": "m3 s-1",
            },
        )
        qsim.name = "streamflow"
        qsim = qsim.to_dataset()
        self.qsim = qsim

        return qsim

    def get_inputs(self) -> xr.Dataset:
        """Return the input data for the Dummy model.

        Returns
        -------
        xr.Dataset
            Input data for the Dummy model, in xarray Dataset format.
        """
        return xr.Dataset(
            data_vars=dict(
                precip=(["time"], self.precip),
                temperature=(["time"], self.temperature),
            ),
            coords=dict(time=pd.date_range("2024-01-01", periods=len(self.precip))),
            attrs=dict(description="input data for the Dummy Model"),
        )

    def get_streamflow(self) -> xr.Dataset:
        """Return the simulated streamflow from the Dummy model.

        Returns
        -------
        xr.Dataset
            Simulated streamflow from the Dummy model, in xarray Dataset format.
        """
        if self.qsim is None:
            return self.run()
        else:
            return self.qsim
