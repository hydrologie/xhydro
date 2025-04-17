"""Implement the ravenpy handler class for emulating raven models in ravenpy."""

import os
import tempfile

import numpy as np
import xarray as xr

try:
    import ravenpy.config.emulators
    from ravenpy import OutputReader
    from ravenpy.config import commands as rc
    from ravenpy.ravenpy import run
except ImportError as e:
    run = None

from ._hm import HydrologicalModel

__all__ = ["RavenpyModel"]


class RavenpyModel(HydrologicalModel):
    r"""Implement the RavenPy model class to build and run ravenpy models.

    Parameters
    ----------
    model_name : {"Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"}
        The name of the ravenpy model to run.
    parameters : np.ndarray or list of float
        The model parameters for simulation or calibration.
    drainage_area : np.ndarray or float
        The watershed drainage area, in km².
    elevation : np.ndarray or float
        The elevation of the watershed, in meters.
    latitude : np.ndarray or float
        The latitude of the watershed centroid.
    longitude : np.ndarray or float
        The longitude of the watershed centroid.
    start_date : dt.datetime
        The first date of the simulation.
    end_date : dt.datetime
        The last date of the simulation.
    meteo_file : str or os.PathLike
        The path to the file containing the observed meteorological data.
    data_type : sequence of str
        # FIXME: This does not accept a dict, but a sequence of str. Please update the docstring.
        The dictionary necessary to tell raven which variables are being fed such that it can adjust its processes internally.
    alt_names_meteo : dict
        A dictionary that allows users to change the names of meteo variables of their dataset to cf-compliant names.
    meteo_station_properties : dict
        The properties of the weather stations providing the meteorological data. Used to adjust weather according to
        differences between station and catchment elevations (adiabatic gradients, etc.).
    workdir : str or  os.PathLike
        Path to save the .rv files and model outputs.
    rain_snow_fraction : str
        The method used by raven to split total precipitation into rain and snow.
    evaporation : str
        The evapotranspiration function used by raven.
    \*\*kwargs : dict
        Dictionary of other parameters to feed to raven according to special cases and that are allowed by the raven
        documentation.
    """

    def __init__(
        self,
        model_name: str,
        parameters: np.ndarray | list[float],
        drainage_area: np.ndarray | float,
        elevation: np.ndarray | float,
        latitude,
        longitude,
        start_date,
        end_date,
        meteo_file,
        data_type,
        alt_names_meteo,
        meteo_station_properties,
        workdir: str | os.PathLike | None = None,
        rain_snow_fraction="RAINSNOW_DINGMAN",
        evaporation="PET_PRIESTLEY_TAYLOR",
        **kwargs,
    ):
        if run is None:
            raise ImportError(
                "RavenPy is not installed. Please install it to use this class."
            )

        if workdir is None:
            workdir = tempfile.mkdtemp(prefix=model_name)
        self.workdir = workdir

        self.model_simulations = None
        self.qsim = None

        # Create HRU object for ravenpy based on catchment properties
        hru = dict(
            area=drainage_area,
            elevation=elevation,
            latitude=latitude,
            longitude=longitude,
            hru_type="land",
        )

        # Create the emulator configuration
        self.default_emulator_config = dict(
            HRUs=[hru],
            params=parameters,
            StartDate=start_date,
            EndDate=end_date,
            # ObservationData=[  # This is here for reference, but not used in this implementation.
            #     rc.ObservationData.from_nc(qobs_path, alt_names=alt_names_flow)
            # ],
            Gauge=[
                rc.Gauge.from_nc(
                    meteo_file,  # File path to the meteorological data
                    data_type=data_type,  # List of all the variables in the file
                    alt_names=alt_names_meteo,
                    # Mapping between the names of the required variables and those in the file.
                    data_kwds=meteo_station_properties,
                )
            ],
            rain_snow_fraction=rain_snow_fraction,
            Evaporation=evaporation,
            **kwargs,
        )
        self.meteo_file = meteo_file
        self.model_name = model_name

    def run(self) -> str | xr.Dataset:
        """Run the ravenpy hydrological model and return simulated streamflow.

        Returns
        -------
        xr.Dataset
            The simulated streamflow from the selected ravenpy model.
        """
        default_emulator_config = self.default_emulator_config
        model_name = self.model_name
        workdir = self.workdir

        if model_name not in [
            "Blended",
            "GR4JCN",
            "HBVEC",
            "HMETS",
            "HYPR",
            "Mohyse",
            "SACSMA",
        ]:
            raise ValueError("The selected model is not available in RavenPy.")

        # Need to remove qobs as pydantic forbids extra inputs...
        if "qobs" in default_emulator_config:
            default_emulator_config.pop("qobs")

        self.model = getattr(ravenpy.config.emulators, model_name)(
            **default_emulator_config
        )
        self.model.write_rv(workdir=workdir)

        outputs_path = run(modelname="raven", configdir=workdir, overwrite=True)
        outputs = OutputReader(path=outputs_path)

        with xr.open_dataset(outputs.files["hydrograph"]) as ds:
            qsim = ds.q_sim.to_dataset(name="qsim").rename({"qsim": "q"})

            if "nbasins" in qsim.dims:
                qsim = qsim.squeeze()

            self.qsim = qsim

        self.model_simulations = outputs

        return qsim

    def get_streamflow(self):
        """Return the precomputed streamflow.

        Returns
        -------
        xr.Dataset
            The simulated streamflow from the selected ravenpy model.
        """
        return self.qsim

    def get_inputs(self) -> xr.Dataset:
        """Return the inputs used to run the ravenpy model.

        Returns
        -------
        xr.Dataset
            The observed meteorological data used to run the ravenpy model simulation.
        """
        ds = xr.open_dataset(self.meteo_file)

        start_date = self.default_emulator_config["StartDate"]
        end_date = self.default_emulator_config["EndDate"]
        ds = ds.sel(time=slice(start_date, end_date))

        return ds
