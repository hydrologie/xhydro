"""Hydrological modelling framework."""

import inspect
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
import xclim as xc
import xscen as xs

from ._hydrotel import Hydrotel
from ._ravenpy_models import RavenpyModel
from ._simplemodels import DummyModel

__all__ = ["format_input", "get_hydrological_model_inputs", "hydrological_model"]


def hydrological_model(model_config):
    """Initialize an instance of a hydrological model.

    Parameters
    ----------
    model_config : dict
        A dictionary containing the configuration for the hydrological model.
        Must contain a key "model_name" with the name of the model to use: "Hydrotel".
        The required keys depend on the model being used. Use the function
        `get_hydrological_model_inputs` to get the required keys for a given model.

    Returns
    -------
    Hydrotel or DummyModel
        An instance of the hydrological model.
    """
    if "model_name" not in model_config:
        raise ValueError("The model name must be provided in the model configuration.")

    model_config = deepcopy(model_config)
    model_name = model_config["model_name"]

    if model_name == "Hydrotel":
        model_config.pop("model_name")
        return Hydrotel(**model_config)

    elif model_name == "Dummy":
        model_config.pop("model_name")
        return DummyModel(**model_config)

    elif model_name in [
        "Blended",
        "GR4JCN",
        "HBVEC",
        "HMETS",
        "HYPR",
        "Mohyse",
        "SACSMA",
    ]:
        return RavenpyModel(**model_config)
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")


def get_hydrological_model_inputs(
    model_name, required_only: bool = False
) -> tuple[dict, str]:
    """Get the required inputs for a given hydrological model.

    Parameters
    ----------
    model_name : str
        The name of the hydrological model to use.
        Currently supported models are: "Hydrotel".
    required_only : bool
        If True, only the required inputs will be returned.

    Returns
    -------
    dict
        A dictionary containing the required configuration for the hydrological model.
    str
        The documentation for the hydrological model.
    """
    if model_name == "Dummy":
        model = DummyModel
    elif model_name == "Hydrotel":
        model = Hydrotel
    elif model_name in [
        "Blended",
        "GR4JCN",
        "HBVEC",
        "HMETS",
        "HYPR",
        "Mohyse",
        "SACSMA",
    ]:
        model = RavenpyModel
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")

    all_config = inspect.getfullargspec(model.__init__).annotations
    if required_only:
        all_config = {
            k: v
            for k, v in all_config.items()
            if k in inspect.getfullargspec(model.__init__).args
        }

    # Add the model name to the configuration
    all_config = {"model_name": model_name, **all_config}

    return all_config, model.__doc__


def format_input(
    ds: xr.Dataset,
    model: str,
    convert_calendar_missing: Optional[Union[float, str, dict]] = None,
    save_as: Optional[Union[str, PathLike]] = None,
    **kwargs,
) -> tuple[xr.Dataset, dict]:
    r"""Reformat CF-compliant meteorological data for use in hydrological models.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset containing the meteorological data. See the "Notes" section for details.
    model : str
        The name of the hydrological model to use.
        Currently supported models are: "Hydrotel".
    convert_calendar_missing : float, str, dict, optional
        Upon conversion of the calendar, missing values will be filled with this value. Keep as None to keep missing values.
        If the value is 'interpolate', the new dates will be linearly interpolated over time.
        A dictionary can be used to specify a different fill value for each variable.
    save_as : str, optional
        Where to save the reformatted data. If None, the data will not be saved.
        This can be useful when multiple files are needed for a single model run (e.g. Hydrotel needs a configuration file).
    \*\*kwargs : dict
        Additional keyword arguments to pass to the save function.

    Returns
    -------
    tuple[xr.Dataset, dict]
        The reformatted dataset and, if applicable, the configuration for the model.

    Notes
    -----
    The input dataset should be CF-compliant or at the very least have latitude, longitude and vertical coordinates
    recognized by `ds.cf`. If using 1D time series, the station dimension should have an attribute `cf_role` set to
    "timeseries_id". Units for variables should be understood by `xclim`. As there is no standard for variable names
    in CF, the following hard-coded names are expected:
    - "pr": precipitation
    - "tasmax": maximum temperature
    - "tasmin": minimum temperature

    Hydrotel requires the following variables: ["pr", "tasmax", "tasmin", "time", "latitude", "longitude", "vertical"].
    """
    if model == "Hydrotel":
        # Convert units
        ds["pr"] = xc.units.convert_units_to(ds["pr"], "mm", context="hydro")
        ds["tasmax"] = xc.units.convert_units_to(ds["tasmax"], "degC")
        ds["tasmin"] = xc.units.convert_units_to(ds["tasmin"], "degC")
        ds[ds.cf["vertical"].name] = xc.units.convert_units_to(
            ds[ds.cf["vertical"].name], "m"
        )

        # Convert calendar
        # FIXME: xscen 0.9.1 still calls the old xclim function. This will be fixed in the next release.
        if xs.__version__ > "0.9.1":
            convert_calendar_kwargs = {"target": "standard", "use_cftime": False}
        else:
            convert_calendar_kwargs = {"target": "default"}
        if isinstance(convert_calendar_missing, dict):
            missing_by_var = convert_calendar_missing
        else:
            missing_by_var = None
            convert_calendar_kwargs["missing"] = convert_calendar_missing
        ds = xs.utils.clean_up(
            ds,
            convert_calendar_kwargs=convert_calendar_kwargs,
            missing_by_var=missing_by_var,
        )

        # Time units in Hydrotel must be exactly "days since 1970-01-01 00:00:00"
        new_time = (
            (ds["time"].values - np.datetime64("1970-01-01 00:00:00"))
            .astype("timedelta64[D]")
            .astype(int)
        )
        ds["time"] = xr.DataArray(
            new_time,
            dims="time",
            coords={"time": new_time},
            attrs={"units": "days since 1970-01-01 00:00:00"},
        )

        # Hydrotel is faster with 1D time series
        if (len(ds.cf["latitude"].dims) == 2) or (ds.cf["latitude"].name in ds.dims):
            mask = ~ds.pr.isnull().all(
                dim="time"
            )  # FIXME: xscen 0.9.1 currently requires a separate mask.
            ds = xs.utils.stack_drop_nans(ds, mask=mask, new_dim="station")

            # Add station ID
            ds = ds.assign_coords(station=("station", np.arange(len(ds.station))))
            ds["station"].attrs = {
                "long_name": "Station data",
                "cf_role": "timeseries_id",
            }

        station_dim = ds.cf.cf_roles["timeseries_id"][0]

        cfg = {
            "TYPE (STATION/GRID/GRID_EXTENT)": "STATION",
            "STATION_DIM_NAME": station_dim,
            "LATITUDE_NAME": ds.cf["latitude"].name,
            "LONGITUDE_NAME": ds.cf["longitude"].name,
            "ELEVATION_NAME": ds.cf["vertical"].name,
            "TIME_NAME": ds.cf["time"].name,
            "TMIN_NAME": "tasmin",
            "TMAX_NAME": "tasmax",
            "PRECIP_NAME": "pr",
        }

        out = (ds, cfg)
        if save_as:
            with Path(save_as).with_suffix(".nc.config").open("w") as f:
                for k, v in cfg.items():
                    f.write(f"{k}; {v}\n")
            ds.to_netcdf(Path(save_as).with_suffix(".nc"), **kwargs)

    else:
        raise NotImplementedError(f"The model '{model}' is not recognized.")

    return out
