"""Hydrological modelling framework."""

import inspect
import re
import warnings
from copy import deepcopy
from os import PathLike
from pathlib import Path

import numpy as np
import xarray as xr
import xclim as xc
import xscen as xs
from packaging.version import Version

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
    convert_calendar_missing: float | str | dict = np.nan,
    save_as: str | PathLike | None = None,
    **kwargs,
) -> tuple[xr.Dataset, dict]:
    r"""Reformat CF-compliant meteorological data for use in hydrological models.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset containing the meteorological data. See the "Notes" section for more information on the expected format.
    model : str
        The name of the hydrological model to use.
        Currently supported models are: "Hydrotel".
    convert_calendar_missing : float, str, dict, optional
        Upon conversion of the calendar, missing values will be filled with this value. Default is np.nan.
        If the value is 'interpolate', the new dates will be linearly interpolated over time.
        A dictionary can be used to specify a different fill value for each variable.
        Keys should be the standard names of the variables (first entry in the list of names in the "Notes" section).
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
    The input dataset should be CF-compliant.
    This function will attempt to detect the variables based on the standard_name attribute, the cell_methods attribute, or the variable name
    (AMIP column) taken from https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html.

    Specifically:

    - If using 1D time series, the station dimension should have an attribute `cf_role` set to "timeseries_id".
    - Units don't need to be canonical, but they should be convertible to the expected units and be understood by `xclim`.
    - The following attempts will be made to detect the variables:
        - Longitude:
            - standard_name: "longitude"
            - variable name: "lon", "longitude"
        - Latitude:
            - standard_name: "latitude"
            - variable name: "lat", "latitude"
        - Elevation:
            - standard_name: "surface_altitude"
            - variable name: "orog", "z", "altitude", "elevation", "height"
        - Precipitation:
            - standard_name: "*precipitation*" (e.g. "lwe_thickness_of_precipitation_amount")
            - variable name: "pr", "precip", "precipitation"
        - Maximum temperature:
            - standard_name: "air_temperature"
            - cell_methods: "time: maximum"
            - variable name: "tasmax", "tmax", "temperature_max"
        - Minimum temperature:
            - standard_name: "air_temperature"
            - cell_methods: "time: minimum"
            - variable name: "tasmin", "tmin", "temperature_min"

    Hydrotel requires the following variables: ["longitude", "latitude", "altitude", "time", "tasmax", "tasmin", "pr"].
    """
    ds = ds.copy()

    # Detect and rename variables if necessary
    variables = {
        "longitude": {"standard_name": "longitude", "names": ["lon", "longitude"]},
        "latitude": {"standard_name": "latitude", "names": ["lat", "latitude"]},
        "elevation": {
            "standard_name": "surface_altitude",
            "names": ["orog", "z", "altitude", "elevation", "height"],
        },
        "pr": {
            "standard_name": ".*precipitation.*",
            "names": ["pr", "precip", "precipitation"],
        },
        "tasmax": {
            "standard_name": "air_temperature",
            "cell_methods": "time: max.*",
            "names": ["tasmax", "tmax", "temperature_max"],
        },
        "tasmin": {
            "standard_name": "air_temperature",
            "cell_methods": "time: min.*",
            "names": ["tasmin", "tmin", "temperature_min"],
        },
    }
    for attributes in variables.values():
        names = attributes.pop("names")
        ds = _detect_variable(ds, attributes, names, return_ds=True)

    def _is_rate(u):
        q = xc.core.units.str2pint(u)
        return q.dimensionality.get("[time]", 0) < 0

    if model == "Hydrotel":
        if not all(
            v in ds for v in ["lon", "lat", "orog", "time", "tasmax", "tasmin", "pr"]
        ):
            raise ValueError(
                f"The dataset is missing the following required variables for Hydrotel: "
                f"{set(ds.variables).difference(['lon', 'lat', 'orog', 'time', 'tasmax', 'tasmin', 'pr'])}."
            )
        # Convert units
        if _is_rate(ds["pr"].attrs.get("units", "")):
            ds["pr"] = xc.units.rate2amount(ds["pr"])
        ds["pr"] = xc.units.convert_units_to(ds["pr"], "mm", context="hydro")
        ds["tasmax"] = xc.units.convert_units_to(ds["tasmax"], "degC")
        ds["tasmin"] = xc.units.convert_units_to(ds["tasmin"], "degC")
        ds["orog"] = xc.units.convert_units_to(ds["orog"], "m")

        # Ensure that longitude is in the range [-180, 180]
        # This tries guessing if lons are wrapped around at 180+ but without much information, this might not be true
        if np.min(ds["lon"]) >= -180 and np.max(ds["lon"]) <= 180:
            pass
        elif np.min(ds["lon"]) >= 0 and np.max(ds["lon"]) <= 360:
            warnings.warn(
                "Longitude values appear to be in the range [0, 360]. They will be converted to [-180, 180]."
            )
            with xr.set_options(keep_attrs=True):
                ds["lon"] = ds["lon"] - 180

        # Convert calendar
        # FIXME: xscen 0.9.1 still calls the old xclim function. This will be fixed in the next release.
        if Version(xs.__version__) > Version("0.9.10"):
            convert_calendar_kwargs = {"calendar": "standard", "use_cftime": False}
        else:
            convert_calendar_kwargs = {"target": "default"}
        if isinstance(convert_calendar_missing, dict):
            missing_by_var = convert_calendar_missing
        else:
            missing_by_var = None
            convert_calendar_kwargs["missing"] = convert_calendar_missing
            if (
                ds.time.dt.calendar
                not in ["standard", "gregorian", "proleptic_gregorian"]
                and convert_calendar_missing is np.nan
            ):
                warnings.warn(
                    f"The calendar '{ds.time.dt.calendar}' needs to be converted to 'standard', but 'convert_calendar_missing' is set to np.nan. "
                    f"NaNs will need to be filled manually before running Hydrotel."
                )
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
        if (len(ds["lat"].dims) == 2) or ("lat" in ds.dims):
            mask = ~ds.pr.isnull().all(
                dim="time"
            )  # FIXME: The mask can be written as an argument once we drop xscen <=0.9.1.
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
            "LATITUDE_NAME": "lat",
            "LONGITUDE_NAME": "lon",
            "ELEVATION_NAME": "orog",
            "TIME_NAME": ds.cf["time"].name,
            "TMIN_NAME": "tasmin",
            "TMAX_NAME": "tasmax",
            "PRECIP_NAME": "pr",
        }

        out = (ds, cfg)
        if save_as:
            Path(save_as).parent.mkdir(parents=True, exist_ok=True)
            with Path(save_as).with_suffix(".nc.config").open("w") as f:
                for k, v in cfg.items():
                    f.write(f"{k}; {v}\n")
            ds.to_netcdf(Path(save_as).with_suffix(".nc"), **kwargs)

    else:
        raise NotImplementedError(f"The model '{model}' is not recognized.")

    return out


def _detect_variable(
    ds: xr.Dataset, attributes: dict, names: list, return_ds: bool = False
) -> xr.Dataset | str:
    """Find a variable in the dataset based on its attributes or name."""
    out = []
    # Search for variables based on attributes
    [
        out.append(v)
        for v in list(ds.variables.keys())
        if all(re.fullmatch(attributes[a], ds[v].attrs.get(a, "")) for a in attributes)
    ]
    # Search for variables based on names
    [out.append(v) for v in names if v in ds.variables]

    out = list(set(out))
    if len(out) > 1:
        raise ValueError(f"Multiple variables found for {names}: {out}.")

    # Return a string
    if return_ds is False:
        if len(out) == 0:
            return ""
        elif len(out) == 1:
            return out[0]

    # Rename the variable in the dataset
    elif return_ds is True:
        if len(out) == 0:
            return ds
        elif len(out) == 1:
            return ds.rename({out[0]: names[0]})
