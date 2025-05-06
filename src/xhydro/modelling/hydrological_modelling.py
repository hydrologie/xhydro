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
from xscen.utils import change_units, clean_up, stack_drop_nans

from ._hydrotel import Hydrotel
from ._ravenpy_models import RavenpyModel

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
    Hydrotel or RavenpyModel
        An instance of the hydrological model.
    """
    if "model_name" not in model_config:
        raise ValueError("The model name must be provided in the model configuration.")

    model_config = deepcopy(model_config)
    model_name = model_config["model_name"]

    if model_name == "Hydrotel":
        model_config.pop("model_name")
        return Hydrotel(**model_config)

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
        Currently supported models are ["Hydrotel", "Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"].
    required_only : bool
        If True, only the required inputs will be returned.

    Returns
    -------
    dict
        A dictionary containing the required configuration for the hydrological model.
    str
        The documentation for the hydrological model.
    """
    if model_name == "Hydrotel":
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


def format_input(  # noqa: C901
    ds: xr.Dataset,
    model: str,
    convert_calendar_missing: float | str | dict | bool = np.nan,
    save_as: str | PathLike | None = None,
    **kwargs,
) -> tuple[xr.Dataset, dict]:
    r"""Reformat CF-compliant meteorological data for use in hydrological models. See the "Notes" section for important details.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset containing the meteorological data. See the "Notes" section for more information on the expected format.
    model : str
        The name of the hydrological model to use.
        Currently supported models are:
        - "Hydrotel", "Raven" (which is an alias for all RavenPy models), "Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA".
    convert_calendar_missing : float | str | dict | bool, optional
        The value to use for missing values when converting the calendar to "standard".
        If the value is a float, it will be used as the fill value for all variables.
        If the value is a string "interpolate", the new dates will be linearly interpolated over time.
        A dictionary can be used to specify a different fill value for each variable.
        Keys should be the names of the variables as they appear in the first entry in the "variable_name" lists of the "Notes" section.
        If True, temperatures will be interpolated and precipitation will be filled with 0.
        If False, the calendar will not be converted. Only possible for "Raven" models.
    save_as : str, optional
        Where to save the reformatted data. If None, the data will not be saved.
        This can be useful when multiple files are needed for a single model run (e.g. Hydrotel needs a configuration file).
    \*\*kwargs : dict
        Additional keyword arguments to pass to the save function.

    Returns
    -------
    xr.Dataset
        The reformatted dataset.
    dict
        For Hydrotel, a dictionary containing the configuration for the meteorological data.
        If `save_as` is provided, the configuration will have been saved to a file with the same name as `save_as`, but with a ".nc.config" extension.
        For Raven, a dictionary containing the 'data_type' and 'alt_names_meteo' keys required for the 'model_config' argument.

    Notes
    -----
    The input dataset should ideally be CF-compliant.
    This function will attempt to detect the variables based on the standard_name attribute, the cell_methods attribute, or the variable name
    (AMIP column) taken from https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html.

    Specifically:

    - If using 1D time series, the station dimension should have an attribute `cf_role` set to "timeseries_id".
    - Units don't need to be canonical, but they should be convertible to the expected units and be understood by `xclim`.
    - WARNING: Snowfall units should be in water equivalent (e.g. mm or kg/m²/s). It must NOT be a height (e.g. cm of fresh snow on the ground).
    - Be aware that the function will first try to detect the variables based on the attributes, and then the variable name.
    - The following attempts will be made to detect the variables:
        - Longitude:
            - standard_name: "longitude"
            - variable name: "longitude", "lon"
        - Latitude:
            - standard_name: "latitude"
            - variable name: "latitude", "lat"
        - Elevation:
            - standard_name: "surface_altitude"
            - variable name: "elevation", "orog", "z", "altitude", "height"
        - Precipitation:
            - standard_name: "*precipitation*" (e.g. "lwe_thickness_of_precipitation_amount")
            - variable name: "pr", "precip", "precipitation"
        - Rainfall:
            - standard_name: "*rainfall*" (e.g. "rainfall_flux", "rainfall_amount")
            - variable name: "prlp", "rainfall", "rain", "precipitation_rain"
        - Snowfall:
            - standard_name: "*snowfall*" (e.g. "snowfall_flux", "snowfall_amount")
            - variable name: "prsn", "snowfall", "precipitation_snow"
        - Maximum temperature:
            - standard_name: "air_temperature"
            - cell_methods: "time: maximum"
            - variable name: "tasmax", "tmax", "t2m_max", "temperature_max"
        - Minimum temperature:
            - standard_name: "air_temperature"
            - cell_methods: "time: minimum"
            - variable name: "tasmin", "tmin", "t2m_min", "temperature_min"
        - Mean temperature:
            - standard_name: "air_temperature"
            - cell_methods: "time: mean"
            - variable name: "tas", "tmean", "t2m", "temperature_mean"

    Hydrotel requires the following variables: ["longitude", "latitude", "elevation", "time", "tasmax", "tasmin", "pr"].
    Raven requires the following variables: ["longitude", "latitude", "elevation", "time", "tasmax/tasmin" or "tas", "pr" or "prlp/prsn"].
    """
    ds = ds.copy()
    if model in ["Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"]:
        model = "Raven"
    if model not in ["Hydrotel", "Raven"]:
        raise NotImplementedError(f"The model '{model}' is not recognized.")

    # Detect and rename variables if necessary
    variables = {
        "longitude": {"standard_name": "longitude", "names": ["longitude", "lon"]},
        "latitude": {"standard_name": "latitude", "names": ["latitude", "lat"]},
        "elevation": {
            "standard_name": "surface_altitude",
            "names": ["elevation", "orog", "z", "altitude", "height"],
        },
        "pr": {
            "standard_name": ".*precipitation.*",
            "names": ["pr", "precip", "precipitation"],
        },
        "prlp": {
            "standard_name": ".*rainfall.*",
            "names": ["prlp", "rainfall", "rain", "precipitation_rain"],
        },
        "prsn": {
            "standard_name": ".*snowfall.*",
            "names": ["prsn", "snowfall", "precipitation_snow"],
        },
        "tasmax": {
            "standard_name": "air_temperature",
            "cell_methods": "time: max.*",
            "names": ["tasmax", "tmax", "t2m_max", "temperature_max"],
        },
        "tasmin": {
            "standard_name": "air_temperature",
            "cell_methods": "time: min.*",
            "names": ["tasmin", "tmin", "t2m_min", "temperature_min"],
        },
        "tas": {
            "standard_name": "air_temperature",
            "cell_methods": "time: mean.*",
            "names": ["tas", "tmean", "t2m", "temperature_mean"],
        },
    }
    for attributes in variables.values():
        names = attributes.pop("names")
        ds = _detect_variable(ds, attributes, names, return_ds=True)

    # Check if the dataset contains the required variables
    required_vars = ["longitude", "latitude", "elevation", "time"]
    if model in ["Raven"]:
        if all(v in ds for v in ["prlp", "prsn", "pr"]) or all(
            v in ds for v in ["tasmax", "tasmin", "tas"]
        ):
            warnings.warn(
                "The dataset contains multiple variables for precipitation or temperature. "
                "Please ensure that only the correct variables are used in 'data_type' and 'alt_names_meteo'."
            )
        # Determine the data_type mode for Raven
        if all(v in ds for v in ["tasmax", "tasmin"]):
            required_vars.extend(["tasmax", "tasmin"])
        if "tas" in ds:
            required_vars.extend(["tas"])
        if not all(v in ds for v in ["tasmax", "tasmin"]) and "tas" not in ds:
            raise ValueError(
                "The dataset is missing the required variables for Raven: 'tasmax/tasmin' or 'tas'."
            )
        if "pr" in ds:
            required_vars.extend(["pr"])
        if all(v in ds for v in ["prlp", "prsn"]):
            required_vars.extend(["prlp", "prsn"])
        if not all(v in ds for v in ["prlp", "prsn"]) and "pr" not in ds:
            raise ValueError(
                "The dataset is missing the required variables for Raven: 'pr' or 'prlp/prsn'."
            )
    elif model == "Hydrotel":
        required_vars.extend(["tasmax", "tasmin", "pr"])

    if not all(v in ds for v in required_vars):
        missing = set(required_vars).difference(set(ds.variables))
        raise ValueError(
            f"The dataset is missing the following required variables for '{model}': "
            f"{missing}."
        )

    # Elevation as a coordinate
    ds = ds.assign_coords({"elevation": ds.elevation})

    # Convert units
    # Precipitation first, since it is more complex
    def _is_rate(u):
        q = xc.core.units.str2pint(u)
        return q.dimensionality.get("[time]", 0) < 0

    for pr in {"pr", "prlp", "prsn"}.intersection(ds.variables):
        if _is_rate(ds[pr].attrs.get("units", "")):
            ds[pr] = xc.units.rate2amount(ds[pr])
        if (
            pr == "prsn"
            and xc.core.units.str2pint(
                ds[pr].attrs.get("units", "")
            ).dimensionality.get("[mass]", 0)
            > 0
        ):
            warnings.warn(
                "The snowfall units contain mass. They will be converted to volume using the density of water (1000 kg/m³),"
                " which is correct if the data is the liquid flux of the solid phase of precipitation (i.e. already in water equivalent). "
                "If your data is anything else, please convert it to water equivalent before using this function."
            )
        ds[pr] = xc.units.convert_units_to(ds[pr], "mm", context="hydro")

    variables_and_units = {
        "elevation": "m",
    }
    for t in {"tasmax", "tasmin", "tas"}.intersection(ds.variables):
        variables_and_units[t] = "degC"
    ds = change_units(ds, variables_and_units)
    ds = change_units(
        ds, variables_and_units
    )  # FIXME: Until xscen>=0.13, run twice to ensure all variables have the exact units requested

    # Ensure that longitude is in the range [-180, 180]
    # This tries guessing if lons are wrapped around at 180+ but without much information, this might not be true
    if np.min(ds["longitude"]) >= -180 and np.max(ds["longitude"]) <= 180:
        pass
    elif np.min(ds["longitude"]) >= 0 and np.max(ds["longitude"]) <= 360:
        warnings.warn(
            "Longitude values appear to be in the range [0, 360]. They will be converted to [-180, 180]."
        )
        with xr.set_options(keep_attrs=True):
            ds["longitude"] = ds["longitude"] - 180

    # Convert calendar
    if convert_calendar_missing is not False:
        var_no_time = [v for v in ds.data_vars if "time" not in ds[v].dims]
        convert_calendar_kwargs = {"calendar": "standard", "use_cftime": False}
        if isinstance(convert_calendar_missing, dict):
            missing_by_var = convert_calendar_missing
        elif convert_calendar_missing is True:
            # Interpolate missing values for temperature and fill with 0 for precipitation
            missing_by_var = {}
            for var in {"tasmax", "tasmin", "tas"}.intersection(ds.variables):
                missing_by_var[var] = "interpolate"
            for var in {"pr", "prlp", "prsn"}.intersection(ds.variables):
                missing_by_var[var] = 0
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
                    f"NaNs will need to be filled manually before running Hydrotel or Raven."
                )

        ds = clean_up(
            ds,
            convert_calendar_kwargs=convert_calendar_kwargs,
            missing_by_var=missing_by_var,
        )
        # FIXME: Temporary fix until xscen>=0.13 or https://github.com/pydata/xarray/issues/10266
        for v in var_no_time:
            if "time" in ds[v].dims:
                ds[v] = ds[v].isel(time=0).drop_vars("time")
    elif model == "Hydrotel":
        # Hydrotel requires the calendar to be "standard"
        if ds.time.dt.calendar not in ["standard", "gregorian", "proleptic_gregorian"]:
            raise ValueError(
                f"The calendar '{ds.time.dt.calendar}' is not supported by Hydrotel. "
                "Please convert it to 'standard' before running the model."
            )

    # Ensure that the spatial coordinates are recognized by cf_xarray (primarily for RavenPy, but useful anyway)
    if ds.cf.coordinates.get("longitude") is None:
        ds["longitude"].attrs["standard_name"] = "longitude"
    if ds.cf.coordinates.get("latitude") is None:
        ds["latitude"].attrs["standard_name"] = "latitude"
    if ds.cf.coordinates.get("vertical") is None:
        ds["elevation"].attrs["standard_name"] = "height"

    # Additional data processing specific to Hydrotel
    if model == "Hydrotel":
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
        if (len(ds["latitude"].dims) == 2) or ("latitude" in ds.dims):
            mask = ~ds.pr.isnull().all(dim="time")
            if xc.core.utils.uses_dask(mask):
                mask = mask.compute()
            ds = stack_drop_nans(ds, mask=mask, new_dim="station")

            # Add station ID
            ds = ds.assign_coords(station=("station", np.arange(len(ds.station))))
            ds["station"].attrs = {
                "long_name": "Station data",
                "cf_role": "timeseries_id",
            }

        try:
            station_dim = ds.cf.cf_roles["timeseries_id"][0]
        except KeyError:
            raise ValueError(
                "The dataset does not contain a dimension with the cf_role 'timeseries_id'. Cannot determine the station dimension."
            )

        cfg = {
            "TYPE (STATION/GRID/GRID_EXTENT)": "STATION",
            "STATION_DIM_NAME": station_dim,
            "LATITUDE_NAME": "latitude",
            "LONGITUDE_NAME": "longitude",
            "ELEVATION_NAME": "elevation",
            "TIME_NAME": ds.cf["time"].name,
            "TMIN_NAME": "tasmin",
            "TMAX_NAME": "tasmax",
            "PRECIP_NAME": "pr",
        }

        if save_as:
            Path(save_as).parent.mkdir(parents=True, exist_ok=True)
            with Path(save_as).with_suffix(".nc.config").open("w") as f:
                for k, v in cfg.items():
                    f.write(f"{k}; {v}\n")
            ds.to_netcdf(Path(save_as).with_suffix(".nc"), **kwargs)

        return ds, cfg

    # Additional data processing specific to Raven
    if model == "Raven":
        is_1d = False
        if (
            ds.cf.axes.get("X") is None
            and ds.cf.cf_roles.get("timeseries_id") is not None
        ):
            # Reorder dimensions to match Raven's expectations for .rvt (station, t)
            ds = ds.transpose(ds.cf.cf_roles["timeseries_id"][0], "time")
            # Rename the station dimension to "station_id"
            ds = ds.rename({ds.cf.cf_roles["timeseries_id"][0]: "station_id"})
            # Raven needs the station dimension to be a string
            ds["station_id"] = ds["station_id"].astype(str)
            is_1d = True

        elif ds.cf.axes.get("X") is not None:
            warnings.warn(
                "2D data is not yet supported by xHydro. Either use RavenPy directly or stack the data to 1D."
            )
            # Reorder dimensions to match Raven's expectations for .rvt (x, y, t)
            # Raven is faster with gridded inputs than with stations when there are a lot of stations
            x_name = ds.cf.axes["X"][0]
            y_name = ds.cf.axes["Y"][0]

            ds = ds.transpose(x_name, y_name, "time")

        elif len(ds.squeeze().dims) == 1 and "time" in ds.dims:
            # 1D time series with no lat/lon dimensions, assume it's a single station
            ds = ds.squeeze()
            # Add a station dimension
            ds = ds.expand_dims("station_id").assign_coords(
                station_id=("station_id", ["0"]),
            )
            ds["longitude"] = ds["longitude"].expand_dims("station_id")
            ds["latitude"] = ds["latitude"].expand_dims("station_id")
            ds["elevation"] = ds["elevation"].expand_dims("station_id")
            is_1d = True

        else:
            raise ValueError(
                "The dataset does not contain a dimension with the cf_role 'timeseries_id' or the axes 'X' and 'Y'. "
                "Cannot determine the spatial dimensions."
            )

        # Prepare the configuration for Raven
        # Reference: https://ravenpy.readthedocs.io/en/latest/_modules/ravenpy/config/defaults.html#
        conv = {
            "tasmax": "TEMP_MAX",
            "tasmin": "TEMP_MIN",
            "tas": "TEMP_AVE",
            "pr": "PRECIP",
            "prlp": "RAINFALL",
            "prsn": "SNOWFALL",
        }

        cfg = dict()
        cfg["data_type"] = [conv[v] for v in required_vars if v in conv]
        cfg["alt_names_meteo"] = {conv[v]: v for v in required_vars if v in conv} | {
            "LONGITUDE_NAME": "longitude",
            "LATITUDE_NAME": "latitude",
            "ELEVATION_NAME": "elevation",
        }

        if save_as:
            Path(save_as).parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(Path(save_as).with_suffix(".nc"), **kwargs)
            cfg["meteo_file"] = str(Path(save_as).with_suffix(".nc"))

        if is_1d:
            cfg["meteo_station_properties"] = {
                "ALL": {
                    "elevation": ds.elevation.values,
                    "latitude": ds.latitude.values,
                    "longitude": ds.longitude.values,
                }
            }
        else:
            cfg["meteo_station_properties"] = None
            if save_as is False:
                warnings.warn(
                    "The dataset is 2D, but 'save_as' is not set. Be sure to provide a `meteo_file` in the model configuration."
                )

        return ds, cfg


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
