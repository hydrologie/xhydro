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
from clisops.utils.dataset_utils import cf_convert_between_lon_frames
from xclim.core.utils import uses_dask
from xscen.spatial import get_grid_mapping
from xscen.utils import change_units, clean_up, stack_drop_nans

from ._hydrotel import Hydrotel
from ._ravenpy_models import RavenpyModel


__all__ = ["format_input", "get_hydrological_model_inputs", "hydrological_model"]


def hydrological_model(model_config: dict) -> Hydrotel | RavenpyModel:
    """
    Initialize an instance of a hydrological model.

    Parameters
    ----------
    model_config : dict
        A dictionary containing the configuration for the hydrological model.
        Must contain a key "model_name" with the name of the model to use: e.g. "Hydrotel".
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
    model_name = str(model_config["model_name"]).upper()

    if model_name == "HYDROTEL":
        model_config.pop("model_name")
        return Hydrotel(**model_config)

    elif model_name in [
        "BLENDED",
        "GR4JCN",
        "HBVEC",
        "HMETS",
        "HYPR",
        "MOHYSE",
        "SACSMA",
    ]:
        return RavenpyModel(**model_config)
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")


def get_hydrological_model_inputs(model_name: str, required_only: bool = False) -> tuple[dict, str]:
    """
    Get the required inputs for a given hydrological model.

    Parameters
    ----------
    model_name : str
        The name of the hydrological model to use.
        Currently supported models are ["HYDROTEL", "Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA"].
    required_only : bool
        If True, only the required inputs will be returned.

    Returns
    -------
    dict
        A dictionary containing the required configuration for the hydrological model.
    str
        The documentation for the hydrological model.
    """
    model_name = model_name.upper()
    if model_name == "HYDROTEL":
        model = Hydrotel
    elif model_name in [
        "BLENDED",
        "GR4JCN",
        "HBVEC",
        "HMETS",
        "HYPR",
        "MOHYSE",
        "SACSMA",
    ]:
        model = RavenpyModel
    else:
        raise NotImplementedError(f"The model '{model_name}' is not recognized.")

    all_config = inspect.getfullargspec(model.__init__).annotations
    if required_only:
        all_config = {k: v for k, v in all_config.items() if k in inspect.getfullargspec(model.__init__).args}

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
    r"""
    Reformat CF-compliant meteorological data for use in hydrological models. See the "Notes" section for important details.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset containing the meteorological data. See the "Notes" section for more information on the expected format.
    model : str
        The name of the hydrological model to use.
        Currently supported models are:
        - "HYDROTEL", "Raven" (which is an alias for all RavenPy models), "Blended", "GR4JCN", "HBVEC", "HMETS", "HYPR", "Mohyse", "SACSMA".
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
        This can be useful when multiple files are needed for a single model run (e.g. HYDROTEL needs a configuration file).
    \*\*kwargs : dict
        Additional keyword arguments to pass to the save function.

    Returns
    -------
    xr.Dataset
        The reformatted dataset.
    dict
        For HYDROTEL, a dictionary containing the configuration for the meteorological data.
        If `save_as` is provided, the configuration will have been saved to a file with the same name as `save_as`, but with a ".nc.config" extension.
        For Raven, a dictionary containing the 'data_type' and 'alt_names_meteo' keys required for the 'model_config' argument.

    Notes
    -----
    The input dataset should ideally be CF-compliant and follow CMIP6's Controlled Vocabulary, but this function will attempt to detect the
    variables based on the standard_name attribute, the cell_methods attribute, or the variable name.
    More information on those attributes can be found here: https://wcrp-cmip.org/cmip-model-and-experiment-documentation/, and specifically
    the 'CMIP6 MIP table' link provided in the 'Search for variables' section.

    Specifically:

    - If using 1D time series, the station dimension should have an attribute `cf_role` set to "timeseries_id".
    - Units don't need to be canonical, but they should be convertible to the expected units and be understood by `xclim`.
    - Elevation represents the altitude of the meteorological data / model grid cell, not the altitude of the ground.
    - Snowfall units should be in water equivalent of precipitation (e.g. mm/day or kg/m²/s), NOT height (e.g. cm of fresh snow on the ground).
    - The function will try to detect the variables based on the attributes and the variable name. The following attempts will be made:
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
            - variable name: "prra", "prlp", "rainfall", "rain", "precipitation_rain"
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

    HYDROTEL requires the following variables: ["longitude", "latitude", "elevation", "time", "tasmax", "tasmin", "pr"].
    Raven requires the following variables: ["longitude", "latitude", "elevation", "time", "tasmax/tasmin" or "tas", "pr" or "prlp/prsn"].
    """
    ds = ds.copy()
    if model.lower() in ["blended", "gr4jcn", "hbvec", "hmets", "hypr", "mohyse", "sacsma", "raven"]:
        model = "Raven"
    elif model.upper() in ["HYDROTEL"]:
        model = "HYDROTEL"
    else:
        raise NotImplementedError(f"The model '{model}' is not recognized.")

    # Detect and rename variables if necessary
    variables = {
        "longitude": {"standard_name": "longitude", "names": ["longitude", "lon"]},
        "latitude": {"standard_name": "latitude", "names": ["latitude", "lat"]},
        "elevation": {
            "standard_name": "height",
            "names": [
                "elevation",
                "orog",
                "z",
                "altitude",
                "height",
                "surface_altitude",
            ],
        },
        "pr": {
            "standard_name": ".*precipitation.*",
            "names": ["pr", "precip", "precipitation"],
        },
        "prra": {
            "standard_name": ".*rainfall.*",
            "names": ["prra", "prlp", "rainfall", "rain", "precipitation_rain"],
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

    other_vars = [v for v in ds.data_vars if v not in variables]
    if len(other_vars) > 0:
        warnings.warn(
            f"The dataset contains the following variables that are not recognized: {other_vars}. These variables will be ignored.", stacklevel=2
        )

    # Check if the dataset contains the required variables
    required_vars = ["time"]
    if model in ["Raven"]:
        # Spatial coordinates can only be missing if the dataset is a single time series
        if any(v not in ds for v in ["longitude", "latitude", "elevation"]) and len(ds.squeeze().dims) == 1:
            warnings.warn(
                "The dataset is missing one or many of: 'longitude', 'latitude', or 'elevation'. They will need to be added "
                "manually using the 'meteo_station_properties' argument in the model configuration.",
                stacklevel=2,
            )
        else:
            required_vars.extend(["longitude", "latitude", "elevation"])

        # Temperature could be either max/min or mean
        if all(v in ds for v in ["tasmax", "tasmin"]):
            required_vars.extend(["tasmax", "tasmin"])
        if "tas" in ds:
            required_vars.extend(["tas"])
        if not all(v in ds for v in ["tasmax", "tasmin"]) and "tas" not in ds:
            raise ValueError("The dataset is missing the required variables for Raven: 'tasmax/tasmin' or 'tas'.")
        # Precipitation could be either pr or rainfall/snowfall
        if "pr" in ds:
            required_vars.extend(["pr"])
        if all(v in ds for v in ["prra", "prsn"]):
            required_vars.extend(["prra", "prsn"])
        if not all(v in ds for v in ["prra", "prsn"]) and "pr" not in ds:
            raise ValueError("The dataset is missing the required variables for Raven: 'pr' or 'prra/prsn'.")

        if all(v in ds for v in ["prra", "prsn", "pr"]) or all(v in ds for v in ["tasmax", "tasmin", "tas"]):
            warnings.warn(
                "The dataset contains multiple variables for precipitation or temperature. "
                "Please ensure that only the correct variables are used in 'data_type' and 'alt_names_meteo'.",
                stacklevel=2,
            )

    elif model == "HYDROTEL":
        required_vars.extend(["longitude", "latitude", "elevation", "tasmax", "tasmin", "pr"])

    if not all(v in ds for v in required_vars):
        missing = set(required_vars).difference(set(ds.variables))
        raise ValueError(f"The dataset is missing the following required variables for '{model}': {missing}.")

    # Convert units
    # Precipitation first, since it is more complex
    def _is_rate(u):
        q = xc.core.units.str2pint(u)
        return q.dimensionality.get("[time]", 0) < 0

    for pr in {"pr", "prra", "prsn"}.intersection(ds.variables):
        if _is_rate(ds[pr].attrs.get("units", "")):
            ds[pr] = xc.units.rate2amount(ds[pr])
        if pr == "prsn" and xc.core.units.str2pint(ds[pr].attrs.get("units", "")).dimensionality.get("[mass]", 0) > 0:
            warnings.warn(
                "The snowfall units contain mass. They will be converted to volume using the density of water (1000 kg/m³),"
                " which is correct if the data is the liquid flux of the solid phase of precipitation (i.e. already in water equivalent). "
                "If your data is anything else, please convert it to water equivalent before using this function.",
                stacklevel=2,
            )
        ds[pr] = xc.units.convert_units_to(ds[pr], "mm", context="hydro")

    # Other variables
    variables_and_units = {}
    if "elevation" in ds:
        variables_and_units["elevation"] = "m"
    for t in {"tasmax", "tasmin", "tas"}.intersection(ds.variables):
        variables_and_units[t] = "degC"
    ds = change_units(ds, variables_and_units)
    ds = change_units(ds, variables_and_units)  # FIXME: Until xscen>=0.13, run twice to ensure all variables have the exact units requested

    # Convert calendar
    if convert_calendar_missing is not False and ds.time.dt.calendar not in [
        "standard",
        "gregorian",
        "proleptic_gregorian",
    ]:
        var_no_time = [v for v in ds.data_vars if "time" not in ds[v].dims]
        convert_calendar_kwargs = {"calendar": "standard", "use_cftime": False}
        if isinstance(convert_calendar_missing, dict):
            missing_by_var = convert_calendar_missing
        elif convert_calendar_missing is True:
            missing_by_var = {}
            # Interpolate missing values for temperature
            for var in {"tasmax", "tasmin", "tas"}.intersection(ds.variables):
                missing_by_var[var] = "interpolate"
            # Fill missing values with 0 for precipitation
            for var in {"pr", "prra", "prsn"}.intersection(ds.variables):
                missing_by_var[var] = 0
        else:
            missing_by_var = None
            convert_calendar_kwargs["missing"] = convert_calendar_missing
            if ds.time.dt.calendar not in ["standard", "gregorian", "proleptic_gregorian"] and convert_calendar_missing is np.nan:
                warnings.warn(
                    f"The calendar '{ds.time.dt.calendar}' needs to be converted to 'standard', but 'convert_calendar_missing' is set to np.nan. "
                    f"NaNs will need to be filled manually before running HYDROTEL or Raven.",
                    stacklevel=2,
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
    elif model == "HYDROTEL":
        # HYDROTEL requires the calendar to be "standard"
        if ds.time.dt.calendar not in ["standard", "gregorian", "proleptic_gregorian"]:
            raise ValueError(
                f"The calendar '{ds.time.dt.calendar}' is not supported by HYDROTEL. Please convert it to 'standard' before running the model."
            )

    # Ensure that the spatial coordinates are recognized by cf_xarray (primarily for RavenPy, but useful anyway)
    if ds.cf.coordinates.get("longitude") is None and "longitude" in ds:
        ds = ds.assign_coords({"longitude": ds.longitude})
        ds["longitude"].attrs["standard_name"] = "longitude"
    if ds.cf.coordinates.get("latitude") is None and "latitude" in ds:
        ds = ds.assign_coords({"latitude": ds.latitude})
        ds["latitude"].attrs["standard_name"] = "latitude"
    if ds.cf.coordinates.get("vertical") is None and "elevation" in ds:
        ds = ds.assign_coords({"elevation": ds.elevation})
        ds["elevation"].attrs["standard_name"] = "height"

    # Manage the spatial dimensions
    # Case 1: Time series with no spatial dimension
    if set(ds.squeeze().dims) == {"time"}:
        ds = ds.squeeze()
        # Add a station dimension
        ds = ds.expand_dims("station_id").assign_coords(
            station_id=("station_id", ["0"]),
        )
        ds["station_id"].attrs = {"cf_role": "timeseries_id"}
        for c in ["longitude", "latitude", "elevation"]:
            if c in ds:
                ds[c] = ds[c].expand_dims("station_id")

        # Reorder dimensions to match model expectations
        if model == "HYDROTEL":
            ds = ds.transpose("time", "station_id")  # HYDROTEL expects (time, station)
        else:
            ds = ds.transpose("station_id", "time")  # Raven expects (station, time)

    # Case 2: Time series with a station dimension
    elif (ds.cf.axes.get("X") is None and ds.cf.cf_roles.get("timeseries_id") is not None) or len(ds.squeeze().dims) == 2:
        station_d = ds.cf.cf_roles.get("timeseries_id")
        if station_d is None:
            station_d = list(set(ds.squeeze().dims).difference({"time"}))
            warnings.warn(
                f"The dataset does not contain a dimension with the cf_role 'timeseries_id'. Using '{station_d}' as the station dimension.",
                UserWarning,
                stacklevel=2,
            )
        station_d = station_d[0]

        # Format the dataset to have an easier matching with Raven
        ds = ds.rename({station_d: "station_id"})
        ds["station_id"] = ds["station_id"].astype(str)  # Raven needs the station dimension to be a string
        ds["station_id"].attrs["cf_role"] = "timeseries_id"

        # Reorder dimensions to match model expectations
        if model == "HYDROTEL":
            ds = ds.transpose("time", "station_id")  # HYDROTEL expects (time, station)
        else:
            ds = ds.transpose("station_id", "time")  # Raven expects (station, time)

    # Case 3: Time series with a gridded dataset
    elif (ds.cf.axes.get("X") is not None) or len(ds.squeeze().dims) == 3:
        x_name = ds.cf.axes.get("X")
        y_name = ds.cf.axes.get("Y")

        # Try to get the x/y dimensions
        if any(d is None for d in [x_name, y_name]):
            if any(d in ds.dims for d in ["lon", "longitude", "x", "rlon"]):
                x_name = [d for d in ds.dims if d in ["lon", "longitude", "x", "rlon"]]
                ds[x_name[0]].attrs["axis"] = "X"
            if any(d in ds.dims for d in ["lat", "latitude", "y", "rlat"]):
                y_name = [d for d in ds.dims if d in ["lat", "latitude", "y", "rlat"]]
                ds[y_name[0]].attrs["axis"] = "Y"
            if x_name is None or y_name is None:
                raise ValueError("The dataset appears to be gridded, but the axes 'X' and 'Y' could not be determined.")

        if model == "HYDROTEL":
            # HYDROTEL is faster with 1D time series
            mask = ~ds.pr.isnull().all(dim="time")
            if xc.core.utils.uses_dask(mask):
                mask = mask.compute()
            ds = stack_drop_nans(ds, mask=mask, new_dim="station_id")

            # Add station ID
            ds = ds.assign_coords(station_id=("station_id", np.arange(len(ds.station_id))))
            ds["station_id"].attrs = {
                "cf_role": "timeseries_id",
            }

            # Remove gridmapping information
            gridmap = get_grid_mapping(ds)
            if len(gridmap) > 0:
                ds = ds.drop_vars(gridmap)
                for v in ds.data_vars:
                    if ds[v].attrs.get("grid_mapping") is not None:
                        ds[v].attrs.pop("grid_mapping")
            if x_name[0] != "longitude":
                ds = ds.drop_vars(x_name)
            if y_name[0] != "latitude":
                ds = ds.drop_vars(y_name)

        else:
            # Elevation data in Raven seems to be sensitive to the order of dimensions (T,Y,X)
            # Until this is resolved, we will enforce this order
            ds = ds.transpose("time", y_name[0], x_name[0])

    else:
        raise ValueError(
            "The dataset does not contain a dimension with the cf_role 'timeseries_id' or the axes 'X' and 'Y'. "
            "Cannot determine the spatial dimensions."
        )

    # Ensure that longitude is in the range [-180, 180]
    if "longitude" in ds:
        if uses_dask(ds.longitude):
            ds = ds.load()
        ds = cf_convert_between_lon_frames(ds, lon_interval=(-180, 180))[0]

    # Additional data processing specific to HYDROTEL
    if model == "HYDROTEL":
        # Time units in HYDROTEL must be exactly "days/minutes since 1970-01-01 00:00:00"
        # We need to convert the time variable to a 1D array of integers to prevent xarray from trying to convert it to a datetime64 array
        new_time = (ds["time"].values - np.datetime64("1970-01-01 00:00:00")).astype("timedelta64[m]").astype(int)
        ds["time"] = xr.DataArray(
            new_time,
            dims="time",
            coords={"time": new_time},
            attrs={"units": "minutes since 1970-01-01 00:00:00"},
        )

        # Prepare the information for the .nc.config file
        cfg = {
            "TYPE (STATION/GRID/GRID_EXTENT)": "STATION",
            "STATION_DIM_NAME": "station_id",
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

            ds = ds.chunk({"station_id": 1, "time": -1})
            ds.to_netcdf(Path(save_as).with_suffix(".nc"), **kwargs)

    # Additional data processing specific to Raven
    else:
        # Prepare the configuration for Raven
        # Reference: https://github.com/CSHS-CWRA/RavenPy/blob/master/src/ravenpy/config/conventions.py
        conv = {
            "tasmax": "TEMP_MAX",
            "tasmin": "TEMP_MIN",
            "tas": "TEMP_AVE",
            "pr": "PRECIP",
            "prra": "RAINFALL",
            "prsn": "SNOWFALL",
        }

        cfg = dict()
        cfg["data_type"] = [conv[v] for v in required_vars if v in conv]
        cfg["alt_names_meteo"] = {conv[v]: v for v in required_vars if v in conv}

        if save_as:
            Path(save_as).parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(Path(save_as).with_suffix(".nc"), **kwargs)
            cfg["meteo_file"] = str(Path(save_as).with_suffix(".nc"))

    return ds, cfg


def _detect_variable(ds: xr.Dataset, attributes: dict, names: list, return_ds: bool = False) -> xr.Dataset | str:
    """Find a variable in the dataset based on its attributes or name."""
    out = []
    # Search for variables based on attributes
    [out.append(v) for v in list(ds.variables.keys()) if all(re.fullmatch(attributes[a], ds[v].attrs.get(a, "")) for a in attributes)]
    # Search for variables based on name
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
