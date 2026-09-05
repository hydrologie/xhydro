"""
Hard-coded template writer for the PyGMET settings file.

This module writes a new TOML settings file from a fixed template, replacing templated values with desired values.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawToml:
    """Inject TOML verbatim."""

    text: str


def _toml_escape_single_quoted(s: str) -> str:
    """
    Manage the single quotes in strings.

    Parameters
    ----------
    s : str
        String that has single quotes and needs to be processed.

    Returns
    -------
    str :
        The string with the processed single quotes.
    """
    # TOML single-quoted strings are literal except that you must escape single quotes.
    return "'" + s.replace("'", "''") + "'"


def to_toml(value: Any) -> str:
    """
    Convert common Python types to TOML literals.

    Parameters
    ----------
    value : Any
        Value to convert to TOML literal.

    Returns
    -------
    str :
        The string literal writable to TOML file.
    """
    if isinstance(value, RawToml):
        return value.text
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_escape_single_quoted(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(to_toml(v) for v in value) + "]"
    if value is None:
        return "''"
    raise TypeError(f"Unsupported type for TOML conversion: {type(value).__name__}")


DYNAMIC_KEYS: Sequence[str] = (
    "stn_lat_name",
    "stn_lon_name",
    "grid_lat_name",
    "grid_lon_name",
    "grid_mask_name",
    "dynamic_grid_lat_name",
    "dynamic_grid_lon_name",
    "nearstn_min",
    "nearstn_max",
    "try_radius",
    "initial_distance",
)

TOML_TEMPLATE = """\
# Some default settings

########################################################################################################################
# general settings
########################################################################################################################

# master seed: control all probabilistic process (e.g., probabilistic estimation, machine learning methods)
# a negative value means random generation without reproducibility
master_seed = 20230104

########################################################################################################################
# settings for gridded estimation using regression or machine learning methods
########################################################################################################################

############################## station/grid file dimension name
# the spatial dims of input stations
stn_lat_name = {stn_lat_name}
stn_lon_name = {stn_lon_name}

# the 2D spatial dims of the target grid domain
# note that target grid domain netcdf must have x/y dims, while lat and lon are 2D arrays
grid_lat_name = {grid_lat_name}
grid_lon_name = {grid_lon_name}
grid_mask_name = {grid_mask_name}

# the 2D spatial dims of dynamic predictor inputs
dynamic_grid_lat_name = {dynamic_grid_lat_name}
dynamic_grid_lon_name = {dynamic_grid_lon_name}

########################################################################################################################
# default settings
# they are as useful as the above settings but using their default values does not affect model run for any case
########################################################################################################################

# gridding methods: locally weighted regression and meachine learning methods.
# Sklearn module is used to support most functions: https://scikit-learn.org/stable/supervised_learning.html
# Locally weighted regression.
#   Two original methods are LWR:Linear and LWR:Logistic.
#   Sklearn-based methods support simple usage with "model.fit()" and "model.predict" or "model.predict_prob",
#   in the format of LWR:linear_model.METHOD
#   Examples of METHOD are LinearRegression, LogisticRegression, Ridge, BayesianRidge, ARDRegression, Lasso, ElasticNet, Lars, etc
# Global regression using machine learnig methods:
#   Machine learning methods are supported by sklearn. Parameters of methods supported by sklearn can be defined at the bottom of this
#   configuration file (optional)
#   Examples: ensemble.RandomForestRegressor, ensemble.RandomForestClassifier, neural_network.MLPRegressor, neural_network.MLPClassifier,
#   ensemble.GradientBoostingClassifier, ensemble.GradientBoostingRegressor
# The parameters of sklearn methods can be defined in the [sklearn] section
gridcore_continuous = 'LWR:Linear'
gridcore_classification = 'LWR:Logistic' # for probability of event
n_splits = 10 # only useful for machine learning methods. cross validation to generate uncertainty estimates.

# output random fields
output_randomfield = false

# Number of stations to consider for each target point. nearstn_min<=nearstn_max.
nearstn_min = {nearstn_min}  # nearby stations: minimum number
nearstn_max = {nearstn_max} # nearby stations: maximum number

# first try this radius (km). if not enough, expand. Could be useful to reduce computation time for large domain search.
try_radius = {try_radius}

# overwrite existing files
overwrite_stninfo = true
overwrite_station_cc = true
overwrite_weight = true
overwrite_cv_reg = true
overwrite_grid_reg = true
overwrite_ens = true
overwrite_spcorr = true

########################################################################################################################
# distance-based weight calculation
########################################################################################################################

initial_distance = {initial_distance} # Initial Search Distance in km (expanded if need be)

# Weight calculation formula. Only two variables/parameters are allowed in the formula
# dist: distance between points (km in the script)
# maxdist (optional): max(initial_distance, max(dist)+1), which is a parameter used in weight calculation
# 3 is the exponential factor and is the default parameter
weight_formula = '(1 - (dist / maxdist) ** 3) ** 3'


########################################################################################################################
# method-related settings
# default values can be directly used
########################################################################################################################

[transform]
# note: the name must be consistent with transform_vars
[transform.boxcox]
exp = 4

[sklearn]
# if no parameters are provided or if the section does not even exist, default parameters will be used.
# just provide the method name, no need to include the submodule name
[sklearn.RandomForestRegressor]
n_estimators = 500 # a example of RandomForestRegressor parameter
n_jobs = 5
[sklearn.RandomForestClassifier]
n_estimators = 500 # a example of RandomForestRegressor parameter
n_jobs = 5
"""


def render_model_settings_toml(params: Mapping[str, Any], strict: bool = True) -> str:
    """
    Render the template into a TOML settings string.

    Parameters
    ----------
    params : Mapping[str, Any]
        The list of parameters to write to the file, in replacement of the template tags.
    strict : bool
        True if all DYNAMIC_KEYS must be present in `params`.

    Returns
    -------
    str :
       The full template with tags replaced.
    """
    missing = [k for k in DYNAMIC_KEYS if k not in params]
    if strict and missing:
        raise KeyError(f"Missing replacements for keys: {missing}")

    mapping = {k: to_toml(params[k]) for k in DYNAMIC_KEYS if k in params}
    return TOML_TEMPLATE.format_map(mapping)


def write_settings_toml(output_path: str | Path, params: Mapping[str, Any], strict: bool = True) -> bool:
    """
    Write a fresh model settings TOML file from the hard-coded template.

    Parameters
    ----------
    output_path : str or Path
        Path to the filled-in template file we want to write to disk.
    params : Mapping[str,Any]
        The list of parameters to write to the file, in replacement of the template tags.
    strict : bool
        True if all DYNAMIC_KEYS must be present in `params`.

    Returns
    -------
    bool :
        True if the code executes with no error.
    """
    output_path = Path(output_path)
    output_path.write_text(render_model_settings_toml(params, strict=strict), encoding="utf-8")
    return True
