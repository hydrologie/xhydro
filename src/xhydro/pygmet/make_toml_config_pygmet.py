"""
Hard-coded TOML template writer.

This module writes a new TOML config file based on a fixed template, replacing only
the fields marked with "# MAKE_DYNAMIC" in the original template.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawToml:
    """Inject TOML verbatim (advanced use)."""

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
    if isinstance(value, (datetime, date)):
        return _toml_escape_single_quoted(value.isoformat())
    if isinstance(value, str):
        return _toml_escape_single_quoted(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(to_toml(v) for v in value) + "]"
    if value is None:
        # TOML has no null; choose what your application expects.
        return '""'
    raise TypeError(f"Unsupported type for TOML conversion: {type(value).__name__}")


# Keys whose values should be provided at runtime:
DYNAMIC_KEYS: Sequence[str] = (
    "case_name",
    "num_processes",
    "modelsettings_file",
    "input_stn_all",
    "infile_grid_domain",
    "outpath_parent",
    "date_start",
    "date_end",
    "input_vars",
    "target_vars",
    "target_vars_WithProbability",
    "minRange_vars",
    "maxRange_vars",
    "transform_vars",
    "predictor_name_static_stn",
    "predictor_name_static_grid",
    "ensemble_end",
    "clen",  # codespell:ignore clen
    "auto_corr_method",
    "target_vars_max_constrain",
)

# The hard-coded template (generated from testcase.config.static_withTemperature.toml):
TOML_TEMPLATE = """\
########################################################################################################################
#  TEMPLATE FILE TO RUN Pygmet for downscaling / generator
########################################################################################################################

case_name = {case_name}

num_processes = {num_processes}

# model setting file
modelsettings_file = {modelsettings_file}

########################################################################################################################
#  Input and output information
########################################################################################################################

input_stn_all = {input_stn_all}

infile_grid_domain = {infile_grid_domain}

outpath_parent = {outpath_parent}

# file list containing dynamic predictor inputs (optional). Give it a nonsense string (e.g., "NA") can turn off dynamic
# predictors
dynamic_predictor_filelist = "NA"

########################################################################################################################
#  Period to process
########################################################################################################################

date_start = {date_start}
date_end = {date_end}

########################################################################################################################
#  Variables to process
########################################################################################################################

input_vars = {input_vars}

# Target variables to output, should always contain precip as first variable
target_vars = {target_vars}

# input variables may need some conversion to get target variables. if input_var and target_var have the same variable
# name, no need to add mapping relation
mapping_InOut_var = []

# Target variables that will get an additional output probability of exceedance (P(exc) and P(non exc))
target_vars_WithProbability = {target_vars_WithProbability}

# Set the minimum and maximum range for each variable, must be same order as input_vars
minRange_vars = {minRange_vars}
maxRange_vars = {maxRange_vars}

############################## dynamic predictors for regression
# only useful when dynamic_predictor_filelist is valid

# dynamic predictors for each target_vars. Empty list means no dynamic predictors
dynamic_predictor_name = []

# dynamic predictors may needs some processing. two keywords: "interp" an "transform"
# Example "cube_root_prec_rate:interp=linear:transform=boxcox"
dynamic_predictor_operation = []


# Transformation to use for each variable, can be any of:
# 'log', 'logit', 'direct', 'sqrt', 'cbrt', 'boxcox', 'yeojohnson', 'standardize'
transform_vars = {transform_vars}

########################################################################################################################
#  Static predictors
########################################################################################################################

# Names of variables in input_stn_all file (static predictors)
predictor_name_static_stn = {predictor_name_static_stn}

# Names of variables in infile_grid_domain file (static predictors)
predictor_name_static_grid = {predictor_name_static_grid}

########################################################################################################################
#  Stochastic and correlation settings
########################################################################################################################

# Number of ensemble members to generate (inclusive end index)
# run ensemble or not: true or false
ensemble_flag = true

# ensemble settings
ensemble_start = 1
ensemble_end = {ensemble_end}

# link variables for random number generation dependence
linkvar = []

# Correlation length for each variable (same order as input_vars)
clen = {clen} # codespell:ignore clen
lag1_auto_cc = [-9999, -9999, -9999] # corresponding to target_vars
cross_cc = [] # corresponding to linkvar

# Method for auto-correlation for each variable (same order as input_vars)
auto_corr_method = {auto_corr_method}

########################################################################################################################
#  Constraints
########################################################################################################################
rolling_window = 31 # 31-monthly rolling mean to remove monthly cycle
# Max constrain list (subset of target_vars)
target_vars_max_constrain = {target_vars_max_constrain}
"""


def render_config_toml(params: Mapping[str, Any], strict: bool = True) -> str:
    """
    Render the template into a TOML config string.

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

    # format_map will raise KeyError if any placeholder isn't present; strict above prevents that.
    return TOML_TEMPLATE.format_map(mapping)


def write_config_toml(output_path: str | Path, params: Mapping[str, Any], strict: bool = True) -> bool:
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
    output_path.write_text(render_config_toml(params, strict=strict), encoding="utf-8")
    return True
