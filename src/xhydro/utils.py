"""Utility functions for xhydro."""

import datetime as dt

import xarray as xr
from xscen.diagnostics import health_checks

__all__ = ["health_checks"]


def update_history(
    hist_str: str,
    *inputs_list: xr.DataArray | xr.Dataset,
    new_name: str | None = None,
    **inputs_kws: xr.DataArray | xr.Dataset,
) -> str:
    r"""Return a history string with the timestamped message and the combination of the history of all inputs.

    The new history entry is formatted as "[<timestamp>] <new_name>: <hist_str> - xhydro version: <xhydro.__version__>."

    Parameters
    ----------
    hist_str : str
        The string describing what has been done on the data.
    \*inputs_list : xr.DataArray or xr.Dataset
        The datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixed by their "name" attribute if available.
    new_name : str, optional
        The name of the newly created variable or dataset to prefix hist_msg.
    \*\*inputs_kws : xr.DataArray or xr.Dataset
        Mapping from names to the datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixes by the passed name.

    Returns
    -------
    str
        The combine history of all inputs starting with `hist_str`.
    """
    from xhydro import (  # pylint: disable=cyclic-import,import-outside-toplevel
        __version__,
    )

    merged_history = merge_attributes(
        "history",
        *inputs_list,
        new_line="\n",
        missing_str="",
        **inputs_kws,
    )
    if len(merged_history) > 0 and not merged_history.endswith("\n"):
        merged_history += "\n"
    merged_history += (
        f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {new_name or ''}: "
        f"{hist_str} - xhydro version: {__version__}"
    )
    return merged_history


def merge_attributes(
    attribute: str,
    *inputs_list: xr.DataArray | xr.Dataset,
    new_line: str = "\n",
    missing_str: str | None = None,
    **inputs_kws: xr.DataArray | xr.Dataset,
) -> str:
    r"""Merge attributes from several DataArrays or Datasets.

    If more than one input is given, its name (if available) is prepended as: "<input name> : <input attribute>".

    Parameters
    ----------
    attribute : str
        The attribute to merge.
    \*inputs_list : xr.DataArray or xr.Dataset
        The datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixed by their `name` attribute if available.
    new_line : str
        The character to put between each instance of the attributes. Usually, in CF-conventions,
        the history attributes uses '\\n' while cell_methods uses ' '.
    missing_str : str
        A string that is printed if an input doesn't have the attribute. Defaults to None, in which
        case the input is simply skipped.
    \*\*inputs_kws : xr.DataArray or xr.Dataset
        Mapping from names to the datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixes by the passed name.

    Returns
    -------
    str
        The new attribute made from the combination of the ones from all the inputs.
    """
    inputs = [(getattr(in_ds, "name", None), in_ds) for in_ds in inputs_list]
    inputs += list(inputs_kws.items())

    merged_attr = ""
    for in_name, in_ds in inputs:
        if attribute in in_ds.attrs or missing_str is not None:
            if in_name is not None and len(inputs) > 1:
                merged_attr += f"{in_name}: "
            merged_attr += in_ds.attrs.get(
                attribute, "" if in_name is None else missing_str
            )
            merged_attr += new_line

    if len(new_line) > 0:
        return merged_attr[: -len(new_line)]  # Remove the last added new_line
    return merged_attr
