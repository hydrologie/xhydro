"""Utility functions for xhydro."""
from typing import Union

import xarray as xr
from xclim.core.units import rate2amount

__all__ = ["compute_volume"]


def compute_volume(
    da: Union[xr.DataArray | xr.Dataset], *, attrs: dict = None, out_units: str = None
):
    """Compute the volume of water from a streamflow variable.

    Parameters
    ----------
    da : Union[xr.DataArray | xr.Dataset]
        Streamflow variable.
    attrs : dict, optional
        Attributes to add to the output variable.
        Default attributes will be added if not provided for "long_name", "units", "cell_methods" and "description".
    out_units : str, optional
        Output units. Defaults to "m3".

    Returns
    -------
    xr.DataArray
        Volume of water.
    """
    if isinstance(da, xr.Dataset):
        da = da["streamflow"]

    default_attrs = {
        "long_name": "Volume of water",
        "units": "m3",
        "cell_methods": "time: sum",
        "description": "Volume of water",
    }
    attrs = attrs or {}
    for k, v in default_attrs.items():
        attrs.setdefault(k, v)
    out = rate2amount(da, out_units=out_units or "m3")
    out.attrs.update(attrs)

    return out
