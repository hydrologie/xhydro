"""Utility functions for xhydro."""
import xarray as xr
from xclim.core.units import rate2amount

__all__ = ["compute_volume"]


def compute_volume(da: xr.DataArray, *, out_units: str = "m3", attrs: dict = None):
    """Compute the volume of water from a streamflow variable.

    Parameters
    ----------
    da : xr.DataArray
        Streamflow variable.
    out_units : str
        Output units. Defaults to "m3".
    attrs : dict
        Attributes to add to the output variable.
        Default attributes for "long_name", "units", "cell_methods" and "description" will be added if not provided.

    Returns
    -------
    xr.DataArray
        Volume of water.
    """
    default_attrs = {
        "long_name": "Volume of water",
        "units": "m3",
        "cell_methods": "time: sum",
        "description": "Volume of water",
    }
    attrs = attrs or {}
    for k, v in default_attrs.items():
        attrs.setdefault(k, v)
    out = rate2amount(da, out_units=out_units)
    out.attrs.update(attrs)

    return out
