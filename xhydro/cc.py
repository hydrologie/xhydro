"""Module to compute climate change statistics using xscen functions."""
import xarray

# Special imports from xscen
from xscen import (  # FIXME: To be replaced with climatological_op once available
    climatological_mean,
    compute_deltas,
    ensemble_stats,
    produce_horizon,
)

__all__ = [
    "climatological_op",
    "compute_deltas",
    "ensemble_stats",
    "produce_horizon",
]


# FIXME: To be deleted once climatological_op is available in xscen
def climatological_op(ds: xarray.Dataset, **kwargs: dict) -> xarray.Dataset:
    r"""Compute climatological operation.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    \*\*kwargs : dict
        Keyword arguments passed to :py:func:`xscen.aggregate.climatological_mean`.

    Returns
    -------
    xarray.Dataset
        Output dataset.

    Notes
    -----
    This is a temporary wrapper to be deleted once climatological_op is available in xscen.
    For the time being, it is a simple wrapper around climatological_mean.
    See :py:func:`xscen.aggregate.climatological_mean` for more details.
    """
    return climatological_mean(ds, **kwargs)
