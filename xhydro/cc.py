"""Module to compute climate change statistics using xscen functions."""

# Special imports from xscen
from xscen import (
    climatological_mean,  # FIXME: To be replaced with climatological_op once available
)
from xscen import compute_deltas, ensemble_stats, produce_horizon

__all__ = [
    "climatological_op",
    "compute_deltas",
    "produce_horizon",
    "ensemble_stats",
]


# FIXME: To be deleted once climatological_op is available in xscen
def climatological_op(ds, **kwargs):
    """Compute climatological operation.

    Notes
    -----
    This is a temporary wrapper to be deleted once climatological_op is available in xscen.
    For the time being, it is a simple wrapper around climatological_mean.
    See :py:func:`xscen.aggregate.climatological_mean` for more details.
    """
    return climatological_mean(ds, **kwargs)
