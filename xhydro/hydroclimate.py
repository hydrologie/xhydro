"""Module to compute hydroclimatic statistics."""

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
    """Compute climatological operation."""
    return climatological_mean(ds, **kwargs)
