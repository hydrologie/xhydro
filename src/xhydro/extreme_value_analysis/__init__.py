"""Extreme value analysis analysis module."""

import warnings

JULIA_WARNING = (
    "Julia not installed, Extreme Value Analysis functionalities will be disabled."
)

try:
    from .julia_import import Extremes, jl
    from .parameterestimation import fit

    __all__ = ["Extremes", "fit", "jl"]

except ImportError:
    warnings.warn(JULIA_WARNING)

    __all__ = ["JULIA_WARNING"]
