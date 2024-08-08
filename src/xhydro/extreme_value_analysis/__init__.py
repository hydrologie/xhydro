"""Extreme value analysis analysis module."""

import warnings

JULIA_WARNING = (
    "Julia not installed, Extreme Value Analysis functionalities will be disabled."
)


try:
    from .julia_import import Extremes, jl

    __all__ = ["JULIA_WARNING", "Extremes", "jl"]

except ImportError:
    warnings.warn(
        JULIA_WARNING,
        ImportWarning,
    )
    __all__ = ["JULIA_WARNING"]
