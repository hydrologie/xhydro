"""Extreme value analysis analysis module."""

import warnings

try:
    from .julia_import import Extremes, jl

    __all__ = ["Extremes", "jl"]

except ImportError:
    warnings.warn(
        "Julia not installed, Extreme Value Analysis functionalities will be disabled.",
        ImportWarning,
    )
    __all__ = []
