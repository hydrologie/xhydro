from xhydro.extreme_value_analysis.julia_import import Extremes, jl


def parametervar(fm, nboot: int = None):
    if nboot is None:
        return Extremes.parametervar(fm)
    else:
        return Extremes.parametervar(fm, nboot)
