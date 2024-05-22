from xhydro.extreme_value_analysis.julia_import import Extremes, jl


def gevfitbayes(df, datacol: str):
    return Extremes.gevfitbayes(df, jl.Symbol(datacol))


def gumbelfitbayes(df, datacol: str):
    return Extremes.gumbelfitbayes(df, jl.Symbol(datacol))


def gpfitbayes(df, datacol: str):
    return Extremes.gumbelfitbayes(df, jl.Symbol(datacol))
