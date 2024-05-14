from xhydro.extreme_value_analysis.julia_import import Extremes, jl


def gevfit(df, datacol: str):
    return Extremes.gevfit(df, jl.Symbol(datacol))


def gumbelfit(df, datacol: str):
    return Extremes.gumbelfit(df, jl.Symbol(datacol))


def gpfit(df, datacol: str):
    return Extremes.gumbelfit(df, jl.Symbol(datacol))
