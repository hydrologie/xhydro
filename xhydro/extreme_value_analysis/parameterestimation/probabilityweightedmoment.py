from julia_import import Extremes, jl


def gevfitpwm(df, datacol: str):
    return Extremes.gevfitpwm(df, jl.Symbol(datacol))


def gumbelfitpwm(df, datacol: str):
    return Extremes.gumbelfitpwm(df, jl.Symbol(datacol))


def gpfitpwm(df, datacol: str):
    return Extremes.gumbelfitpwm(df, jl.Symbol(datacol))
