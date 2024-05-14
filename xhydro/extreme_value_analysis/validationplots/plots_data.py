from juliacall import convert as jl_convert  # type: ignore

from xhydro.extreme_value_analysis.julia_import import Extremes, jl


def probplot_data(fm):
    return Extremes.probplot_data(fm)


def qqplot_data(fm):
    return Extremes.qqplot_data(fm)


def returnlevelplot_data(fm):
    return Extremes.returnlevelplot_data(fm)


def histplot_data(fm):
    return Extremes.histplot_data(fm)


# TODO: fix type issue
# def mrlplot_data(y: list, steps: int):
#     return Extremes.mrlplot_data(jl_convert(jl.Array, y), steps)
