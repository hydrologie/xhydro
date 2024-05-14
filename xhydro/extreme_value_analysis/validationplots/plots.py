from xhydro.extreme_value_analysis.julia_import import Extremes


def diagnosticplots(fm):
    return Extremes.diagnosticplots(fm)


def probplot(fm):
    return Extremes.probplot(fm)


def qqplot(fm):
    return Extremes.qqplot(fm)


def qqplotci(fm):
    return Extremes.qqplotci(fm)


def returnlevelplot(fm):
    return Extremes.returnlevelplot(fm)


def returnlevelplotci(fm):
    return Extremes.returnlevelplotci(fm)


def histplot(fm):
    return Extremes.histplot(fm)


# def mrlplot(y, steps)
#     return Extremes.mrlplot(y, steps) #TODO: fix type issue
