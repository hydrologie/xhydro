from typing import List

from julia_import import Extremes, jl


def getcluster(
    y: list[float], u1: float, u2: float
) -> list:  # kamil List[Extremes.Cluster]
    return Extremes.getcluster(y, u1, u2)
