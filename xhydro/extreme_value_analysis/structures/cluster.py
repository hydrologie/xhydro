from julia_import import Extremes, jl
from juliacall import convert as jl_convert  

class Cluster:
    u_1: float
    u_2: float
    position: list[int]
    value: list[float]

    def __init__(self, u_1: float, u_2: float, position: list[int], value: list[float]):
        self.u_1 = u_1
        self.u_2 = u_2
        self.position = position
        self.value = value

    def __repr__(self):
        return f"u_1 : {self.u_1} \nu_2 : {self.u_2} \nposition : {self.position} \nvalue : {self.value}"
    
    def length(self) -> int:
        return len(self.position)
    
    def maximum(self) -> int:
        return max(self.value)
    
    def sum(self) -> float:
        return sum(self.value) 