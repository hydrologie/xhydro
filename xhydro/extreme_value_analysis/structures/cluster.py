from julia_import import Extremes, jl
from juliacall import convert as jl_convert  

__all__ = ["getcluster"]

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



class ClusterError(Exception):
    """Custom exception for cluster-related errors."""
    pass

def getcluster(y: list[float], **kwargs) -> list[Cluster]:
    """
    Get clusters from the data using different methods based on provided arguments.
    
    Example usage:\n
    getcluster(y, u_1=0.5, u_2=1.0)\n
    getcluster(y, u=0.5, runlength=5)
    
    Parameters:
    y (list[float]): Input data.
    u_1 (float, optional): First threshold for method 1.
    u_2 (float, optional): Second threshold for method 1.
    u (float, optional): Threshold for method 2.
    runlength (int, optional): Run length for method 2.
    
    Returns:
    list[Cluster]: list of clusters.
    
    Raises:
    ClusterError: If the input arguments do not match the expected signatures.

    """
    if 'u_1' in kwargs and 'u_2' in kwargs and 'u' not in kwargs and 'runlength' not in kwargs:
        u_1, u_2 = kwargs['u_1'], kwargs['u_2']
        y = jl_convert(jl.Vector[jl.Real], y)
        jl_clusters = _getcluster_1(y, u_1, u_2)
    elif 'u_1' not in kwargs and 'u_2' not in kwargs and 'u' in kwargs and 'runlength' in kwargs:
        u, runlength = kwargs['u'], kwargs['runlength']
        y = jl_convert(jl.Vector[jl.Real], y)
        jl_clusters = _getcluster_2(y, u, runlength=runlength)
    else:
        raise ClusterError(
            "getcluster() has two accepted signatures:\n"
            "1. getcluster(y, u_1, u_2)\n"
            "2. getcluster(y, u, runlength)\n"
        )

    py_clusters = [jl_cluster_to_py_cluster(cluster) for cluster in jl_clusters]
    return py_clusters

def _getcluster_1(y: list[float], u_1: float = None, u_2: float = None) -> list[Cluster]:
    u_1, u_2 = jl_convert(jl.Real, u_1), jl_convert(jl.Real, u_2)
    return Extremes.getcluster(y, u_1, u_2) # juliacall.VectorValue
    
def _getcluster_2(y: list[float], u: float = None, runlength: int = None) -> list[Cluster]:
        u = jl_convert(jl.Real, u)
        return Extremes.getcluster(y, u, runlength=runlength) 

def jl_cluster_to_py_cluster(jl_cluster) -> Cluster:
    return Cluster(
        getattr(jl_cluster, 'u₁'), 
        getattr(jl_cluster, 'u₂'), 
        list(getattr(jl_cluster, 'position')), 
        list(getattr(jl_cluster, 'value'))
    )

