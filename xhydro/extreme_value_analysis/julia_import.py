from juliacall import Main as jl
from juliapkg import add

add("Extremes", "fe3fe864-1b39-11e9-20b8-1f96fa57382d")
jl.seval("using Extremes")
Extremes = jl.Extremes
