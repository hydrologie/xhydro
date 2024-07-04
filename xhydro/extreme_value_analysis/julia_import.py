from juliacall import Main as jl
from juliapkg import add

#TODO dynamic import?
add("Extremes", "fe3fe864-1b39-11e9-20b8-1f96fa57382d")
jl.seval("using Extremes")
Extremes = jl.Extremes

# add("MambaLite", "ee262687-6dc1-48e3-a463-c0a35b1bf9d0")
# jl.seval("using MambaLite")
# MambaLite = jl.MambaLite
