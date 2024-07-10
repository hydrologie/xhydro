import juliapkg
from juliacall import Main as jl
import os
import sys
import warnings
from types import ModuleType
from typing import cast
import io
import contextlib

# Check if JuliaCall is already loaded, and if so, warn the user
# about the relevant environment variables. If not loaded,
# set up sensible defaults.
if "juliacall" in sys.modules:
    warnings.warn(
        "juliacall module already imported. "
        "Make sure that you have set the environment variable `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` to avoid segfaults. "
        "Also note that xhydro will not be able to configure `PYTHON_JULIACALL_THREADS` or `PYTHON_JULIACALL_OPTLEVEL` for you."
    )
else:
    # Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") != "yes":
        warnings.warn(
            "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. "
            + "You will experience segfaults if running with multithreading."
        )

    if os.environ.get("PYTHON_JULIACALL_THREADS", "auto") != "auto":
        warnings.warn(
            "PYTHON_JULIACALL_THREADS environment variable is set to something other than 'auto', "
            "so xhydro was not able to set it. You may wish to set it to `'auto'` for full use "
            "of your CPU."
        )

    # TODO: Remove these when juliapkg lets you specify this
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        ("PYTHON_JULIACALL_THREADS", "auto"),
        ("PYTHON_JULIACALL_OPTLEVEL", "3"),
    ):
        os.environ[k] = os.environ.get(k, default)

# check if a function prints out a sub-string or not
def check_function_output(func, expected_output, *args, **kwargs):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func(*args, **kwargs)
    output = f.getvalue()
    return expected_output in output

# juliapkg.rm("Extremes")
# juliapkg.resolve()
if not check_function_output(juliapkg.deps.status, "Extremes"):
    juliapkg.add("Extremes", "fe3fe864-1b39-11e9-20b8-1f96fa57382d")
    juliapkg.resolve()
jl = cast(ModuleType, jl)
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch) # not sure how this is useful, PySR only uses it for testing
jl.seval("using Extremes")
Extremes = jl.Extremes



