"""Load and install Julia dependancies into python environment."""

import contextlib
import io
import os
import sys
import warnings
from types import ModuleType
from typing import cast

import juliapkg
from juliacall import Main as jl  # noqa: N813

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


def check_function_output(func, expected_output, *args, **kwargs) -> bool:
    r"""
    Check if a function's output contains a specific sub-string.

    Parameters
    ----------
    func : callable
        The function to be tested.
    expected_output : str
        The sub-string to search for in the output of the function.
    \*args : tuple
        Positional arguments to pass to the function.
    \**kwargs : tuple
        Keyword arguments to pass to the function.

    Returns
    -------
    bool
        True if the expected_output is found in the function's output, False otherwise.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func(*args, **kwargs)
    output = f.getvalue()
    return expected_output in output


# juliapkg.rm("Extremes")
# juliapkg.resolve()
deps = {
    "Extremes": "fe3fe864-1b39-11e9-20b8-1f96fa57382d",
    "DataFrames": "a93c6f00-e57d-5684-b7b6-d8193f3e46c0",
}
for dependancy, uuid in deps.items():
    if not check_function_output(juliapkg.deps.status, dependancy):
        juliapkg.add(dependancy, uuid)
juliapkg.resolve()
jl = cast(ModuleType, jl)
jl_version = (
    jl.VERSION.major,
    jl.VERSION.minor,
    jl.VERSION.patch,
)  # not sure how this is useful, PySR only uses it for testing
jl.seval("using Extremes")
Extremes = jl.Extremes
