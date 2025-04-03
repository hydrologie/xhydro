"""Load and install Julia dependencies into python environment."""

import contextlib
import io
import os
import sys
import warnings
from types import ModuleType
from typing import cast

try:
    import juliapkg
    from juliacall import Main as jl  # noqa: N813

except (ImportError, ModuleNotFoundError) as e:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    raise ImportError(JULIA_WARNING) from e

__all__ = ["Extremes", "jl"]

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

    # TODO: Remove these when juliapkg lets you specify this
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        ("PYTHON_JULIACALL_THREADS", "auto"),
        ("PYTHON_JULIACALL_OPTLEVEL", "3"),
    ):
        os.environ[k] = os.environ.get(k, default)

    # Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "no") not in ["yes", ""]:
        warnings.warn(
            "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. "
            + "You will experience segfaults if running with multithreading."
        )

    if os.environ.get("PYTHON_JULIACALL_THREADS", "no") != "auto":
        warnings.warn(
            "PYTHON_JULIACALL_THREADS environment variable is set to something other than 'auto', "
            "so xhydro was not able to set it. You may wish to set it to `'auto'` for full use "
            "of your CPU."
        )


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


# It was not necessary to add a dependency dictionary as we only need Extremes.jl, however this mechanism is more
# scalable in case we need to add many other julia dependencies in the future
deps = {
    "Optim": {"uuid": "429524aa-4258-5aef-a3af-852621145aeb", "version": "=1.11.0"},
    "Extremes": {"uuid": "fe3fe864-1b39-11e9-20b8-1f96fa57382d", "version": "=1.0.3"},
}
for dependency, info in deps.items():
    juliapkg.add(dependency, info["uuid"], version=info.get("version"))

juliapkg.resolve()
jl = cast(ModuleType, jl)
jl_version = (
    jl.VERSION.major,
    jl.VERSION.minor,
    jl.VERSION.patch,
)  # NOTE: this is not used right now, but could be used for debugging purposes
jl.seval("using Extremes")
Extremes = jl.Extremes
