"""Helper functions for testing data management."""

import importlib.resources as ilr
import logging
import os
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urljoin

import pooch

from xhydro import __version__ as __xhydro_version__

__all__ = [
    "DATA_DIR",
    "DATA_URL",
    "DEVEREAUX",
    "generate_registry",
    "load_registry",
    "populate_testing_data",
]

_default_cache_dir = pooch.os_cache("xhydro-testdata")

DATA_DIR = os.getenv("XHYDRO_DATA_DIR", _default_cache_dir)
"""Sets the directory to store the testing datasets.

If not set, the default location will be used (based on ``platformdirs``, see :func:`pooch.os_cache`).

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XHYDRO_DATA_DIR="/path/to/my/data"

or setting the variable at runtime:

.. code-block:: console

    $ env XHYDRO_DATA_DIR="/path/to/my/data" pytest
"""

TESTDATA_BRANCH = os.getenv("XHYDRO_TESTDATA_BRANCH", "main")
"""Sets the branch of hydrologie/xhydro-testdata to use when fetching testing datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XHYDRO_TESTDATA_BRANCH="my_testing_branch"

or setting the variable at runtime:

.. code-block:: console

    $ env XHYDRO_TESTDATA_BRANCH="my_testing_branch" pytest
"""

DATA_URL = f"https://github.com/hydrologie/xhydro-testdata/raw/{TESTDATA_BRANCH}"


def generate_registry(
    filenames: list[str] | None = None, base_url: str = DATA_URL
) -> None:
    """Generate a registry file for the test data.

    Parameters
    ----------
    filenames : list of str, optional
        List of filenames to generate the registry file for.
        If not provided, all files under xhydro/testing/data will be used.
    base_url : str, optional
        Base URL to the test data repository.
    """
    # Gather the data folder and registry file locations from installed package_data
    data_folder = ilr.files("xhydro").joinpath("testing/data")
    registry_file = ilr.files("xhydro").joinpath("testing/registry.txt")

    # Download the files to the installed xhydro/testing/data folder
    if filenames is None:
        with ilr.as_file(data_folder) as data:
            for file in data.rglob("*"):
                filename = file.relative_to(data).as_posix()
                pooch.retrieve(
                    url=urljoin(base_url, filename),
                    known_hash=None,
                    fname=filename,
                    path=data_folder,
                )

    # Generate the registry file
    with ilr.as_file(data_folder) as data, ilr.as_file(registry_file) as registry:
        pooch.make_registry(data.as_posix(), registry.as_posix())


def load_registry(file: str | Path | None = None) -> dict[str, str]:
    """Load the registry file for the test data.

    Parameters
    ----------
    file : str or Path, optional
        Path to the registry file. If not provided, the registry file found within the package data will be used.

    Returns
    -------
    dict
        Dictionary of filenames and hashes.
    """
    # Get registry file from package_data
    if file is None:
        registry_file = ilr.files("xhydro").joinpath("testing/registry.txt")
        if registry_file.is_file():
            logging.info("Registry file found in package_data: %s", registry_file)
    else:
        registry_file = Path(file)
        if not registry_file.is_file():
            raise FileNotFoundError(f"Registry file not found: {registry_file}")

    # Load the registry file
    registry = dict()
    with registry_file.open() as buffer:
        for entry in buffer.readlines():
            registry[entry.split()[0]] = entry.split()[1]

    return registry


DEVEREAUX = pooch.create(
    path=pooch.os_cache("xhydro-testdata"),
    base_url=DATA_URL,
    version=__xhydro_version__,
    version_dev="main",
    env="XHYDRO_DATA_DIR",
    allow_updates="XHYDRO_DATA_UPDATES",
    registry=load_registry(),
)
"""Pooch registry instance for xhydro test data.

Notes
-----
There are two environment variables that can be used to control the behaviour of this registry:

  - ``XHYDRO_DATA_DIR``: If this environment variable is set, it will be used as the base directory to store the data
    files. The directory should be an absolute path (i.e., it should start with ``/``). Otherwise,
    the default location will be used (based on ``platformdirs``, see :func:`pooch.os_cache`).

  - ``XHYDRO_DATA_UPDATES``: If this environment variable is set, then the data files will be downloaded even if the
    upstream hashes do not match. This is useful if you want to always use the latest version of the data files.

Examples
--------
Using the registry to download a file:

.. code-block:: python

    from xhydro.testing.utils import DEVEREAUX
    import xarray as xr

    example_file = DEVEREAUX.fetch("example.nc")
    data = xr.open_dataset(example_file)
"""


def populate_testing_data(
    registry: str | Path | None = None,
    temp_folder: Path | None = None,
    branch: str = TESTDATA_BRANCH,
    _local_cache: Path = _default_cache_dir,
) -> None:
    """Populate the local cache with the testing data.

    Parameters
    ----------
    registry : str or Path, optional
        Path to the registry file. If not provided, the registry file from package_data will be used.
    temp_folder : Path, optional
        Path to a temporary folder to use as the local cache. If not provided, the default location will be used.
    branch : str, optional
        Branch of hydrologie/xhydro-testdata to use when fetching testing datasets.
    _local_cache : Path, optional
        Path to the local cache. Defaults to the default location.

    Returns
    -------
    None
        The testing data will be downloaded to the local cache.
    """
    # Get registry file from package_data or provided path
    registry = load_registry(registry)

    # Set the local cache to the temp folder
    if temp_folder is not None:
        _local_cache = temp_folder
    # Set the branch
    DEVEREAUX.version_dev = branch
    # Set the local cache
    DEVEREAUX.path = _local_cache

    # Download the files
    for filename in registry.keys():
        DEVEREAUX.fetch(filename)
