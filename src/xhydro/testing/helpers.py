"""Helper functions for testing data management."""

import importlib.resources as ilr
import logging
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import IO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import urlretrieve

import pooch

from xhydro import __version__ as __xhydro_version__

__all__ = [
    "TESTDATA_BRANCH",
    "TESTDATA_CACHE_DIR",
    "TESTDATA_REPO_URL",
    "audit_url",
    "default_testdata_cache",
    "default_testdata_repo_url",
    "default_testdata_version",
    "deveraux",
    "load_registry",
    "populate_testing_data",
]

default_testdata_version = "v2025.3.31"
"""Default version of the testing data to use when fetching datasets."""

default_testdata_repo_url = (
    "https://raw.githubusercontent.com/hydrologie/xhydro-testdata/"
)
"""Default URL of the testing data repository to use when fetching datasets."""

try:
    default_testdata_cache = Path(pooch.os_cache("xhydro-testdata"))
    """Default location for the testing data cache."""
except AttributeError:
    default_testdata_cache = None


TESTDATA_REPO_URL = str(os.getenv("XCLIM_TESTDATA_REPO_URL", default_testdata_repo_url))
"""Sets the URL of the testing data repository to use when fetching datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XHYDRO_TESTDATA_REPO_URL="https://github.com/my_username/xhydro-testdata/"

or setting the variable at runtime:

.. code-block:: console

    $ env XHYDRO_TESTDATA_REPO_URL="https://github.com/my_username/xhydro-testdata/" pytest
"""

TESTDATA_BRANCH = os.getenv("XHYDRO_TESTDATA_BRANCH", default_testdata_version)
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

TESTDATA_CACHE_DIR = os.getenv("XHYDRO_TESTDATA_CACHE_DIR", default_testdata_cache)
"""Sets the directory to store the testing datasets.

If not set, the default location will be used (based on ``platformdirs``, see :func:`pooch.os_cache`).

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XHYDRO_TESTDATA_CACHE_DIR="/path/to/my/data"

or setting the variable at runtime:

.. code-block:: console

    $ env XHYDRO_TESTDATA_CACHE_DIR="/path/to/my/data" pytest
"""


def load_registry(
    branch: str = TESTDATA_BRANCH, repo: str = TESTDATA_REPO_URL
) -> dict[str, str]:
    """
    Load the registry file for the test data.

    Parameters
    ----------
    branch : str
        Branch of the repository to use when fetching testing datasets.
    repo : str
        URL of the repository to use when fetching testing datasets.

    Returns
    -------
    dict
        Dictionary of filenames and hashes.
    """
    if not repo.endswith("/"):
        repo = f"{repo}/"
    remote_registry = audit_url(
        urljoin(
            urljoin(repo, branch if branch.endswith("/") else f"{branch}/"),
            "data/registry.txt",
        )
    )

    if repo != default_testdata_repo_url:
        external_repo_name = urlparse(repo).path.split("/")[-2]
        external_branch_name = branch.split("/")[-1]
        registry_file = Path(
            str(
                ilr.files("xhydro").joinpath(
                    f"testing/registry.{external_repo_name}.{external_branch_name}.txt"
                )
            )
        )
        urlretrieve(remote_registry, registry_file)  # noqa: S310

    elif branch != default_testdata_version:
        custom_registry_folder = Path(
            str(ilr.files("xhydro").joinpath(f"testing/{branch}"))
        )
        custom_registry_folder.mkdir(parents=True, exist_ok=True)
        registry_file = custom_registry_folder.joinpath("registry.txt")
        urlretrieve(remote_registry, registry_file)  # noqa: S310

    else:
        registry_file = Path(str(ilr.files("xhydro").joinpath("testing/registry.txt")))

    if not registry_file.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_file}")

    # Load the registry file
    with registry_file.open(encoding="utf-8") as f:
        registry = {line.split()[0]: line.split()[1] for line in f}
    return registry


def deveraux(  # noqa: PR01
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    cache_dir: str | Path = TESTDATA_CACHE_DIR,
    data_updates: bool = True,
):
    """Pooch registry instance for xhydro test data.

    Parameters
    ----------
    repo : str
        URL of the repository to use when fetching testing datasets.
    branch : str
        Branch of repository to use when fetching testing datasets.
    cache_dir : str or Path
        The path to the directory where the data files are stored.
    data_updates : bool
        If True, allow updates to the data files. Default is True.

    Returns
    -------
    pooch.Pooch
        The Pooch instance for accessing the xhydro testing data.

    Notes
    -----
    There are three environment variables that can be used to control the behaviour of this registry:
        - ``XHYDRO_TESTDATA_CACHE_DIR``: If this environment variable is set, it will be used as the base directory to
          store the data files. The directory should be an absolute path (i.e., it should start with ``/``).
          Otherwise,the default location will be used (based on ``platformdirs``, see :py:func:`pooch.os_cache`).
        - ``XHYDRO_TESTDATA_REPO_URL``: If this environment variable is set, it will be used as the URL of the repository
          to use when fetching datasets. Otherwise, the default repository will be used.
        - ``XHYDRO_TESTDATA_BRANCH``: If this environment variable is set, it will be used as the branch of the repository
          to use when fetching datasets. Otherwise, the default branch will be used.

    Examples
    --------
    Using the registry to download a file:

    .. code-block:: python

        import xarray as xr
        from xhydro.testing.helpers import devereaux

        example_file = deveraux().fetch("example.nc")
        data = xr.open_dataset(example_file)
    """
    if pooch is None:
        raise ImportError(
            "The `pooch` package is required to fetch the xhydro testing data. "
            "You can install it with `pip install pooch` or `pip install xhydro[dev]`."
        )
    if not repo.endswith("/"):
        repo = f"{repo}/"
    remote = audit_url(
        urljoin(urljoin(repo, branch if branch.endswith("/") else f"{branch}/"), "data")
    )

    _devereaux = pooch.create(
        path=cache_dir,
        base_url=remote,
        version=default_testdata_version,
        version_dev=branch,
        allow_updates=data_updates,
        registry=load_registry(branch=branch, repo=repo),
    )

    # Add a custom fetch method to the Pooch instance
    # Needed to address: https://github.com/readthedocs/readthedocs.org/issues/11763
    _devereaux.fetch_diversion = _devereaux.fetch

    # Overload the fetch method to add user-agent headers
    @wraps(_devereaux.fetch_diversion)
    def _fetch(*args: str, **kwargs: bool | Callable) -> str:  # numpydoc ignore=GL08

        def _downloader(
            url: str,
            output_file: str | IO,
            poocher: pooch.Pooch,
            check_only: bool | None = False,
        ) -> None:
            """Download the file from the URL and save it to the save_path."""
            headers = {"User-Agent": f"xhydro ({__xhydro_version__})"}
            downloader = pooch.HTTPDownloader(headers=headers)
            return downloader(url, output_file, poocher, check_only=check_only)

        # default to our http/s downloader with user-agent headers
        kwargs.setdefault("downloader", _downloader)
        return _devereaux.fetch_diversion(*args, **kwargs)

    # Replace the fetch method with the custom fetch method
    _devereaux.fetch = _fetch

    return _devereaux


def populate_testing_data(
    temp_folder: Path | None = None,
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    local_cache: Path = TESTDATA_CACHE_DIR,
) -> None:
    """Populate the local cache with the testing data.

    Parameters
    ----------
    temp_folder : Path, optional
        Path to a temporary folder to use as the local cache. If not provided, the default location will be used.
    repo : str, optional
        URL of the repository to use when fetching testing datasets.
    branch : str, optional
        Branch of xhydro-testdata to use when fetching testing datasets.
    local_cache : Path
        The path to the local cache. Defaults to the location set by the platformdirs library.
        The testing data will be downloaded to this local cache.
    """
    # Create the Pooch instance
    n = deveraux(repo=repo, branch=branch, cache_dir=temp_folder or local_cache)

    # Download the files
    errored_files = []
    for file in load_registry():
        try:
            n.fetch(file, processor=pooch.Unzip())
        except HTTPError:  # noqa: PERF203
            msg = f"File `{file}` not accessible in remote repository."
            logging.error(msg)
            errored_files.append(file)
        else:
            logging.info("Files were downloaded successfully.")

    if errored_files:
        logging.error(
            "The following files were unable to be downloaded: %s",
            errored_files,
        )


# Testing Utilities
def audit_url(url: str, context: str | None = None) -> str:
    """Check if the URL is well-formed.

    Parameters
    ----------
    url : str
        The URL to check.
    context : str, optional
        Additional context to include in the error message.

    Returns
    -------
    str
        The URL if it is well-formed.

    Raises
    ------
    URLError
        If the URL is not well-formed.
    """
    msg = ""
    result = urlparse(url)
    if result.scheme == "http":
        msg = f"{context if context else ''} URL is not using secure HTTP: '{url}'".strip()
    if not all([result.scheme, result.netloc]):
        msg = f"{context if context else ''} URL is not well-formed: '{url}'".strip()

    if msg:
        logging.error(msg)
        raise URLError(msg)
    return url
