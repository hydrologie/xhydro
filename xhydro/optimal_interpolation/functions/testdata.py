"""Tools for searching for and acquiring test data."""

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from shutil import copy
from typing import Optional, Union
from urllib.error import HTTPError
from urllib.request import urlretrieve

from platformdirs import user_cache_dir
from xarray import Dataset
from xarray import open_dataset as _open_dataset

_default_cache_dir = user_cache_dir("xhydro_testing_data")

LOGGER = logging.getLogger("XHYDRO")

__all__ = [
    "get_file",
    "get_local_testdata",
    "open_dataset",
]


# FIXME: Flattered as I am to see xclim's code here, this should all be rewritten to use pooch for data fetching.


def file_md5_checksum(fname):
    """Check that the data respects the md5 checksum.

    Parameters
    ----------
    fname : str
        Path to the file to check.

    Returns
    -------
    str
        The md5 checksum of the file.
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def get_local_testdata(  # noqa
    patterns: Union[str, Sequence[str]],
    temp_folder: Union[str, Path],
    branch: str = "master",
    _local_cache: Union[str, Path] = _default_cache_dir,
) -> Union[Path, list[Path]]:
    """Copy specific testdata from a default cache to a temporary folder.

    Return files matching `pattern` in the default cache dir and move to a local temp folder.

    Parameters
    ----------
    patterns : str or Sequence of str
        Glob patterns, which must include the folder.
    temp_folder : str or Path
        Target folder to copy files and filetree to.
    branch : str
        For GitHub-hosted files, the branch to download from. Default: "master".
    _local_cache : str or Path
        Local cache of testing data.

    Returns
    -------
    Union[Path, List[Path]]
        The matched files found in the cache dir.
    """
    temp_paths = []

    if isinstance(patterns, str):
        patterns = [patterns]

    for pattern in patterns:
        potential_paths = [
            path for path in Path(temp_folder).joinpath(branch).glob(pattern)
        ]
        if potential_paths:
            temp_paths.extend(potential_paths)
            continue

        testdata_path = Path(_local_cache)
        if not testdata_path.exists():
            raise RuntimeError(f"{testdata_path} does not exists")
        paths = [path for path in testdata_path.joinpath(branch).glob(pattern)]
        if not paths:
            raise FileNotFoundError(
                f"No data found for {pattern} at {testdata_path}/{branch}."
            )

        main_folder = Path(temp_folder).joinpath(branch).joinpath(Path(pattern).parent)
        main_folder.mkdir(exist_ok=True, parents=True)

        for file in paths:
            temp_file = main_folder.joinpath(file.name)
            if not temp_file.exists():
                copy(file, main_folder)
            temp_paths.append(temp_file)

    # Return item directly when singleton, for convenience
    return temp_paths[0] if len(temp_paths) == 1 else temp_paths


def _get(
    fullname: Path,
    github_url: str,
    branch: str,
    cache_dir: Path,
) -> Path:
    """Get the file from a github repo."""
    cache_dir = cache_dir.absolute()
    local_file = cache_dir / branch / fullname

    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")

    if not local_file.is_file():
        # This will always leave this directory on disk.
        # We may want to add an option to remove it.
        local_file.parent.mkdir(parents=True, exist_ok=True)

        url = "/".join((github_url, "raw", branch, fullname.as_posix()))
        LOGGER.info(f"Fetching remote file: {fullname.as_posix()}")
        try:
            urlretrieve(url, local_file)  # nosec
        except HTTPError as e:
            msg = f"{local_file.name} not found. Aborting file retrieval."
            local_file.unlink()
            raise FileNotFoundError(msg) from e

    return local_file


# idea copied from xclim that borrowed it from xarray that was borrowed from Seaborn
def get_file(  # noqa
    name: Union[str, Path, Sequence[Union[str, Path]]],
    github_url: str = "https://github.com/hydrologie/xhydro-testdata",
    branch: str = "master",
    cache_dir: Union[str, Path] = _default_cache_dir,
) -> Union[Path, list[Path]]:
    """Return a file from an online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str or Path or Sequence of str or Path
        Name of the file or list/tuple of names of files containing the dataset(s) including suffixes.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str
        For GitHub-hosted files, the branch to download from. Default: "master".
    cache_dir : str or Path
        The directory in which to search for and write cached data.

    Returns
    -------
    Path or list of Path
        The path to a fetched file or a list of paths to fetched files.
    """
    if isinstance(name, (str, Path)):
        name = [name]

    cache_dir = Path(cache_dir)

    files = list()
    for n in name:
        fullname = Path(n)
        files.append(
            _get(
                fullname=fullname,
                github_url=github_url,
                branch=branch,
                cache_dir=cache_dir,
            )
        )
    if len(files) == 1:
        return files[0]
    return files


# idea copied from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str,
    suffix: Optional[str] = None,
    github_url: str = "https://github.com/hydrologie/xhydro-testdata",
    branch: str = "master",
    cache_dir: Union[str, Path] = _default_cache_dir,
    **kwds,
) -> Dataset:
    r"""Open a dataset from the online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset. If no suffix is given, assumed to be netCDF ('.nc' is appended).
    suffix : str, optional
        If no suffix is given, assumed to be netCDF ('.nc' is appended). For no suffix, set "".
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache_dir : str or Path
        The directory in which to search for and write cached data.
    \*\*kwds : dict
        For NetCDF files, keywords passed to xarray.open_dataset.

    Returns
    -------
    xr.Dataset
        The xarray Dataset object.
    """
    name = Path(name)
    cache_dir = Path(cache_dir)
    if suffix is None:
        suffix = ".nc"
    fullname = name.with_suffix(suffix)

    local_file = _get(
        fullname=fullname,
        github_url=github_url,
        branch=branch,
        cache_dir=cache_dir,
    )

    ds = _open_dataset(local_file, **kwds)
    return ds
