"""Tools for searching for and acquiring test data."""
import hashlib
import logging
import re
import warnings
from pathlib import Path
from shutil import copy
from typing import List, Optional, Sequence, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlretrieve

import requests
from platformdirs import user_cache_dir
from xarray import Dataset
from xarray import open_dataset as _open_dataset

_default_cache_dir = user_cache_dir("raven_testing_data")

LOGGER = logging.getLogger("RAVEN")

__all__ = [
    "get_local_testdata",
    "open_dataset",
    "query_folder",
    "get_file",
]


def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def get_local_testdata(
    patterns: Union[str, Sequence[str]],
    temp_folder: Union[str, Path],
    branch: str = "master",
    _local_cache: Union[str, Path] = _default_cache_dir,
) -> Union[Path, List[Path]]:
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
    suffix: str,
    cache_dir: Path,
) -> Path:
    cache_dir = cache_dir.absolute()
    local_file = cache_dir / branch / fullname
    md5_name = fullname.with_suffix(f"{suffix}.md5")
    md5_file = cache_dir / branch / md5_name

    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")

    if local_file.is_file():
        local_md5 = file_md5_checksum(local_file)
        try:
            url = "/".join((github_url, "raw", branch, md5_name.as_posix()))
            LOGGER.debug(f"Attempting to fetch remote file md5: {md5_name.as_posix()}")
            urlretrieve(url, md5_file)  # nosec
            with open(md5_file) as f:
                remote_md5 = f.read()
            if local_md5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"MD5 checksum for {local_file.as_posix()} does not match upstream md5. "
                    "Attempting new download."
                )
                warnings.warn(msg)
        except (HTTPError, URLError):
            msg = f"{md5_name.as_posix()} not accessible online. Unable to determine validity with upstream repo."
            warnings.warn(msg)

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

        url = "/".join((github_url, "raw", branch, md5_name.as_posix()))
        LOGGER.info(f"Fetching remote file md5: {md5_name.as_posix()}")
        try:
            urlretrieve(url, md5_file)  # nosec
        except HTTPError as e:
            msg = f"{md5_name.as_posix()} not found. Aborting file retrieval."
            local_file.unlink()
            raise FileNotFoundError(msg) from e

        local_md5 = file_md5_checksum(local_file)
        try:
            with open(md5_file) as f:
                remote_md5 = f.read()
            if local_md5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"{local_file.as_posix()} and md5 checksum do not match. "
                    "There may be an issue with the upstream origin data."
                )
                raise OSError(msg)
        except OSError as e:
            LOGGER.error(e)

    return local_file


# idea copied from xclim that borrowed it from xarray that was borrowed from Seaborn
def get_file(
    name: Union[str, Path, Sequence[Union[str, Path]]],
    github_url: str = "https://github.com/Ouranosinc/raven-testdata",
    branch: str = "master",
    cache_dir: Union[str, Path] = _default_cache_dir,
) -> Union[Path, List[Path]]:
    """
    Return a file from an online GitHub-like repository.
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
    """
    if isinstance(name, (str, Path)):
        name = [name]

    cache_dir = Path(cache_dir)

    files = list()
    for n in name:
        fullname = Path(n)
        suffix = fullname.suffix
        files.append(
            _get(
                fullname=fullname,
                github_url=github_url,
                branch=branch,
                suffix=suffix,
                cache_dir=cache_dir,
            )
        )
    if len(files) == 1:
        return files[0]
    return files


# Credits to Anselme  https://stackoverflow.com/a/62003257/7322852 (CC-BY-SA 4.0)
def query_folder(
    folder: Optional[str] = None,
    pattern: Optional[str] = None,
    github_url: str = "https://github.com/Ouranosinc/raven-testdata",
    branch: str = "master",
) -> List[str]:
    """
    Lists the files available for retrieval from a remote git repository with get_file.
    If provided a folder name, will perform a globbing-like filtering operation for parent folders.

    Parameters
    ----------
    folder : str, optional
        Relative pathname of the sub-folder from the top-level.
    pattern : str, optional
        Regex pattern to identify a file.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str
        For GitHub-hosted files, the branch to download from. Default: "master".

    Returns
    -------
    list of str
    """
    repo_name = github_url.strip("https://github.com/")

    url = f"https://api.github.com/repos/{repo_name}/git/trees/{branch}?recursive=1"
    r = requests.get(url)
    res = r.json()

    try:
        md5_files = [f["path"] for f in res["tree"] if f["path"].endswith(".md5")]
        if folder:
            folder = "/".join("/".split(folder)) if "/" in folder else folder
            md5_files = [f for f in md5_files if folder in Path(f).parent.as_posix()]
        files = [re.sub(".md5$", "", f) for f in md5_files]

        if pattern:
            regex = re.compile(pattern)
            files = [string for string in files if re.search(regex, string)]
    except KeyError:
        if {"message", "documentation_url"}.issubset(set(res.keys())):
            raise ConnectionRefusedError(res["message"])
        else:
            raise

    return files


# idea copied from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str,
    suffix: Optional[str] = None,
    dap_url: Optional[str] = None,
    github_url: str = "https://github.com/Ouranosinc/raven-testdata",
    branch: str = "master",
    cache: bool = True,
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
    dap_url : str, optional
        URL to OPeNDAP folder where the data is stored. If supplied, supersedes github_url.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache : bool
        If True, then cache data locally for use on subsequent calls.
    cache_dir : str or Path
        The directory in which to search for and write cached data.
    \*\*kwds
        For NetCDF files, keywords passed to xarray.open_dataset.

    Returns
    -------
    xr.Dataset

    See Also
    --------
    xarray.open_dataset
    """
    name = Path(name)
    cache_dir = Path(cache_dir)
    if suffix is None:
        suffix = ".nc"
    fullname = name.with_suffix(suffix)

    if dap_url is not None:
        dap_file = urljoin(dap_url, str(name))
        try:
            ds = _open_dataset(dap_file, **kwds)
            return ds
        except OSError:
            msg = "OPeNDAP file not read. Verify that service is available."
            LOGGER.error(msg)
            raise

    local_file = _get(
        fullname=fullname,
        github_url=github_url,
        branch=branch,
        suffix=suffix,
        cache_dir=cache_dir,
    )

    try:
        ds = _open_dataset(local_file, **kwds)
        if not cache:
            ds = ds.load()
            local_file.unlink()
        return ds
    except OSError:
        raise
