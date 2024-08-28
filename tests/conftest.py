"""Pytest configuration for xHydro tests."""

import os
from pathlib import Path

import pytest
from xclim.testing.utils import open_dataset as _open_dataset


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory) -> Path:
    yield Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def open_dataset(threadsafe_data_dir):
    # FIXME: This is a temporary fix against the latest xclim-testdata release. It should be removed once xclim itself is updated.
    def _open_session_scoped_file(
        file: str | os.PathLike, branch: str = "v2023.12.14", **xr_kwargs
    ):
        xr_kwargs.setdefault("engine", "h5netcdf")
        return _open_dataset(
            file, cache_dir=threadsafe_data_dir, branch=branch, **xr_kwargs
        )

    return _open_session_scoped_file
