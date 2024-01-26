from pathlib import Path

import pytest
import xarray

from xhydro.optimal_interpolation.functions.testdata import _default_cache_dir  # noqa: F822
from xhydro.optimal_interpolation.functions.testdata import get_file, open_dataset


class TestRemoteFileAccess:
    git_url = "https://github.com/Mayetea/xhydro-testdata"
    branch = "optimal_interpolation"

    @pytest.mark.online
    def test_get_file(self):
        file = get_file(name="data/optimal_interpolation/stations_retenues_validation_croisee.csv",
                        github_url=self.git_url,
                        branch=self.branch)

        assert Path(_default_cache_dir).joinpath(
                self.branch,
                "data/optimal_interpolation/stations_retenues_validation_croisee.csv",
            ).exists()
        assert file.is_file()
        with file.open() as f:
            header = f.read()
            assert header is not None

    @pytest.mark.online
    def test_open_dataset(self):
        ds = open_dataset(
            name="data/optimal_interpolation/A20_HYDOBS_TEST.nc",
            github_url=self.git_url,
            branch=self.branch,
        )

        assert (
            Path(_default_cache_dir)
            .joinpath(
                self.branch,
                "data/optimal_interpolation",
                "A20_HYDOBS_TEST.nc",
            )
            .exists()
        )
        assert isinstance(ds, xarray.Dataset)

