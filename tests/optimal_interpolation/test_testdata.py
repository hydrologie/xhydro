from pathlib import Path

import pytest
import xarray

from xhydro.optimal_interpolation.functions.testdata import _default_cache_dir  # noqa: F822
from xhydro.optimal_interpolation.functions.testdata import get_file, open_dataset, query_folder


class TestRemoteFileAccess:
    git_url = "https://github.com/Mayetea/xhydro-testdata"
    branch = "optimal_interpolation"

    @pytest.mark.online
    def test_get_file_default_cache(self):
        file = get_file(name="stations_retenues.csv", branch=self.branch)

        assert Path(_default_cache_dir).exists()
        assert file.is_file()
        with file.open() as f:
            header = f.read()
            assert ":FileType          rvi ASCII Raven 2.8.2" in header

    @pytest.mark.online
    def test_open_dataset(self):
        ds = open_dataset(
            name="raven-gr4j-cemaneige/Salmon-River-Near-Prince-George_meteo_daily.nc",
            branch=self.branch,
        )

        assert (
            Path(_default_cache_dir)
            .joinpath(
                self.branch,
                "raven-gr4j-cemaneige",
                "Salmon-River-Near-Prince-George_meteo_daily.nc",
            )
            .exists()
        )
        assert isinstance(ds, xarray.Dataset)

    @pytest.mark.online
    def test_open_dataset_false_cache(self):
        ds = open_dataset(
            name="raven-gr4j-cemaneige/Salmon-River-Near-Prince-George_meteo_daily_3d.nc",
            branch=self.branch,
            cache=False,
        )

        assert (
            not Path(_default_cache_dir)
            .joinpath(
                "raven-gr4j-cemaneige",
                "Salmon-River-Near-Prince-George_meteo_daily_3d.nc",
            )
            .exists()
        )
        assert isinstance(ds, xarray.Dataset)


class TestQueryFolder:
    git_url = "https://github.com/Ouranosinc/raven-testdata"
    branch = "master"

    @pytest.mark.online
    @pytest.mark.xfail(reason="Query folder is API rate limited")
    def test_query_specific_folder(self):
        folder = query_folder(folder="raven-gr4j-cemaneige", branch=self.branch)
        assert len(folder) == 8

    @pytest.mark.online
    @pytest.mark.xfail(reason="Query folder is API rate limited")
    def test_query_folder_patterns(self):
        mohyse = query_folder(
            folder="/regionalisation_data/tests/", pattern="MOHYSE", branch=self.branch
        )
        assert len(mohyse) == 1
        assert mohyse[0] == str(
            Path("regionalisation_data", "tests", "MOHYSE_parameters.csv")
        )

    @pytest.mark.online
    @pytest.mark.xfail(reason="Query folder is API rate limited")
    def test_query_folder_patterns_excessive_slashes(self):
        mohyse = query_folder(
            folder="///regionalisation_data/////tests///",
            pattern="MOHYSE",
            branch=self.branch,
        )
        assert len(mohyse) == 1
        assert mohyse[0] == str(
            Path("regionalisation_data", "tests", "MOHYSE_parameters.csv")
        )
