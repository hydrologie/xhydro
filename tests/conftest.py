# noqa: D100

from pathlib import Path

import pandas as pd
import pytest
from pooch import Unzip

from xhydro.testing.helpers import deveraux


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory) -> Path:
    data_dir = Path(tmp_path_factory.getbasetemp().joinpath("data"))
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def genextreme_data(threadsafe_data_dir):
    extremes_data_folder = threadsafe_data_dir.joinpath("extremes_value_analysis")

    ge = deveraux().fetch(
        "extreme_value_analysis/genextreme.zip",
        processor=Unzip(extract_dir=extremes_data_folder.absolute().as_posix()),
    )

    mappings = dict()
    mappings["gev_nonstationary"] = pd.read_csv(ge[0])
    mappings["gev_stationary"] = pd.read_csv(ge[1])
    return mappings


@pytest.fixture(scope="session")
def genpareto_data(threadsafe_data_dir):
    extremes_data_folder = threadsafe_data_dir.joinpath("extremes_value_analysis")

    gp = deveraux().fetch(
        "extreme_value_analysis/genpareto.zip",
        processor=Unzip(extract_dir=extremes_data_folder.absolute().as_posix()),
    )

    mappings = dict()
    mappings["gp_nonstationary"] = pd.read_csv(gp[0])
    mappings["gp_stationary"] = pd.read_csv(gp[1])
    return mappings
