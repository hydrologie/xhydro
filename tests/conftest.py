# noqa: D100

from pathlib import Path

import pandas as pd
import pytest
from pooch import Unzip

from xhydro.testing.helpers import DEVEREAUX


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory) -> Path:
    yield Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def genextreme_data(threadsafe_data_dir):

    genextreme_data = Unzip(
        DEVEREAUX.fetch("extreme_value_analysis/genextreme.zip"),
        extract_dir=threadsafe_data_dir,
    )
    genpareto_data = Unzip(
        DEVEREAUX.fetch("extreme_value_analysis/genpareto.zip"),
        extract_dir=threadsafe_data_dir,
    )

    GEV_NONSTATIONARY = pd.read_csv(genextreme_data[0])
    GEV_STATIONARY = pd.read_csv(genextreme_data[1])
    GP_NONSTATIONARY = pd.read_csv(genpareto_data[0])
    GP_STATIONARY = pd.read_csv(genpareto_data[1])

    return GEV_NONSTATIONARY, GEV_STATIONARY, GP_NONSTATIONARY, GP_STATIONARY
