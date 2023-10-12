# noqa: D100
import shutil
from pathlib import Path

import pytest

pytest_datapath = Path(__file__).parent.parent / "_data"


@pytest.fixture(scope="session", autouse=True)
def cleanup_data_dir(request):
    """Remove the data folder after the tests are done."""

    def remove_data_folder():
        if pytest_datapath.exists():
            shutil.rmtree(pytest_datapath)

    request.addfinalizer(remove_data_folder)
