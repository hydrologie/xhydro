import os

from conftest import pytest_datapath

from xhydro.modelling import Hydrotel


class TestHydrotel:
    @staticmethod
    def make_tmp_dir(name, make_resultats=True):
        os.makedirs(pytest_datapath / name, exist_ok=True)
        os.makedirs(pytest_datapath / name / "meteo", exist_ok=True)
        os.makedirs(pytest_datapath / name / "simulation" / "simulation", exist_ok=True)
        if make_resultats:
            os.makedirs(
                pytest_datapath / name / "simulation" / "simulation" / "resultats",
                exist_ok=True,
            )

    def test_hydrotel(self):
        # TODO: This needs to write to a temporary directory.
        pass
