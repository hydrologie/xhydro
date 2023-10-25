# noqa: D100


import numpy as np
import pytest
from xclim.testing.helpers import test_timeseries as timeseries


@pytest.fixture
def make_meteo():
    def _make_meteo():
        meteo = timeseries(
            np.zeros(365 * 2),
            start="2001-01-01",
            freq="D",
            variable="tasmin",
            as_dataset=True,
            units="degC",
        )
        meteo["tasmax"] = timeseries(
            np.ones(365 * 2),
            start="2001-01-01",
            freq="D",
            variable="tasmax",
            units="degC",
        )
        meteo["pr"] = timeseries(
            np.ones(365 * 2) * 10,
            start="2001-01-01",
            freq="D",
            variable="pr",
            units="mm",
        )
        meteo = meteo.expand_dims("stations").assign_coords(stations=["010101"])
        meteo = meteo.assign_coords(coords={"lat": 46, "lon": -77, "z": 0})
        for c in ["lat", "lon", "z"]:
            meteo[c] = meteo[c].expand_dims("stations")
        return meteo

    return _make_meteo


@pytest.fixture
def make_debit_aval():
    def _make_debit_aval():
        debit_aval = timeseries(
            np.zeros(365 * 2),
            start="2001-01-01",
            freq="D",
            variable="streamflow",
            as_dataset=True,
        )
        debit_aval = debit_aval.expand_dims("troncon").assign_coords(troncon=[0])
        debit_aval = debit_aval.assign_coords(coords={"idtroncon": 0})
        debit_aval["idtroncon"] = debit_aval["idtroncon"].expand_dims("troncon")
        debit_aval = debit_aval.rename({"streamflow": "debit_aval"})
        debit_aval["debit_aval"].attrs = {
            "units": "m3/s",
            "description": "Debit en aval du troncon",
        }
        return debit_aval

    return _make_debit_aval
