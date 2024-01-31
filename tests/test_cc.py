import numpy as np
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro as xh


# Smoke test for xscen functions that are imported into xhydro
def test_xscen_imported():
    assert all(
        callable(getattr(xh.cc, f))
        for f in [
            "climatological_op",
            "compute_deltas",
            "ensemble_stats",
            "produce_horizon",
        ]
    )


def test_climatological_op():
    ds = timeseries(
        np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
        variable="streamflow",
        start="2001-01-01",
        freq="YS",
        as_dataset=True,
    )
    # Test that climatological_op returns 5
    assert xh.cc.climatological_op(ds) == 5
