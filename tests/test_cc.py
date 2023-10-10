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
