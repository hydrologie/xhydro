import numpy as np

import xhydro.optimal_interpolation.utilities as util


def test_general_ecf():
    h = np.array([0, 1, 2])
    param = np.array([0.5, 50])

    # Test the three forms for the general_ecf function
    assert np.allclose(
        util.general_ecf(h, param, form=1), np.array([0.5, 0.49990132, 0.49961051])
    )
    assert np.allclose(
        util.general_ecf(h, param, form=2), np.array([0.5, 0.49990001, 0.49960016])
    )
    assert np.allclose(
        util.general_ecf(h, param, form=3), np.array([0.5, 0.49009934, 0.48039472])
    )
