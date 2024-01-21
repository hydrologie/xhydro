import xhydro.optimal_interpolation.functions.utilities as util
import numpy as np

def test_convert_tuple_to_dict():
    tuple_to_convert = [['test', 0],[1,'test']]
    assert util.convert_list_to_dict(tuple_to_convert) == {'test':0, 1:'test'}

def test_initialize_nan_arrays():
    dimensions = (2, 3)
    quantities = 4
    return_values = util.initialize_nan_arrays(dimensions, quantities)

    assert len(return_values) == quantities
    assert len(return_values[0]) == 2
    assert len(return_values[0][0]) == 3

def test_general_ecf():
    h = np.array([0, 1, 2])
    param = np.array([0.5, 50])
    form = 1

    assert np.allclose(util.general_ecf(h, param, form), np.array([0.5, 0.49990132, 0.49961051]))

    form = 2

    assert np.allclose(util.general_ecf(h, param, form), np.array([0.5, 0.49990001, 0.49960016]))

    form = 3

    assert np.allclose(util.general_ecf(h, param, form), np.array([0.5, 0.49009934, 0.48039472]))

