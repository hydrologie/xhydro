"""Conversion functions between Julia and Python objects."""

import numpy as np

try:
    from juliacall import convert as jl_convert

    from xhydro.extreme_value_analysis.julia_import import jl
    from xhydro.extreme_value_analysis.structures.dataitem import Variable

    __all__ = [
        "jl_matrix_tuple_to_py_list",
        "jl_vector_to_py_list",
        "jl_vector_tuple_to_py_list",
        "py_list_to_jl_vector",
        "py_str_to_jl_symbol",
        "py_variable_to_jl_variable",
    ]
except ImportError as e:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    raise ImportError(JULIA_WARNING) from e


def py_variable_to_jl_variable(py_var: Variable):
    r"""
    Convert a python Variable object to a Julia Variable object.

    Parameters
    ----------
    py_var : Variable
        The Variable object to convert.

    Returns
    -------
    julia.Extremes.Variable
        The converted Julia Variable object.
    """
    return jl.Extremes.Variable(
        py_var.name, jl_convert(jl.Vector[jl.Real], py_var.value)
    )


def py_str_to_jl_symbol(string: str):
    r"""
    Convert a python string to a Julia Symbol object.

    Parameters
    ----------
    string : str
        The string object to convert.

    Returns
    -------
    julia.Symbol
        The converted Julia Symbol object.
    """
    return jl.Symbol(string)


def py_list_to_jl_vector(py_list: list):
    r"""
    Convert a python list to a Julia Vector object.

    Parameters
    ----------
    py_list : list
        The list object to convert.

    Returns
    -------
    julia.Vector
        The converted Julia Vector object.

    Raises
    ------
    ValueError
        If the list contains mixed types that are not all strings or numbers.
        If the list contains unsupported types/complex structures such that it cannot be converted to a Julia Vector.
    """
    py_list = [float(i) if isinstance(i, np.float32) else i for i in py_list]

    if all(isinstance(i, float) or isinstance(i, int) for i in py_list):
        return jl_convert(jl.Vector[jl.Real], py_list)
    elif all(isinstance(i, str) for i in py_list):
        return jl_convert(jl.Vector[jl.String], py_list)
    elif not (
        all(isinstance(i, float) or isinstance(i, int) for i in py_list)
    ) and not (all(isinstance(i, str) for i in py_list)):
        raise ValueError(
            f" Cannot convert unsupported type {type(py_list)} to julia vector: all values are not strings or numbers"
        )
    else:
        raise ValueError(
            f" Cannot convert unsupported type {type(py_list)} to julia vector"
        )


def jl_vector_to_py_list(jl_vector) -> list:
    r"""
    Convert a Julia vector to a python list.

    Parameters
    ----------
    jl_vector : julia.Vector
        The julia.Vector object to convert.

    Returns
    -------
    list
        The converted list object.
    """
    return list(jl_vector)


def jl_vector_tuple_to_py_list(jl_vector_tuple) -> list:
    r"""
    Convert a julia vector containing a single tuple (i.e. [(1,2,3)]) to a python list (i.e. [1,2,3]).

    Parameters
    ----------
    jl_vector_tuple : julia.Vector[Tuple]
        The julia.Vector object to convert.

    Returns
    -------
    list
        The converted list object.
    """
    (jl_sub_tuple,) = jl_vector_tuple  # Unpack the single tuple from the list
    py_sub_list = list(jl_sub_tuple)
    return py_sub_list


def jl_matrix_tuple_to_py_list(jl_matrix_tuple) -> list[tuple]:
    r"""
    Convert a julia matrix of tuples to a python list of tuples.

    Parameters
    ----------
    jl_matrix_tuple : julia.Matrix[Tuple]
        The julia.Matrix[Tuple] object to convert.

    Returns
    -------
    list[tuple]
        The converted list[tuple] object.
    """
    py_list = [tuple(row) for row in jl_matrix_tuple]
    return py_list
