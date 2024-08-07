"""DataItem, Variable and VariableStd classes. Python equivalents to classes of the same names in Extremes.jl."""


class DataItem:
    """Abstract base class for Variable and VariableStd classes."""

    pass


class Variable(DataItem):
    r"""
    Variable class.

    Parameters
    ----------
    name : str
        Name of the Variable.
    value : list of float
        Values of the Variable.
    """

    name: str
    value: list[float]

    def __init__(self, name: str, value: list[float]):
        self.name, self.value = name, value

    def __repr__(self):
        """String representation of the Variable."""
        return f"\t\t\t\tname: {self.name}\n\t\t\t\tvalue: {self.value}\n"


class VariableStd(DataItem):
    r"""
    VariableStd class.

    Parameters
    ----------
    name : str
        Name of the VariableStd.
    value : list of float
        Values of the VariableStd.
    offset : float
        Offset applied to the values of the VariableStd.
    scale : float
        Scale factor applied to the values of the VariableStd.
    """

    name: str
    value: list[float]
    offset: float
    scale: float

    def __init__(self, name: str, value: list[float], offset: float, scale: float):
        self.name, self.value, self.offset, self.scale = name, value, offset, scale
