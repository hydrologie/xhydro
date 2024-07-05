class DataItem:
    pass

class Variable(DataItem):
    name: str
    value: list[float]

    def __init__(self, name: str, value: list[float]):
        self.name, self.value = name, value
    
    def __repr__(self):
        return f"\t\t\t\tname: {self.name}\n\t\t\t\tvalue: {self.value}\n"

class VariableStd(DataItem):
    name: str
    value: list[float]
    offset: float
    scale: float

    def __init__(self, name: str, value: list[float], offset: float, scale: float):
        self.name, self.value, self.offset, self.scale = name, value, offset, scale





    