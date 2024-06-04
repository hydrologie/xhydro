class ReturnLevelModel:
    pass

class ReturnLevel:
    model: ReturnLevelModel # placeholder for the julia model, not translated into python
    returnperiod: float
    value: list[float]

    def __init__(self, model: ReturnLevelModel, returnperiod: float, value: list[float]):
        self.model = model
        self.returnperiod = returnperiod
        self.value = value

    def __repr__(self) -> str:
        return f"ReturnLevel:\nreturnperiod={self.returnperiod}, value={self.value})"

