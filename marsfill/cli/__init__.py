from marsfill.cli.dataset import Dataset
from marsfill.cli.model import Model

class CLI:
    def __init__(self):
        pass

    def dataset() -> Dataset:
        return Dataset()
    
    def model() -> Model:
        return Model()
