import fire

from marsfill.cli.dataset import Dataset
from marsfill.cli.model import Model

class CLI:
    """Linha de comando para o Marsfill."""

    def dataset(self) -> Dataset:
        """Acessa as funcionalidades do dataset."""
        
        return Dataset()
    
    def model(self) -> Model:
        """Acessa as funcionalidades do modelo DPT-ViT."""

        return Model()

def main():
    fire.Fire(CLI)
