import fire

from marsfill.cli.dataset import Dataset

class CLI:
    """Linha de comando para o Marsfill."""

    def dataset(self) -> Dataset:
        """Acessa as funcionalidades do dataset."""
        
        return Dataset()

def main():
    fire.Fire(CLI)
