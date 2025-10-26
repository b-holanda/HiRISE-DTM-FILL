class Dataset:
    """Lidar com o dataset de treinamento e teste para o modelo Marsfill."""

    def __init__(self):
        pass

    def build(self, output: str = "dataset"):
        """Constrói o dataset de treinamento e teste e salva na pasta especificada.

        Args:
            output (str): Caminho para a pasta onde o dataset será salvo.
        """
        print(f"Construindo o dataset e salvando em {output}...")
