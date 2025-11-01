import os

from pathlib import Path
from marsfill.dataset.build import Build
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile_for_hardware

logger = Logger()

class Dataset:
    """Lidar com o dataset de treinamento e teste para o modelo Marsfill."""

    def __init__(self):
        pass

    def build( self) -> None:
        """Constrói o dataset de treinamento e teste e salva na pasta especificada.

        Args:
            samples (int): número de amostras. default 100
            output (str): Caminho para a pasta onde o dataset será salvo. default: dataset
            urls_to_scan (str): API com catálogo de datasets. default: https://www.uahirise.org/PDS/DTM/PSP/ e https://www.uahirise.org/PDS/DTM/ESP/
            tile_size (int): tamanho do recorte. default 512
            stride (int): passo da normalização. default 256
        """

        profile = get_profile_for_hardware()

        if not profile:
            logger.error("Nenhum perfil compatível encontrado para o hardware do sistema.")
            return

        samples: int = profile["make"].get("samples", 100)
        output: str = profile["make"].get("output", "datasets")
        urls_to_scan: list[str] = profile["make"].get("urls_to_scan", ["https://www.uahirise.org/PDS/DTM/PSP/", "https://www.uahirise.org/PDS/DTM/ESP/"])
        tile_size: int = profile["make"].get("tile_size", 512)
        stride: int = profile["make"].get("stride", 256)

        download_dir = os.path.join(Path(__file__).parent.parent.parent, output)

        logger.info("Iniciando montagem de dataset")
        logger.info(f"samples={samples}")
        logger.info(f"output={output}")
        logger.info(f"urls_to_scan={urls_to_scan}")
        logger.info(f"tile_size={tile_size}")
        logger.info(f"stride={stride}")

        Build(
            urls_to_scan=urls_to_scan, 
            download_dir=Path(download_dir), 
            samples=samples,
            tile_size=tile_size,
            stride=stride,
        ).run()

if __name__ == "__main__":
    dataset = Dataset()

    dataset.build()
