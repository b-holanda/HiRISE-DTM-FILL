import os

from pathlib import Path
from marsfill.dataset.build import Build
from marsfill.utils import Logger

logger = Logger()

class Dataset:
    """Lidar com o dataset de treinamento e teste para o modelo Marsfill."""

    def __init__(self):
        pass

    def build(
            self, 
            samples: int,
            output: str = "datasets",
            catalog_url: str = "https://stac.astrogeology.usgs.gov/api", 
            collection: str = "mro_ctx_controlled_usgs_dtms",
            collection_nickname = "ctx",
            wanted_assets = ["orthoimage", "geoid_adjusted_dem", "thumbnail"]
        ) -> None:
        """Constrói o dataset de treinamento e teste e salva na pasta especificada.

        Args:
            output (str): Caminho para a pasta onde o dataset será salvo. default: dataset
            catalog_url (str): API com catálogo de datasets. default: https://stac.astrogeology.usgs.gov/api
            collection (str): coleção de datasets. default: mro_ctx_controlled_usgs_dtms
        """

        download_dir = os.path.join(Path(__file__).parent.parent.parent, output)

        logger.info("Iniciando montagem de dataset")
        logger.info(f"output={output}")
        logger.info(f"catalog_url={catalog_url}")
        logger.info(f"collection={collection}")
        logger.info(f"samples={samples}")

        Build(
            catalog_url=catalog_url, 
            collection=collection, 
            download_dir=Path(download_dir), 
            samples=samples,
            collection_nickname=collection_nickname,
            wanted_assets=wanted_assets
        ).run()
