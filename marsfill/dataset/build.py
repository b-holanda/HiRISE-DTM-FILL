import os
import urllib
import urllib.request

from pathlib import Path
from urllib.error import HTTPError
from pystac_client import Client

from marsfill.utils import Logger


logger = Logger()

class Build:
    def __init__(
            self, 
            catalog_url: str, 
            collection: str, 
            download_dir: Path,
            samples: int,
            wanted_assets: str, 
            collection_nickname: str
        ) -> None:

        self._catalog_url = catalog_url
        self._collection = collection
        self._download_dir = download_dir
        self._samples = samples
        self._wanted_assets = wanted_assets
        self._collection_nickname = collection_nickname

    def _download_datasets(self, catalog: Client) -> None:
        response = catalog.search(collections=[self._collection], max_items=self._samples)

        logger.info(f"Encontrados {response.matched()} datasets")

        count = 1

        for item in response.items():
            download_path = self._download_dir / "sources" / self._collection_nickname / str(count)

            if not download_path.exists():
                download_path.mkdir(exist_ok=True, parents=True)

            logger.info(f"Processando dataset: {item.id} na local: {download_path} [{count}/{self._samples}]")

            for asset_key, asset in item.assets.items():
                if asset_key not in self._wanted_assets:
                    logger.info(f"Pulando arquivo desnecessário (key): {asset_key}")
                    continue
                
                try:
                    logger.info(f"Fazendo download do arquivo: {os.path.basename(asset.href)}")
                    urllib.request.urlretrieve(asset.href, download_path / f"{asset_key}.{os.path.basename(asset.href).split('.')[1]}")

                except HTTPError as e:
                    if e.code == 404:
                        logger.info(f"AVISO: Arquivo não encontrado (404): {asset.href}")
                        continue
                    raise e
                except Exception as e:
                    logger.info(f"ERRO: Falha ao baixar {asset.href}: {e}")
                    continue

            count += 1
 
    def run(self) -> None:
        catalog = Client.open(self._catalog_url)

        logger.info(f"Iniciando download de datasets fonte da coleção: {self._collection}")

        self._download_datasets(catalog=catalog)
