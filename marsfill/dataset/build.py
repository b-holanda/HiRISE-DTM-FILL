import os
import shutil
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
        response = catalog.search(collections=[self._collection])

        logger.info(f"Encontrados {response.matched()} datasets")

        count = 1
        datasets = {}

        for item in response.items():
            download_path = self._download_dir / "sources" / self._collection_nickname / str(count)

            if not download_path.exists():
                download_path.mkdir(exist_ok=True, parents=True)

            logger.info(f"Baixando: {download_path} [{count}/{self._samples}]")

            datasets[download_path] = {
                "invalid": 0,
                "valid": 0,
            }

            for asset_key, asset in item.assets.items():
                
                logger.info(asset_key)

                if asset_key not in self._wanted_assets:
                    continue

                try:
                    filename = f"{asset_key}.{os.path.basename(asset.href).split('.')[1]}"
                    download_full_path = download_path / filename

                    if download_full_path.exists():
                        datasets[download_path]["valid"] += 1

                        continue

                    urllib.request.urlretrieve(asset.href, download_full_path)

                    datasets[download_path]["valid"] += 1

                except HTTPError as e:
                    if e.code == 404:
                        datasets[download_path]["invalid"] += 1

                        continue
                    raise e

            if datasets[download_path]["invalid"] > 0:
                logger.info(f"Removendo: {download_path} inválido")

                shutil.rmtree(download_path)

                continue

            if (count == self._samples):
                break

            count += 1
 
    def run(self) -> None:
        catalog = Client.open(self._catalog_url)

        logger.info(f"Iniciando download de datasets fonte da coleção: {self._collection}")

        self._download_datasets(catalog=catalog)
