import os
import urllib
import urllib.request
import requests
import re
import random
import shutil

from pathlib import Path
from pystac_client import Client
import numpy as np
from osgeo import gdal

from marsfill.utils import Logger, CandidateFile, get_dtm_candidate, get_ortho_candidate

gdal.UseExceptions()

logger = Logger()

class Build:
    def __init__(
            self, 
            catalog_url: str, 
            collection: str, 
            download_dir: Path,
            samples: int,
            tile_size: int,
            stride: int
        ) -> None:

        self._catalog_url = catalog_url
        self._collection = collection
        self._download_dir = download_dir
        self._samples = samples
        self._tile_size = tile_size
        self._stride = stride
        self._datasets = []
        self._assignments = []

    def _list_datasets(self, catalog: Client) -> None:
        response = catalog.search(collections=[self._collection], max_items=self._samples)

        logger.info(f"Encontrados {response.matched()} datasets")

        ortho_pattern = re.compile(r"^ESP_\d{6}_\d{4}_RED_[A-Z]_\d{2}_ORTHO\.tif$")
        dtm_pattern = re.compile(r"^DTEPD_\d{6}_\d{4}_\d{6}_\d{4}_[A-Z]\d{2}\.tif$")

        for item in response.items():
            logger.info(f"Verificando dataset: {item.id}")

            pair = []

            for _, asset in item.assets.items():
                filename = os.path.basename(asset.href)

                if dtm_pattern.match(filename):
                    filename = "DTM.tif"
                elif ortho_pattern.match(filename):
                    filename = "ORTHO.tif"
                else:
                    continue

                response = requests.head(asset.href)

                if (response.status_code == 200):
                    pair.append(CandidateFile(filename=filename, href=asset.href))

                    if len(pair) == 2:
                        break

            self._datasets.append(pair)

    def _download_candidate(self, candidate: CandidateFile, out_dir: Path) -> None:
        logger.info(f"Baixando: {os.path.basename(candidate.href)}...")

        urllib.request.urlretrieve(candidate.href, out_dir)

    def _align_dtm_to_ortho(self, dtm: Path, ortho: Path, aligned: Path) -> None:
        logger.info("Gerando novo DTM alinhado com o arquivo orthoretificado")

        ortho_dataset = gdal.Open(ortho)
        ortho_geo_transform = ortho_dataset.GetGeoTransform()
        ortho_geo_projection = ortho_dataset.GetProjection()

        x_res = ortho_geo_transform[1]
        y_res = ortho_geo_transform[5]
        x_min = ortho_geo_transform[0]
        y_max = ortho_geo_transform[3]

        x_max = x_min + ortho_dataset.RasterXSize * x_res
        y_min = y_max + ortho_dataset.RasterYSize * y_res

        ortho_dataset = None

        gdal.Warp(
            aligned,
            dtm,
            format="GTiff",
            outputBounds=[x_min, y_min, x_max, y_max],
            xRes=x_res,
            yRes=abs(y_res),
            dstSRS=ortho_geo_projection,
            resampleAlg='cubic',
            creationOptions=['COMPRESS=LZW'],
        )

        os.remove(dtm)

    def _normalize_pair(self, ortho_arr, dtm_arr, x: int, y: int):
        ortho_tile = ortho_arr[y : y + self._tile_size, x : x + self._tile_size].astype(np.float32)
        dtm_tile = dtm_arr[y : y + self._tile_size, x : x + self._tile_size].astype(np.float32)

        min_o, max_o = ortho_tile.min(), ortho_tile.max()
        ortho_normalized = (ortho_tile - min_o) / (max_o - min_o + 1e-8)

        min_d, max_d = dtm_tile.min(), dtm_tile.max()
        dtm_normormalized = (dtm_tile - min_d) / (max_d - min_d + 1e-8)

        return ortho_normalized, dtm_normormalized

    def _prepare_assignments(self) -> None:
        """
        Calcula as contagens de 80/10/10 e cria uma lista
        de atribuições de destino embaralhada.
        """
        logger.info(
            f"Preparando {self._samples} atribuições (80% treino, 10% teste, 10% validação)"
        )

        test_count = int(np.round(0.1 * self._samples))
        val_count = int(np.round(0.1 * self._samples))

        train_count = self._samples - test_count - val_count

        logger.info(
            f"Contagens calculadas: Treino={train_count}, Teste={test_count}, Validação={val_count}"
        )

        self._assignments = (
            ['train'] * train_count +
            ['test'] * test_count +
            ['validation'] * val_count
        )
        
        # 4. Embaralhar a lista para garantir aleatoriedade
        random.shuffle(self._assignments)

    def _save_tile(self, normalized, base, projection, x, y, nodata, out_path):
        new_geo_transform = (
            base[0] + x * base[1],
            base[1],
            base[2],
            base[3] + y * base[5],
            base[4],
            base[5]
        )

        driver = gdal.GetDriverByName('GTiff')

        out_dataset = driver.Create(
            out_path,
            normalized.shape[1],
            normalized.shape[0],
            1,
            gdal.GDT_Float32
        )

        out_dataset.SetGeoTransform(new_geo_transform)
        out_dataset.SetProjection(projection)

        out_band = out_dataset.GetRasterBand(1)

        if nodata is not None:
            out_band.SetNoDataValue(nodata)

        out_band.WriteArray(normalized)
        out_band.FlushCache()
        out_dataset = None

    def _process_tiles(self, dtm: Path, ortho: Path, out_dir: Path, count: int):
        ortho_dataset = gdal.Open(ortho)
        dtm_dataset = gdal.Open(dtm)

        ortho_arr = ortho_dataset.ReadAsArray()
        dtm_arr = dtm_dataset.ReadAsArray()

        base_geo_transform = ortho_dataset.GetGeoTransform()
        base_projection = ortho_dataset.GetProjection()

        nodata_val = dtm_dataset.GetRasterBand(1).GetNoDataValue()
        if nodata_val is None:
            nodata_val = -3.4028234663852886e+38 
 
        nodata_mask = (dtm_arr == nodata_val) | np.isnan(dtm_arr)

        height, width = ortho_arr.shape
        tile_count = 0

        for y in range(0, height - self._tile_size, self._stride):
            for x in range(0, width - self._tile_size, self._stride):
                mask_tile_nodata = nodata_mask[y : y + self._tile_size, x : x + self._tile_size]

                if np.any(mask_tile_nodata):
                    logger.info("Pulando bloco pois contém nodata")

                    continue

                logger.info(f"Noralizando bloco {tile_count}")

                ortho_normalized, dtm_normormalized = self._normalize_pair(
                    ortho_arr=ortho_arr, 
                    dtm_arr=dtm_arr, 
                    x=x, 
                    y=y
                )

                ortho_tile_path = out_dir / f"ORTHO_P{count}_T{tile_count}.tif"
                dtm_tile_path = out_dir / f"DTM_P{count}_T{tile_count}.tif"

                output_nodata = -9999.0

                self._save_tile(
                    normalized=ortho_normalized, 
                    base=base_geo_transform, 
                    projection=base_projection, 
                    x=x, 
                    y=y, 
                    nodata=output_nodata, 
                    out_path=ortho_tile_path
                )
                self._save_tile(
                    normalized=dtm_normormalized,
                    base=base_geo_transform, 
                    projection=base_projection, 
                    x=x, 
                    y=y, 
                    nodata=output_nodata, 
                    out_path=dtm_tile_path
                )

                tile_count += 1
        return tile_count

    def run(self) -> None:
        catalog = Client.open(self._catalog_url)

        logger.info(f"Iniciando indexação de datasets fonte da coleção: {self._collection}")

        self._list_datasets(catalog=catalog)

        if len(self._datasets) < self._samples:
            logger.info(f"A coleção {self._collection} não possui datasets suficientes...")
            return

        logger.info(f"Indexados {len(self._datasets)} datasets")

        self._prepare_assignments()

        count = 1

        for pair in self._datasets:
            dtm_candidate = get_dtm_candidate(pair)
            ortho_candidate = get_ortho_candidate(pair)

            download_sources = self._download_dir / "sources" / str(count) 

            if not self._assignments:
                logger.info("Lista de atribuições vazia. Usando 'train' como padrão.")
                destination = "train"
            else:
                destination = self._assignments.pop()

            tiles_dir = self._download_dir / destination

            download_sources.mkdir(exist_ok=True, parents=True)
            tiles_dir.mkdir(exist_ok=True, parents=True)

            source_dtm_path = download_sources / dtm_candidate.filename
            source_ortho_path = download_sources / ortho_candidate.filename
            source_dtm_aligned = download_sources / "DTM_aligned.tif"

            try:
                logger.info(f"Fazendo download par [{count}/{self._samples}]")

                self._download_candidate(candidate=dtm_candidate, out_dir=source_dtm_path)
                self._download_candidate(candidate=ortho_candidate, out_dir=source_ortho_path)

                self._align_dtm_to_ortho(dtm=source_dtm_path, ortho=source_ortho_path, aligned=source_dtm_aligned)

                self._process_tiles(dtm=source_dtm_aligned, ortho=source_ortho_path, out_dir=tiles_dir, count=count)

            except Exception as e:
                if tiles_dir.exists():
                    shutil.rmtree(tiles_dir)
                raise e
            finally:
                if download_sources.exists():
                    shutil.rmtree(download_sources)
                count += 1
