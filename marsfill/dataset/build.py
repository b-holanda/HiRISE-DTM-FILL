import os
import time
import requests
import random
import shutil

from pathlib import Path
import numpy as np
from osgeo import gdal

from marsfill.utils import Logger
from marsfill.dataset.hirise_indexer import ProductPair, HirisePDSIndexerDFS

gdal.UseExceptions()

logger = Logger()

class Build:
    def __init__(
            self, 
            urls_to_scan: list[str], 
            download_dir: Path,
            samples: int,
            tile_size: int,
            stride: int
        ) -> None:
        self._urls_to_scan = urls_to_scan
        self._download_dir = download_dir
        self._samples = samples
        self._tile_size = tile_size
        self._stride = stride
        self._assignments = []

    def _list_datasets(self) -> list[ProductPair]:
        indexer = HirisePDSIndexerDFS(self._urls_to_scan)

        return indexer.index_pairs(max_pairs=self._samples)
    def _download_candidate(
        self, candidate: str, out_dir: Path, retries: int = 3, backoff_factor: float = 0.5
    ) -> None:
        logger.info(f"Baixando: {os.path.basename(candidate)}...")

        for attempt in range(retries):
            try:
                with requests.get(candidate, stream=True, timeout=(15, 300)) as r:
                    r.raise_for_status()

                    with open(out_dir, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                        
                    logger.info(f"Download concluído para: {os.path.basename(candidate)}")
                    return

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ChunkedEncodingError,
                    BrokenPipeError) as e:

                logger.info(
                    f"Tentativa {attempt + 1}/{retries} falhou para {candidate} com erro: {e}"
                )

                if attempt + 1 == retries:
                    logger.info(f"Falha final no download após {retries} tentativas.")
                    raise e
        
                sleep_time = backoff_factor * (2 ** attempt)
                logger.info(f"Aguardando {sleep_time:.2f}s para tentar novamente...")
                time.sleep(sleep_time)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                logger.info(f"Erro HTTP {status} para {candidate}. Não haverá retentativa.")
                raise e

    def _align_dtm_to_ortho(self, dtm: Path, ortho: Path, aligned: Path) -> None:
        logger.info("Gerando novo DTM alinhado com o arquivo orthoretificado")

        ortho_dataset = gdal.Open(str(ortho))
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
            str(aligned),
            str(dtm),
            format="GTiff",
            outputBounds=[x_min, y_min, x_max, y_max],
            xRes=x_res,
            yRes=abs(y_res),
            dstSRS=ortho_geo_projection,
            resampleAlg='cubic',
            creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'],
        )

        os.remove(dtm)

    def _normalize_pair(self, ortho_tile_arr, dtm_tile_arr):
        """Normaliza um par de blocos (tiles) já lidos do disco."""

        ortho_tile = ortho_tile_arr.astype(np.float32)
        dtm_tile = dtm_tile_arr.astype(np.float32)

        min_o, max_o = ortho_tile.min(), ortho_tile.max()
        ortho_normalized = (ortho_tile - min_o) / (max_o - min_o + 1e-8)

        min_d, max_d = dtm_tile.min(), dtm_tile.max()
        dtm_normormalized = (dtm_tile - min_d) / (max_d - min_d + 1e-8)

        return ortho_normalized, dtm_normormalized

    def _prepare_assignments(self) -> None:
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
            str(out_path), 
            normalized.shape[1],
            normalized.shape[0],
            1,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'BIGTIFF=YES'],
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
        ortho_dataset = gdal.Open(str(ortho))
        dtm_dataset = gdal.Open(str(dtm))

        # Pega as infos dos *datasets*, não de arrays em memória
        base_geo_transform = ortho_dataset.GetGeoTransform()
        base_projection = ortho_dataset.GetProjection()
        nodata_val = dtm_dataset.GetRasterBand(1).GetNoDataValue()

        # Pega as dimensões dos datasets
        width = ortho_dataset.RasterXSize
        height = ortho_dataset.RasterYSize

        if nodata_val is None:
            nodata_val = -3.4028234663852886e+38 
        
        tile_count = 0
        
        # Pega as bandas (camadas) dos arquivos para leitura
        dtm_band = dtm_dataset.GetRasterBand(1)
        ortho_band = ortho_dataset.GetRasterBand(1)

        logger.info(f"Iniciando processamento de blocos para {ortho}...")

        for y in range(0, height - self._tile_size, self._stride):
            for x in range(0, width - self._tile_size, self._stride):
                dtm_tile_arr = dtm_band.ReadAsArray(
                    x, y, self._tile_size, self._tile_size
                )

                nodata_mask = (dtm_tile_arr == nodata_val) | np.isnan(dtm_tile_arr)
                if np.any(nodata_mask):
                    logger.info("Pulando bloco pois contém nodata")
                    continue

                ortho_tile_arr = ortho_band.ReadAsArray(
                    x, y, self._tile_size, self._tile_size
                )

                logger.info(f"Normalizando bloco {tile_count}")

                ortho_normalized, dtm_normormalized = self._normalize_pair(
                    ortho_tile_arr=ortho_tile_arr, 
                    dtm_tile_arr=dtm_tile_arr
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

                if tile_count > 100:
                    break

        ortho_dataset = None
        dtm_dataset = None
        
        logger.info(f"Processamento de blocos concluído. {tile_count} blocos gerados.")
        return tile_count

    def run(self) -> None:
        logger.info("Iniciando indexação de datasets")

        datasets = self._list_datasets()

        logger.info(f"Indexados {len(datasets)} datasets")

        self._prepare_assignments()

        count = 1

        for pair in datasets:
            dtm_candidate = pair.dtm_url
            ortho_candidate = pair.ortho_url

            if not dtm_candidate or not ortho_candidate:
                logger.info(f"Par incompleto pulado. DTM: {dtm_candidate}, ORTHO: {ortho_candidate}")
                continue

            download_sources = self._download_dir / "sources" / str(count) 

            if not self._assignments:
                logger.info("Lista de atribuições vazia. Usando 'train' como padrão.")
                destination = "train"
            else:
                destination = self._assignments.pop()

            tiles_dir = self._download_dir / destination

            download_sources.mkdir(exist_ok=True, parents=True)
            tiles_dir.mkdir(exist_ok=True, parents=True)

            source_dtm_path = download_sources / os.path.basename(dtm_candidate)
            source_ortho_path = download_sources / os.path.basename(ortho_candidate)
            source_dtm_aligned = download_sources / "DTM_aligned.tif"

            source_dtm_path.touch()
            source_ortho_path.touch()
            source_dtm_aligned.touch()

            try:
                logger.info(f"Fazendo download par [{count}/{self._samples}]")

                self._download_candidate(candidate=dtm_candidate, out_dir=source_dtm_path)
                self._download_candidate(candidate=ortho_candidate, out_dir=source_ortho_path)

                self._align_dtm_to_ortho(dtm=source_dtm_path, ortho=source_ortho_path, aligned=source_dtm_aligned)

                self._process_tiles(dtm=source_dtm_aligned, ortho=source_ortho_path, out_dir=tiles_dir, count=count)

            except Exception as e:
                logger.error(f"Falha ao processar o par {count}: {e}", exc_info=True)
                if tiles_dir.exists():
                    shutil.rmtree(tiles_dir)
                raise e
            finally:
                if download_sources.exists():
                    shutil.rmtree(download_sources)
                count += 1
