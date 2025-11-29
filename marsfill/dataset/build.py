import gc
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from osgeo import gdal
from requests.exceptions import RequestException

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

from marsfill.utils import Logger
from marsfill.dataset.hirise_indexer import ProductPair, HirisePDSIndexerDFS

gdal.UseExceptions()
logger = Logger()


def _safe_remove_file(path: Path) -> None:
    """Remove um arquivo local, ignorando erros e ausências."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        logger.debug(f"Remoção ignorada para {path}")


def _configure_gdal_cache(cache_max_mb: int) -> None:
    """
    Ajusta o cache do GDAL para evitar uso excessivo de memória.
    """
    cache_max_mb = max(cache_max_mb, 32)
    try:
        if hasattr(gdal, "SetConfigOption"):
            gdal.SetConfigOption("GDAL_CACHEMAX", str(cache_max_mb))
        if hasattr(gdal, "SetCacheMax"):
            gdal.SetCacheMax(cache_max_mb * 1024 * 1024)
    except Exception as error:
        logger.debug(f"Não foi possível ajustar cache do GDAL: {error}")


def _available_memory_mb() -> float:
    """Retorna memória disponível aproximada em MB."""
    if psutil:
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            pass

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        return -1.0

    return -1.0


class DatasetBuilder:
    """Constrói o dataset Marsfill a partir de pares HiRISE (DTM e ORTHO)."""

    def __init__(
        self,
        urls_to_scan: List[str],
        total_samples: int,
        tile_size: int,
        stride_size: int,
        download_directory: Optional[Path] = None,
        batch_size: int = 500,
        max_workers: Optional[int] = None,
        gdal_cache_max_mb: int = 256,
    ) -> None:
        self.urls_to_scan = urls_to_scan
        self.total_samples = total_samples
        self.tile_size = tile_size
        self.stride_size = stride_size

        self.download_directory = download_directory
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.assignments = []
        self.gdal_cache_max_mb = gdal_cache_max_mb

        if download_directory:
            self.download_directory = Path(download_directory)
            self.download_directory.mkdir(parents=True, exist_ok=True)
        else:
            self.download_directory = Path(tempfile.mkdtemp(prefix="marsfill_dataset_"))

        _configure_gdal_cache(self.gdal_cache_max_mb)

        logger.info(
            f"Modo Local: '{self.download_directory}' | cache GDAL={self.gdal_cache_max_mb}MB"
        )

    def _list_datasets(self, max_pairs: Optional[int] = None) -> List[ProductPair]:
        indexer = HirisePDSIndexerDFS(self.urls_to_scan)
        return indexer.index_pairs(max_pairs=max_pairs or self.total_samples)

    def _prepare_assignments(self) -> None:
        """
        Distribui 80% para treino e 20% para validação.
        Sem conjunto de Teste.
        """
        logger.info(f"Preparando {self.total_samples} atribuições (80% Treino / 20% Validação)...")
        
        validation_count = int(np.round(0.2 * self.total_samples))
        train_count = self.total_samples - validation_count

        self.assignments = ["train"] * train_count + ["validation"] * validation_count
        random.shuffle(self.assignments)
        
        logger.info(f"Distribuição final: {train_count} Treino, {validation_count} Validação")

    @staticmethod
    def worker_process_pair(
        pair_data: Dict[str, object],
        tile_size: int,
        stride_size: int,
        download_directory: Path,
        gdal_cache_max_mb: int = 256,
        batch_size: int = 500
    ) -> Tuple[str, List[Path]]:
        """
        Processa um par DTM/ORTHO e salva arquivos parquet parciais localmente.
        """
        digital_terrain_model_url = pair_data["dtm_url"]
        ortho_image_url = pair_data["ortho_url"]
        dataset_split = pair_data["split"]
        pair_identifier = os.path.basename(ortho_image_url).replace(".JP2", "")

        _configure_gdal_cache(gdal_cache_max_mb)

        # Diretório temporário do worker
        worker_id = os.getpid()
        work_directory = Path(download_directory) / f"worker_{worker_id}" / pair_identifier
        work_directory.mkdir(parents=True, exist_ok=True)

        local_ortho_path = work_directory / f"{pair_identifier}_ortho.tif"
        local_dtm_path = work_directory / f"{pair_identifier}_dtm.img"
        local_aligned_path = work_directory / f"{pair_identifier}_aligned.tif"

        # Destino final
        final_output_dir = Path(download_directory) / dataset_split
        final_output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []
        buffer_tiles = []
        file_counter = 0

        try:
            def download_content_as_bytes(url: str) -> bytes:
                attempts = 5
                delay_seconds = 2.0
                for attempt in range(1, attempts + 1):
                    try:
                        with requests.get(url, stream=True, timeout=(15, 300)) as response:
                            response.raise_for_status()
                            return response.content
                    except RequestException as error:
                        if attempt == attempts:
                            raise
                        logger.warning(
                            "Falha ao baixar %s (tentativa %s/%s): %s",
                            url, attempt, attempts, error
                        )
                        time.sleep(delay_seconds)
                        delay_seconds *= 2

            ortho_bytes = download_content_as_bytes(ortho_image_url)
            dtm_bytes = download_content_as_bytes(digital_terrain_model_url)

            _safe_remove_file(local_ortho_path)
            _safe_remove_file(local_dtm_path)
            _safe_remove_file(local_aligned_path)

            local_ortho_path.write_bytes(ortho_bytes)
            local_dtm_path.write_bytes(dtm_bytes)

            ortho_dataset = gdal.Open(str(local_ortho_path))
            geo_transform = ortho_dataset.GetGeoTransform()
            projection_ref = ortho_dataset.GetProjection()
            raster_width = ortho_dataset.RasterXSize
            raster_height = ortho_dataset.RasterYSize

            x_min = geo_transform[0]
            x_max = x_min + raster_width * geo_transform[1]
            y_max = geo_transform[3]
            y_min = y_max + raster_height * geo_transform[5]

            gdal.Warp(
                str(local_aligned_path),
                str(local_dtm_path),
                format="GTiff",
                outputBounds=[x_min, y_min, x_max, y_max],
                xRes=geo_transform[1],
                yRes=abs(geo_transform[5]),
                dstSRS=projection_ref,
                resampleAlg="cubic",
                creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
            )

            aligned_dataset = gdal.Open(str(local_aligned_path))
            dtm_band = aligned_dataset.GetRasterBand(1)
            ortho_band = ortho_dataset.GetRasterBand(1)

            nodata_value = dtm_band.GetNoDataValue()
            if nodata_value is None:
                nodata_value = -3.4028234663852886e38

            for y_coordinate in range(0, raster_height - tile_size, stride_size):
                for x_coordinate in range(0, raster_width - tile_size, stride_size):
                    dtm_tile = dtm_band.ReadAsArray(
                        x_coordinate, y_coordinate, tile_size, tile_size
                    )

                    if np.any((dtm_tile == nodata_value) | np.isnan(dtm_tile)):
                        continue

                    ortho_tile = ortho_band.ReadAsArray(
                        x_coordinate, y_coordinate, tile_size, tile_size
                    )

                    ortho_tile = ortho_tile.astype(np.float32)
                    dtm_tile = dtm_tile.astype(np.float32)

                    min_ortho, max_ortho = ortho_tile.min(), ortho_tile.max()
                    ortho_normalized = (ortho_tile - min_ortho) / (max_ortho - min_ortho + 1e-8)

                    min_dtm, max_dtm = dtm_tile.min(), dtm_tile.max()
                    dtm_normalized = (dtm_tile - min_dtm) / (max_dtm - min_dtm + 1e-8)

                    buffer_tiles.append(
                        {
                            "pair_id": pair_identifier,
                            "tile_x": x_coordinate,
                            "tile_y": y_coordinate,
                            "ortho_bytes": ortho_normalized.astype(np.float16).tobytes(),
                            "dtm_bytes": dtm_normalized.astype(np.float16).tobytes(),
                        }
                    )

                    if len(buffer_tiles) >= batch_size:
                        filename = f"{pair_identifier}_part_{file_counter:04d}.parquet"
                        output_path = final_output_dir / filename
                        
                        df = pd.DataFrame(buffer_tiles)
                        table = pa.Table.from_pandas(df)
                        pq.write_table(table, output_path, compression="snappy")
                        
                        generated_files.append(output_path)
                        buffer_tiles = []
                        file_counter += 1
                        gc.collect()

            if buffer_tiles:
                filename = f"{pair_identifier}_part_{file_counter:04d}.parquet"
                output_path = final_output_dir / filename
                df = pd.DataFrame(buffer_tiles)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path, compression="snappy")
                generated_files.append(output_path)
                buffer_tiles = []

            ortho_dataset = None
            aligned_dataset = None
            dtm_band = None
            ortho_band = None

        except Exception as error:
            logger.exception(
                "Erro processando par %s (split=%s | mem disponível ~%.1f MB)", 
                pair_identifier, dataset_split, _available_memory_mb()
            )
            raise

        finally:
            _safe_remove_file(local_ortho_path)
            _safe_remove_file(local_dtm_path)
            _safe_remove_file(local_aligned_path)
            shutil.rmtree(work_directory, ignore_errors=True)

        return dataset_split, generated_files

    def run(self) -> None:
        """
        Executa o pipeline completo de construção do dataset.
        """
        logger.info("Iniciando pipeline OTIMIZADO...")
        
        mem_avail_gb = _available_memory_mb() / 1024.0
        safe_workers = max(1, int(mem_avail_gb // 8))
        
        user_workers = self.max_workers if self.max_workers else (os.cpu_count() or 1)
        active_workers = min(safe_workers, user_workers)
        
        logger.info(f"RAM Disp: {mem_avail_gb:.1f}GB. Workers calculados: {active_workers} (User Req: {user_workers})")

        replacement_buffer = max(10, int(0.2 * self.total_samples))
        datasets = self._list_datasets(max_pairs=self.total_samples + replacement_buffer)
        logger.info(f"Indexados {len(datasets)} pares")

        self._prepare_assignments()

        primary_datasets = datasets[: self.total_samples]
        replacement_queue = datasets[self.total_samples :]

        tasks = []
        for pair in primary_datasets:
            destination_split = self.assignments.pop() if self.assignments else "train"
            tasks.append(
                {
                    "dtm_url": pair.dtm_url,
                    "ortho_url": pair.ortho_url,
                    "split": destination_split,
                }
            )

        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.worker_process_pair,
                    task,
                    self.tile_size,
                    self.stride_size,
                    self.download_directory,
                    self.gdal_cache_max_mb,
                    self.batch_size
                ): task
                for task in tasks
            }

            futures_pending = set(future_to_task.keys())

            while futures_pending:
                for future in as_completed(list(futures_pending)):
                    futures_pending.discard(future)
                    task_info = future_to_task.pop(future)
                    
                    try:
                        split_name, parquet_files = future.result()

                        if parquet_files:
                            logger.info(
                                "💾 Worker finalizou %s com %d arquivos (ex: %s)",
                                split_name,
                                len(parquet_files),
                                parquet_files[0].name,
                            )
                        else:
                            logger.info("⚠️ Worker finalizou %s sem tiles válidos", split_name)

                    except RequestException as error:
                        logger.warning(
                            "Falha no worker (Download/Network) split=%s: %s",
                            task_info.get("split"), error
                        )
                        if replacement_queue:
                            new_pair = replacement_queue.pop(0)
                            new_task = {
                                "dtm_url": new_pair.dtm_url,
                                "ortho_url": new_pair.ortho_url,
                                "split": task_info.get("split"),
                            }
                            new_future = executor.submit(
                                self.worker_process_pair,
                                new_task,
                                self.tile_size,
                                self.stride_size,
                                self.download_directory,
                                self.gdal_cache_max_mb,
                                self.batch_size
                            )
                            future_to_task[new_future] = new_task
                            futures_pending.add(new_future)
                            logger.info(
                                "RETRY: Novo par submetido: %s", new_task["ortho_url"]
                            )

                    except BrokenProcessPool as error:
                        logger.critical("CRITICAL: Process Pool Quebrou (Possível OOM). Reinicie com menos workers.")
                        futures_pending.clear()
                        break
                    
                    except Exception as error:
                         logger.exception(
                            "Erro genérico na task split=%s: %s",
                            task_info.get("split"), error
                        )

        logger.info("Processamento finalizado!")
