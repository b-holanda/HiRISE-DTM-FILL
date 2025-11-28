import os
import requests
from requests.exceptions import RequestException
import random
import boto3
from boto3.s3.transfer import TransferConfig
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
import gc
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Optional, Tuple, List, Dict, Any
import zipfile
import shutil
import tempfile
import time

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
        s3_bucket_name: Optional[str] = None,
        s3_prefix: str = "dataset/v1/",
        batch_size: int = 500,
        max_workers: Optional[int] = None,
        s3_client: Optional[Any] = None,
        gdal_cache_max_mb: int = 256, # OTIMIZAÇÃO: Valor padrão reduzido para ser conservador
    ) -> None:
        self.urls_to_scan = urls_to_scan
        self.total_samples = total_samples
        self.tile_size = tile_size
        self.stride_size = stride_size

        self.download_directory = download_directory
        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix if s3_prefix.endswith("/") else f"{s3_prefix}/"
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.assignments = []
        self.gdal_cache_max_mb = gdal_cache_max_mb

        if download_directory:
            self.download_directory = Path(download_directory)
            self.download_directory.mkdir(parents=True, exist_ok=True)
            self._temporary_download_dir = False
        else:
            self.download_directory = Path(tempfile.mkdtemp(prefix="marsfill_dataset_"))
            self._temporary_download_dir = True

        _configure_gdal_cache(self.gdal_cache_max_mb)

        self.is_s3_mode = bool(s3_bucket_name)

        if s3_client and not s3_bucket_name:
            raise ValueError("Para salvar no S3, forneça o 's3_bucket_name'.")

        if s3_bucket_name:
            self.s3_client = s3_client if s3_client else boto3.client("s3")
            logger.info(
                f"Modo Cloud: Bucket='{self.s3_bucket_name}', Prefix='{self.s3_prefix}', "
                f"cache GDAL={self.gdal_cache_max_mb}MB, workspace='{self.download_directory}'"
            )
        else:
            self.s3_client = None
            logger.info(
                f"Modo Local: '{self.download_directory}' | cache GDAL={self.gdal_cache_max_mb}MB"
            )

    def _list_datasets(self, max_pairs: Optional[int] = None) -> List[ProductPair]:
        indexer = HirisePDSIndexerDFS(self.urls_to_scan)
        return indexer.index_pairs(max_pairs=max_pairs or self.total_samples)

    def _prepare_assignments(self) -> None:
        logger.info(f"Preparando {self.total_samples} atribuições...")
        test_count = int(np.round(0.1 * self.total_samples))
        validation_count = int(np.round(0.1 * self.total_samples))
        train_count = self.total_samples - test_count - validation_count

        self.assignments = (
            ["train"] * train_count + ["test"] * test_count + ["validation"] * validation_count
        )
        random.shuffle(self.assignments)

    @staticmethod
    def _index_to_label(index: int) -> str:
        letters = []
        while True:
            index, remainder = divmod(index, 26)
            letters.append(chr(ord("a") + remainder))
            if index == 0:
                break
            index -= 1
        return "".join(reversed(letters))

    @staticmethod
    def worker_process_pair(
        pair_data: Dict[str, Any],
        tile_size: int,
        stride_size: int,
        download_directory: Path,
        s3_bucket_name: Optional[str],
        s3_prefix: str,
        gdal_cache_max_mb: int = 256,
        batch_size: int = 500
    ) -> Tuple[str, List[Path]]: # OTIMIZAÇÃO: Retorna lista de caminhos de arquivo, não os dados
        """
        Processa um par DTM/ORTHO e salva arquivos parquet parciais localmente.
        Retorna o split e a lista de arquivos gerados.
        """
        digital_terrain_model_url = pair_data["dtm_url"]
        ortho_image_url = pair_data["ortho_url"]
        dataset_split = pair_data["split"]
        test_label = pair_data.get("test_label")
        pair_identifier = os.path.basename(ortho_image_url).replace(".JP2", "")

        _configure_gdal_cache(gdal_cache_max_mb)

        # OTIMIZAÇÃO: Diretório único por worker/processo para evitar colisão de I/O
        worker_id = os.getpid()
        work_directory = Path(download_directory) / f"worker_{worker_id}" / pair_identifier
        work_directory.mkdir(parents=True, exist_ok=True)

        local_ortho_path = work_directory / f"{pair_identifier}_ortho.tif"
        local_dtm_path = work_directory / f"{pair_identifier}_dtm.img"
        local_aligned_path = work_directory / f"{pair_identifier}_aligned.tif"

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

            # Lógica de Teste (Cópia dos originais)
            if dataset_split == "test" and test_label:
                # Nota: Salva em pasta temporária do worker, o Run move depois se necessário
                # Mas para simplificar a lógica de teste complexa, mantemos a lógica de upload direto aqui
                # ou simplificamos. Vou manter a lógica original de upload S3 direto para o teste para não quebrar.
                raw_dir = work_directory / "raw_test"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_dtm_path = raw_dir / "dtm.IMG"
                raw_ortho_path = raw_dir / "ortho.JP2"

                shutil.copyfile(local_dtm_path, raw_dtm_path)
                shutil.copyfile(local_ortho_path, raw_ortho_path)

                if s3_bucket_name:
                    try:
                        client = boto3.client("s3")
                        base_key = f"{s3_prefix}test/test-{test_label}"
                        client.upload_file(str(raw_dtm_path), s3_bucket_name, f"{base_key}/dtm.IMG")
                        client.upload_file(str(raw_ortho_path), s3_bucket_name, f"{base_key}/ortho.JP2")
                        logger.info(f"☁️ [TEST] Upload original em s3://{s3_bucket_name}/{base_key}")
                    except Exception as s3_error:
                        logger.error(f"Falha ao enviar originais de teste: {s3_error}")
                
                # Limpeza
                shutil.rmtree(raw_dir, ignore_errors=True)

            # --- PROCESSAMENTO GDAL ---
            ortho_dataset = gdal.Open(str(local_ortho_path))
            geo_transform = ortho_dataset.GetGeoTransform()
            projection_ref = ortho_dataset.GetProjection()
            raster_width = ortho_dataset.RasterXSize
            raster_height = ortho_dataset.RasterYSize

            x_min = geo_transform[0]
            x_max = x_min + raster_width * geo_transform[1]
            y_max = geo_transform[3]
            y_min = y_max + raster_height * geo_transform[5]

            # Warp
            gdal.Warp(
                str(local_aligned_path),
                str(local_dtm_path),
                format="GTiff",
                outputBounds=[x_min, y_min, x_max, y_max],
                xRes=geo_transform[1],
                yRes=abs(geo_transform[5]),
                dstSRS=projection_ref,
                resampleAlg="cubic",
                creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"], # OTIMIZAÇÃO: TILED ajuda na leitura por blocos
            )

            aligned_dataset = gdal.Open(str(local_aligned_path))
            dtm_band = aligned_dataset.GetRasterBand(1)
            ortho_band = ortho_dataset.GetRasterBand(1)

            nodata_value = dtm_band.GetNoDataValue()
            if nodata_value is None:
                nodata_value = -3.4028234663852886e38

            # --- LOOP OTIMIZADO DE CORTE ---
            for y_coordinate in range(0, raster_height - tile_size, stride_size):
                for x_coordinate in range(0, raster_width - tile_size, stride_size):
                    dtm_tile = dtm_band.ReadAsArray(
                        x_coordinate, y_coordinate, tile_size, tile_size
                    )

                    # Check rápido com numpy
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

                    # OTIMIZAÇÃO: Salva Parquet periodicamente para limpar memória do Worker
                    if len(buffer_tiles) >= batch_size:
                        filename = f"{pair_identifier}_part_{file_counter:04d}.parquet"
                        output_path = work_directory / filename
                        
                        df = pd.DataFrame(buffer_tiles)
                        table = pa.Table.from_pandas(df)
                        pq.write_table(table, output_path, compression="snappy")
                        
                        generated_files.append(output_path)
                        buffer_tiles = [] # Limpa buffer
                        file_counter += 1
                        gc.collect() # Força Garbage Collector

            # Salva o restante do buffer
            if buffer_tiles:
                filename = f"{pair_identifier}_part_{file_counter:04d}.parquet"
                output_path = work_directory / filename
                df = pd.DataFrame(buffer_tiles)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path, compression="snappy")
                generated_files.append(output_path)
                buffer_tiles = []

            # Cleanup GDAL Explícito
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

        return dataset_split, generated_files

    def _package_assets(self) -> None:
        """
        Compacta os datasets gerados (train, test, validation) em arquivos .zip.
        """
        logger.info("📦 Iniciando empacotamento de assets (Zip)...")
        dataset_splits = ["train", "validation", "test"]

        if not self.is_s3_mode or not self.s3_client:
            assets_directory = self.download_directory / self.s3_prefix.strip("/") / "assets"
            assets_directory.mkdir(parents=True, exist_ok=True)

            for split_name in dataset_splits:
                source_directory = self.download_directory / self.s3_prefix.strip("/") / split_name
                if not source_directory.exists():
                    continue

                output_zip_path = assets_directory / split_name
                logger.info(f"   Compactando {split_name} em {output_zip_path}.zip...")

                shutil.make_archive(
                    base_name=str(output_zip_path), format="zip", root_dir=source_directory
                )
                logger.info(f"✅ {split_name}.zip criado com sucesso (Local).")

            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            for split_name in dataset_splits:
                s3_source_prefix = f"{self.s3_prefix}{split_name}/"
                logger.info(f"   Processando {split_name} no S3...")

                paginator = self.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=self.s3_bucket_name, Prefix=s3_source_prefix)

                found_files_keys = []
                for page in pages:
                    for obj in page.get("Contents", []):
                        found_files_keys.append(obj["Key"])

                if not found_files_keys:
                    logger.info(f"   Nenhum arquivo encontrado para {split_name}, pulando.")
                    continue

                local_zip_path = temporary_path / f"{split_name}.zip"

                with zipfile.ZipFile(local_zip_path, "w", zipfile.ZIP_STORED) as zip_file_handle:
                    for file_key in found_files_keys:
                        file_name = os.path.basename(file_key)
                        if not file_name:
                            continue

                        try:
                            s3_object = self.s3_client.get_object(
                                Bucket=self.s3_bucket_name, Key=file_key
                            )
                            file_content = s3_object["Body"].read()
                            zip_file_handle.writestr(file_name, file_content)
                        except Exception as error:
                            logger.error(f"Erro ao baixar {file_key} para zip: {error}")

                s3_destination_key = f"{self.s3_prefix}assets/{split_name}.zip"
                file_size_mb = os.path.getsize(local_zip_path) / 1e6
                logger.info(f"   Enviando {split_name}.zip para o S3 ({file_size_mb:.2f} MB)...")

                try:
                    with open(local_zip_path, "rb") as file_pointer:
                        transfer_config = TransferConfig(
                            multipart_threshold=8 * 1024 * 1024,
                            multipart_chunksize=64 * 1024 * 1024,
                            max_concurrency=4,
                            use_threads=True,
                        )
                        self.s3_client.upload_fileobj(
                            file_pointer,
                            self.s3_bucket_name,
                            s3_destination_key,
                            Config=transfer_config,
                        )
                    logger.info(
                        f"✅ Upload concluído: s3://{self.s3_bucket_name}/{s3_destination_key}"
                    )
                except Exception as error:
                    logger.error(f"Erro no upload do zip: {error}")

        self._cleanup_local_artifacts()

    def _cleanup_local_artifacts(self) -> None:
        """Remove artefatos locais já enviados para o S3."""
        if not self.download_directory:
            return

        target_root = self.download_directory / self.s3_prefix.strip("/")
        if target_root.exists():
            try:
                shutil.rmtree(target_root, ignore_errors=True)
                logger.info(f"🧹 Artefatos locais removidos: {target_root}")
            except Exception as error:
                logger.warning(f"Não foi possível limpar artefatos locais em {target_root}: {error}")

        if getattr(self, "_temporary_download_dir", False):
            try:
                shutil.rmtree(self.download_directory, ignore_errors=True)
            except Exception as error:
                logger.debug(f"Falha ao remover diretório temporário {self.download_directory}: {error}")

    def run(self) -> None:
        """
        Executa o pipeline completo de construção do dataset (OTIMIZADO).
        """
        logger.info("Iniciando pipeline OTIMIZADO...")
        
        # OTIMIZAÇÃO: Cálculo de workers baseado em RAM disponível
        mem_avail_gb = _available_memory_mb() / 1024.0
        # Estimativa: 8GB por worker (GDAL Warp + buffers). Se for muito conservador, mude para 6GB.
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
        test_counter = 0
        for pair in primary_datasets:
            destination_split = self.assignments.pop() if self.assignments else "train"
            test_label = None
            if destination_split == "test":
                test_label = self._index_to_label(test_counter)
                test_counter += 1
            tasks.append(
                {
                    "dtm_url": pair.dtm_url,
                    "ortho_url": pair.ortho_url,
                    "split": destination_split,
                    "test_label": test_label,
                }
            )

        # OTIMIZAÇÃO: Não acumulamos mais dados em memória no processo pai.
        # Apenas despachamos tasks e gerenciamos arquivos resultantes.

        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.worker_process_pair,
                    task,
                    self.tile_size,
                    self.stride_size,
                    self.download_directory,
                    self.s3_bucket_name, # Passa nome do bucket, mas worker não faz upload dos parquets
                    self.s3_prefix,
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

                        # O Worker já salvou os parquets em uma pasta temporária dele.
                        # Agora movemos para o local final ou fazemos upload.
                        
                        for parquet_path in parquet_files:
                            file_name = parquet_path.name
                            
                            # Configura destino local final
                            full_prefix_path = Path(self.s3_prefix.strip("/")) / split_name
                            final_local_dir = self.download_directory / full_prefix_path
                            final_local_dir.mkdir(parents=True, exist_ok=True)
                            
                            final_dest_path = final_local_dir / file_name
                            
                            # Move do tmp do worker para pasta estruturada
                            shutil.move(str(parquet_path), str(final_dest_path))

                            # Upload S3 se necessário
                            if self.is_s3_mode and self.s3_client:
                                s3_key = f"{self.s3_prefix}{split_name}/{file_name}"
                                try:
                                    transfer_config = TransferConfig(
                                        multipart_threshold=8 * 1024 * 1024,
                                        multipart_chunksize=64 * 1024 * 1024,
                                        use_threads=True
                                    )
                                    self.s3_client.upload_file(
                                        str(final_dest_path), 
                                        self.s3_bucket_name, 
                                        s3_key,
                                        Config=transfer_config
                                    )
                                    logger.info(f"☁️ Upload S3: {file_name}")
                                    # Remove local para economizar espaço se estiver em modo nuvem total
                                    _safe_remove_file(final_dest_path)
                                except Exception as e:
                                    logger.error(f"Erro upload S3 {file_name}: {e}")
                            else:
                                logger.info(f"💾 Salvo Local: {file_name}")

                        # Limpa pasta temporária do worker (que ficou vazia após o move)
                        if parquet_files:
                            worker_dir = parquet_files[0].parent
                            shutil.rmtree(worker_dir, ignore_errors=True)

                    except RequestException as error:
                        logger.warning(
                            "Falha no worker (Download/Network) split=%s: %s",
                            task_info.get("split"), error
                        )
                        # Lógica de Retry (Substituição)
                        if replacement_queue:
                            new_pair = replacement_queue.pop(0)
                            new_task = {
                                "dtm_url": new_pair.dtm_url,
                                "ortho_url": new_pair.ortho_url,
                                "split": task_info.get("split"),
                                "test_label": task_info.get("test_label"),
                            }
                            new_future = executor.submit(
                                self.worker_process_pair,
                                new_task,
                                self.tile_size,
                                self.stride_size,
                                self.download_directory,
                                self.s3_bucket_name,
                                self.s3_prefix,
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

        self._package_assets()

        logger.info("Processamento finalizado!")
