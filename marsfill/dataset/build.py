import os
import requests
import random
import boto3
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, Any
import zipfile
import shutil
import tempfile

from marsfill.utils import Logger
from marsfill.dataset.hirise_indexer import ProductPair, HirisePDSIndexerDFS

gdal.UseExceptions()
logger = Logger()


class DatasetBuilder:
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
    ) -> None:
        """
        Inicializa o construtor do dataset.

        Argumentos:
            urls_to_scan (List[str]): Lista de URLs base para escanear os produtos HiRISE.
            total_samples (int): Número total de pares de produtos para processar.
            tile_size (int): Tamanho da janela (largura e altura) para o recorte das imagens.
            stride_size (int): Passo do deslocamento da janela deslizante.
            download_directory (Optional[Path]): Caminho local para salvar os dados. Se None, usa S3.
            s3_bucket_name (Optional[str]): Nome do bucket S3 para salvar os dados. Obrigatório se download_directory for None.
            s3_prefix (str): Prefixo (pasta) dentro do bucket ou diretório local.
            batch_size (int): Quantidade de tiles para acumular antes de salvar um arquivo parquet.
            max_workers (Optional[int]): Número máximo de processos paralelos.
            s3_client (Optional[Any]): Instância de cliente boto3 S3 injetada para facilitar testes.
        """
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

        if self.download_directory is None:
            if not self.s3_bucket_name:
                raise ValueError("Para salvar no S3, forneça o 's3_bucket_name'.")

            self.s3_client = s3_client if s3_client else boto3.client("s3")
            logger.info(f"Modo Cloud: Bucket='{self.s3_bucket_name}', Prefix='{self.s3_prefix}'")
        else:
            self.s3_client = None
            logger.info(f"Modo Local: '{self.download_directory}'")

    def _list_datasets(self) -> List[ProductPair]:
        """
        Escaneia as URLs fornecidas e retorna uma lista de pares de produtos (Ortho e DTM).

        Retorna:
            List[ProductPair]: Lista contendo objetos com as URLs dos pares encontrados.
        """
        indexer = HirisePDSIndexerDFS(self.urls_to_scan)
        return indexer.index_pairs(max_pairs=self.total_samples)

    def _prepare_assignments(self) -> None:
        """
        Define aleatoriamente quais amostras pertencerão aos conjuntos de treino, teste e validação.
        A distribuição é fixa em 80% treino, 10% teste e 10% validação.
        """
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
        """
        Converte um índice em label alfabético (a, b, c, ..., z, aa, ab, ...)
        """
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
        download_directory: Optional[Path],
        s3_bucket_name: Optional[str],
        s3_prefix: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Processa um par de imagens (Orthoimagem e DTM) em um processo separado.
        Realiza o download em memória, alinhamento (warp), recorte (tiling) e normalização.

        Argumentos:
            pair_data (Dict[str, Any]): Dicionário contendo 'dtm_url', 'ortho_url' e 'split'.
            tile_size (int): Tamanho do recorte quadrado.
            stride_size (int): Tamanho do passo para recorte.

        Retorna:
            Tuple[str, List[Dict[str, Any]]]: Uma tupla contendo o nome do split (ex: 'train') e uma lista de dicionários com os tiles processados.
        """
        digital_terrain_model_url = pair_data["dtm_url"]
        ortho_image_url = pair_data["ortho_url"]
        dataset_split = pair_data["split"]
        test_label = pair_data.get("test_label")
        pair_identifier = os.path.basename(ortho_image_url).replace(".JP2", "")

        virtual_ortho_path = f"/vsimem/{pair_identifier}_ortho.tif"
        virtual_dtm_path = f"/vsimem/{pair_identifier}_dtm.img"
        virtual_aligned_path = f"/vsimem/{pair_identifier}_aligned.tif"

        processed_results = []

        try:

            def download_content_as_bytes(url: str) -> bytes:
                with requests.get(url, stream=True, timeout=(15, 300)) as response:
                    response.raise_for_status()
                    return response.content

            ortho_bytes = download_content_as_bytes(ortho_image_url)
            dtm_bytes = download_content_as_bytes(digital_terrain_model_url)

            # Salva cópias integrais para o conjunto de teste
            if dataset_split == "test" and test_label:
                if download_directory:
                    raw_dir = (
                        download_directory
                        / Path(s3_prefix.strip("/"))
                        / "test"
                        / f"test-{test_label}"
                    )
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    (raw_dir / "dtm.IMG").write_bytes(dtm_bytes)
                    (raw_dir / "ortho.JP2").write_bytes(ortho_bytes)
                    logger.info(f"💾 [TEST] Copiado original para {raw_dir}")
                else:
                    try:
                        client = boto3.client("s3")
                        base_key = f"{s3_prefix}test/test-{test_label}"
                        client.put_object(
                            Bucket=s3_bucket_name, Key=f"{base_key}/dtm.IMG", Body=dtm_bytes
                        )
                        client.put_object(
                            Bucket=s3_bucket_name, Key=f"{base_key}/ortho.JP2", Body=ortho_bytes
                        )
                        logger.info(f"☁️ [TEST] Upload original em s3://{s3_bucket_name}/{base_key}")
                    except Exception as s3_error:
                        logger.error(f"Falha ao enviar originais de teste: {s3_error}")

            gdal.FileFromMemBuffer(virtual_ortho_path, ortho_bytes)
            gdal.FileFromMemBuffer(virtual_dtm_path, dtm_bytes)

            ortho_dataset = gdal.Open(virtual_ortho_path)
            geo_transform = ortho_dataset.GetGeoTransform()
            projection_ref = ortho_dataset.GetProjection()
            raster_width = ortho_dataset.RasterXSize
            raster_height = ortho_dataset.RasterYSize

            x_min = geo_transform[0]
            x_max = x_min + raster_width * geo_transform[1]
            y_max = geo_transform[3]
            y_min = y_max + raster_height * geo_transform[5]

            gdal.Warp(
                virtual_aligned_path,
                virtual_dtm_path,
                format="GTiff",
                outputBounds=[x_min, y_min, x_max, y_max],
                xRes=geo_transform[1],
                yRes=abs(geo_transform[5]),
                dstSRS=projection_ref,
                resampleAlg="cubic",
                creationOptions=["COMPRESS=LZW"],
            )

            aligned_dataset = gdal.Open(virtual_aligned_path)
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

                    mask = (dtm_tile == nodata_value) | np.isnan(dtm_tile)
                    if np.any(mask):
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

                    processed_results.append(
                        {
                            "pair_id": pair_identifier,
                            "tile_x": x_coordinate,
                            "tile_y": y_coordinate,
                            "ortho_bytes": ortho_normalized.tobytes(),
                            "dtm_bytes": dtm_normalized.tobytes(),
                        }
                    )

            ortho_dataset = None
            aligned_dataset = None

        except Exception as error:
            print(f"Erro worker {pair_identifier}: {error}")

        finally:
            gdal.Unlink(virtual_ortho_path)
            gdal.Unlink(virtual_dtm_path)
            gdal.Unlink(virtual_aligned_path)

        return dataset_split, processed_results

    def _save_batch(self, data_list: List[Dict], dataset_split: str, batch_index: int) -> None:
        """
        Salva um lote de dados processados em arquivo Parquet (Local ou S3).

        Argumentos:
            data_list (List[Dict]): Lista de tiles processados.
            dataset_split (str): O conjunto de dados (train, test, validation).
            batch_index (int): O índice sequencial deste lote.
        """
        if not data_list:
            return

        dataframe = pd.DataFrame(data_list)
        pyarrow_table = pa.Table.from_pandas(dataframe)
        file_name = f"data_part_{batch_index:05d}.parquet"

        if self.download_directory:
            try:
                full_prefix = Path(self.s3_prefix.strip("/")) / dataset_split
                local_subdirectory = self.download_directory / full_prefix
                local_subdirectory.mkdir(parents=True, exist_ok=True)

                output_path = local_subdirectory / file_name
                pq.write_table(pyarrow_table, output_path, compression="snappy")
                logger.info(
                    f"💾 [{dataset_split.upper()}] Salvo: {output_path} ({len(dataframe)} tiles)"
                )
            except Exception as error:
                logger.error(f"Falha ao salvar no disco local: {error}")

        else:
            try:
                output_buffer = io.BytesIO()
                pq.write_table(pyarrow_table, output_buffer, compression="snappy")
                output_buffer.seek(0)

                s3_key = f"{self.s3_prefix}{dataset_split}/{file_name}"

                self.s3_client.put_object(
                    Bucket=self.s3_bucket_name, Key=s3_key, Body=output_buffer
                )
                logger.info(
                    f"☁️ [{dataset_split.upper()}] S3 Upload: {s3_key} ({len(dataframe)} tiles)"
                )
            except Exception as error:
                logger.error(f"Falha upload S3: {error}")

    def _package_assets(self) -> None:
        """
        Compacta os datasets gerados (train, test, validation) em arquivos .zip.
        Se estiver rodando em modo S3, baixa os parquets, zipa e faz upload do zip.
        Se local, apenas cria o zip no diretório.
        """
        logger.info("📦 Iniciando empacotamento de assets (Zip)...")
        dataset_splits = ["train", "validation", "test"]

        if self.download_directory:
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

        else:
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

                    with zipfile.ZipFile(
                        local_zip_path, "w", zipfile.ZIP_STORED
                    ) as zip_file_handle:
                        for file_key in found_files_keys:
                            file_name = os.path.basename(file_key)

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
                    logger.info(
                        f"   Enviando {split_name}.zip para o S3 ({file_size_mb:.2f} MB)..."
                    )

                    try:
                        with open(local_zip_path, "rb") as file_pointer:
                            self.s3_client.put_object(
                                Bucket=self.s3_bucket_name,
                                Key=s3_destination_key,
                                Body=file_pointer,
                            )
                        logger.info(
                            f"✅ Upload concluído: s3://{self.s3_bucket_name}/{s3_destination_key}"
                        )
                    except Exception as error:
                        logger.error(f"Erro no upload do zip: {error}")

    def run(self) -> None:
        """
        Executa o pipeline completo:
        1. Lista datasets.
        2. Atribui splits (treino/teste).
        3. Processa imagens em paralelo.
        4. Salva lotes em parquet.
        5. Empacota resultados em ZIP.
        """
        logger.info("Iniciando pipeline...")
        datasets = self._list_datasets()
        logger.info(f"Indexados {len(datasets)} pares")

        self._prepare_assignments()

        tasks = []
        test_counter = 0
        for pair in datasets:
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

        data_buffers = {"train": [], "validation": [], "test": []}

        batch_counters = {"train": 0, "validation": 0, "test": 0}

        active_workers = self.max_workers if self.max_workers else (os.cpu_count() or 1)
        logger.info(f"Workers: {active_workers} | Batch Size: {self.batch_size}")

        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            futures = [
                executor.submit(
                    self.worker_process_pair,
                    task,
                    self.tile_size,
                    self.stride_size,
                    self.download_directory,
                    self.s3_bucket_name,
                    self.s3_prefix,
                )
                for task in tasks
            ]

            for future in as_completed(futures):
                try:
                    split_name, tiles_list = future.result()

                    if tiles_list:
                        data_buffers[split_name].extend(tiles_list)
                        if len(data_buffers[split_name]) >= self.batch_size:
                            self._save_batch(
                                data_list=data_buffers[split_name],
                                dataset_split=split_name,
                                batch_index=batch_counters[split_name],
                            )
                            data_buffers[split_name] = []
                            batch_counters[split_name] += 1

                except Exception as error:
                    logger.error(f"Erro no loop principal: {error}")

        for split_name, buffer_data in data_buffers.items():
            if buffer_data:
                self._save_batch(
                    data_list=buffer_data,
                    dataset_split=split_name,
                    batch_index=batch_counters[split_name],
                )

        self._package_assets()

        logger.info("Processamento finalizado!")
