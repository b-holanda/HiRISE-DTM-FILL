import gc
import os
import random
import shutil
import tempfile
import time
import uuid
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

# --- Funções Auxiliares Globais ---

def _safe_remove_file(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass

def _configure_gdal_cache(cache_max_mb: int) -> None:
    cache_max_mb = max(cache_max_mb, 32)
    try:
        if hasattr(gdal, "SetConfigOption"):
            gdal.SetConfigOption("GDAL_CACHEMAX", str(cache_max_mb))
        if hasattr(gdal, "SetCacheMax"):
            gdal.SetCacheMax(cache_max_mb * 1024 * 1024)
    except Exception:
        pass

def _available_memory_mb() -> float:
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

def _save_tile_as_tif(array: np.ndarray, path: Path) -> None:
    """Salva um array numpy como GeoTIFF Float32 simples."""
    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    # Cria o dataset: x, y, bandas, tipo
    out_ds = driver.Create(str(path), cols, rows, 1, gdal.GDT_Float32)
    if out_ds is None:
        raise IOError(f"Falha ao criar arquivo TIF: {path}")
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()
    
    # Limpeza explícita para fechar o arquivo
    out_band = None
    out_ds = None

# --- Classes Principais ---

class DatasetBuilder:
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
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.gdal_cache_max_mb = gdal_cache_max_mb
        self.assignments = []

        if download_directory:
            self.download_directory = Path(download_directory)
            self.download_directory.mkdir(parents=True, exist_ok=True)
        else:
            self.download_directory = Path(tempfile.mkdtemp(prefix="marsfill_dataset_"))

        # Diretório temporário para troca de arquivos entre Produtor e Consumidor
        self.temp_exchange_dir = self.download_directory / "temp_exchange"
        self.temp_exchange_dir.mkdir(parents=True, exist_ok=True)

        _configure_gdal_cache(self.gdal_cache_max_mb)
        logger.info(f"Workspace: '{self.download_directory}'")

    def _list_datasets(self, max_pairs: Optional[int] = None) -> List[ProductPair]:
        indexer = HirisePDSIndexerDFS(self.urls_to_scan)
        return indexer.index_pairs(max_pairs=max_pairs or self.total_samples)

    def _prepare_assignments(self) -> None:
        logger.info(f"Preparando {self.total_samples} atribuições (80/20)...")
        validation_count = int(np.round(0.2 * self.total_samples))
        train_count = self.total_samples - validation_count
        self.assignments = ["train"] * train_count + ["validation"] * validation_count
        random.shuffle(self.assignments)

    # ---------------------------------------------------------
    # PRODUTOR: Baixa, Alinha, Corta e Salva TIFs Temporários
    # ---------------------------------------------------------
    @staticmethod
    def worker_producer(
        pair_data: Dict[str, object],
        tile_size: int,
        stride_size: int,
        download_directory: Path,
        temp_exchange_dir: Path,
        queue: multiprocessing.Queue, # Fila para enviar metadados
        gdal_cache_max_mb: int
    ) -> str:
        """
        Gera tiles e os salva como arquivos .tif individuais na pasta temporária.
        Envia o caminho desses arquivos para a Fila.
        """
        _configure_gdal_cache(gdal_cache_max_mb)
        
        dataset_split = str(pair_data["split"])
        pair_identifier = os.path.basename(str(pair_data["ortho_url"])).replace(".JP2", "")
        
        # Diretório de trabalho local do worker para download/warp
        worker_id = os.getpid()
        work_directory = download_directory / f"proc_{worker_id}_{uuid.uuid4().hex[:6]}"
        work_directory.mkdir(parents=True, exist_ok=True)

        local_ortho = work_directory / "ortho.tif"
        local_dtm = work_directory / "dtm.img"
        local_aligned = work_directory / "aligned.tif"

        tiles_produced_count = 0

        try:
            # 1. Download (Simplificado)
            def download(url: str) -> bytes:
                with requests.get(url, stream=True, timeout=(15, 300)) as r:
                    r.raise_for_status()
                    return r.content
            
            local_ortho.write_bytes(download(str(pair_data["ortho_url"])))
            local_dtm.write_bytes(download(str(pair_data["dtm_url"])))

            # 2. Warp/Alinhamento
            ds_ortho = gdal.Open(str(local_ortho))
            gt = ds_ortho.GetGeoTransform()
            w, h = ds_ortho.RasterXSize, ds_ortho.RasterYSize
            
            # Limites para warp
            min_x = gt[0]
            max_x = min_x + w * gt[1]
            max_y = gt[3]
            min_y = max_y + h * gt[5]

            gdal.Warp(
                str(local_aligned), str(local_dtm), format="GTiff",
                outputBounds=[min_x, min_y, max_x, max_y],
                xRes=gt[1], yRes=abs(gt[5]),
                dstSRS=ds_ortho.GetProjection(),
                resampleAlg="cubic",
                creationOptions=["COMPRESS=LZW", "TILED=YES"]
            )
            
            ds_aligned = gdal.Open(str(local_aligned))
            band_dtm = ds_aligned.GetRasterBand(1)
            band_ortho = ds_ortho.GetRasterBand(1)
            
            nodata = band_dtm.GetNoDataValue()
            if nodata is None: nodata = -3.4028234663852886e38

            # 3. Loop de Corte e Salvamento
            for y in range(0, h - tile_size, stride_size):
                for x in range(0, w - tile_size, stride_size):
                    dtm_arr = band_dtm.ReadAsArray(x, y, tile_size, tile_size)
                    
                    if np.any((dtm_arr == nodata) | np.isnan(dtm_arr)):
                        continue

                    ortho_arr = band_ortho.ReadAsArray(x, y, tile_size, tile_size)
                    
                    # Normalização
                    ortho_arr = ortho_arr.astype(np.float32)
                    dtm_arr = dtm_arr.astype(np.float32)
                    
                    o_min, o_max = ortho_arr.min(), ortho_arr.max()
                    ortho_norm = (ortho_arr - o_min) / (o_max - o_min + 1e-8)
                    
                    d_min, d_max = dtm_arr.min(), dtm_arr.max()
                    dtm_norm = (dtm_arr - d_min) / (d_max - d_min + 1e-8)

                    # 4. Salvar TIFs Temporários para o Consumidor
                    # Cria um ID único para este tile
                    tile_uuid = uuid.uuid4().hex
                    tile_dir = temp_exchange_dir / dataset_split / tile_uuid
                    tile_dir.mkdir(parents=True, exist_ok=True)

                    path_ortho_tif = tile_dir / "ortho.tif"
                    path_dtm_tif = tile_dir / "dtm.tif"

                    _save_tile_as_tif(ortho_norm, path_ortho_tif)
                    _save_tile_as_tif(dtm_norm, path_dtm_tif)

                    # 5. Enviar para a Fila
                    # O consumidor precisa saber onde estão os arquivos e metadados
                    tile_metadata = {
                        "split": dataset_split,
                        "pair_id": pair_identifier,
                        "tile_x": x,
                        "tile_y": y,
                        "ortho_path": str(path_ortho_tif),
                        "dtm_path": str(path_dtm_tif),
                        "temp_dir": str(tile_dir) # Para deletar depois
                    }
                    
                    queue.put(tile_metadata)
                    tiles_produced_count += 1

            ds_ortho = None
            ds_aligned = None

        except Exception as e:
            logger.error(f"Erro Produtor {pair_identifier}: {e}")
        finally:
            shutil.rmtree(work_directory, ignore_errors=True)
            
        return f"{pair_identifier}: {tiles_produced_count} tiles"

    # ---------------------------------------------------------
    # CONSUMIDOR: Lê Fila, Agrupa, Grava Parquet e Limpa
    # ---------------------------------------------------------
    @staticmethod
    def worker_consumer(
        queue: multiprocessing.Queue,
        download_directory: Path,
        batch_size: int,
        consumer_id: int
    ) -> None:
        """
        Consome itens da fila. Quando um buffer (train ou validation) atinge batch_size,
        lê os arquivos do disco, cria o parquet e deleta os temporários.
        """
        logger.info(f"🔧 Consumidor {consumer_id} iniciado.")
        
        # Buffers separados para não misturar train/validation no mesmo arquivo
        buffers: Dict[str, List[Dict]] = {"train": [], "validation": []}
        file_counters = {"train": 0, "validation": 0}

        def flush_buffer(split: str):
            if not buffers[split]:
                return

            items_to_process = buffers[split]
            data_rows = []
            dirs_to_delete = []

            # Lê os arquivos do disco e transforma em bytes para o parquet
            for item in items_to_process:
                try:
                    # Lendo o arquivo TIF como bytes brutos
                    # O arquivo é um GeoTIFF válido, então lemos os bytes do arquivo
                    # para salvar no Parquet (conforme lógica original de 'bytes')
                    with open(item["ortho_path"], "rb") as f:
                        ortho_bytes = f.read()
                    with open(item["dtm_path"], "rb") as f:
                        dtm_bytes = f.read()

                    data_rows.append({
                        "pair_id": item["pair_id"],
                        "tile_x": item["tile_x"],
                        "tile_y": item["tile_y"],
                        "ortho_bytes": ortho_bytes, 
                        "dtm_bytes": dtm_bytes
                    })
                    dirs_to_delete.append(item["temp_dir"])
                except Exception as e:
                    logger.error(f"Consumidor erro leitura {item['pair_id']}: {e}")

            if data_rows:
                df = pd.DataFrame(data_rows)
                table = pa.Table.from_pandas(df)
                
                # Gera nome único para evitar colisão entre consumidores
                filename = f"batch_c{consumer_id}_{file_counters[split]:05d}.parquet"
                out_dir = download_directory / split
                out_dir.mkdir(parents=True, exist_ok=True)
                
                pq.write_table(table, out_dir / filename, compression="snappy")
                file_counters[split] += 1
                
                logger.info(f"💾 Consumidor {consumer_id} gravou {filename} ({len(data_rows)} tiles)")

            # Limpeza dos arquivos temporários
            for d in dirs_to_delete:
                _safe_remove_file(Path(d) / "ortho.tif")
                _safe_remove_file(Path(d) / "dtm.tif")
                try:
                    Path(d).rmdir()
                except:
                    pass
            
            buffers[split] = [] # Limpa buffer da memória
            gc.collect()

        while True:
            try:
                item = queue.get()
                
                # Poison Pill: Sinal de parada
                if item is None:
                    logger.info(f"🛑 Consumidor {consumer_id} recebeu sinal de parada.")
                    # Flusha o que sobrou
                    flush_buffer("train")
                    flush_buffer("validation")
                    break

                split = item["split"]
                buffers[split].append(item)

                if len(buffers[split]) >= batch_size:
                    flush_buffer(split)

            except Exception as e:
                logger.error(f"Erro fatal Consumidor {consumer_id}: {e}")
                break

    # ---------------------------------------------------------
    # ORQUESTRAÇÃO
    # ---------------------------------------------------------
    def run(self) -> None:
        logger.info("Iniciando Pipeline Produtor-Consumidor...")

        # 1. Definição de Recursos (Regra 80/20)
        mem_avail_gb = _available_memory_mb() / 1024.0
        safe_max_workers = max(1, int(mem_avail_gb // 4)) # Mais conservador pois temos consumidores + produtores
        
        total_workers_limit = self.max_workers if self.max_workers else (os.cpu_count() or 1)
        total_workers = min(safe_max_workers, total_workers_limit)

        if total_workers < 2:
            logger.warning("Poucos workers disponíveis. Forçando 1 Produtor e 1 Consumidor.")
            n_consumers = 1
            n_producers = 1
        else:
            # Regra: Sempre 20% escrita (mínimo 1)
            n_consumers = max(1, int(total_workers * 0.2))
            n_producers = total_workers - n_consumers

        logger.info(f"Arquitetura: {total_workers} Total | {n_producers} Produtores | {n_consumers} Consumidores")

        # 2. Preparação dos Dados
        replacement_buffer = max(10, int(0.2 * self.total_samples))
        datasets = self._list_datasets(max_pairs=self.total_samples + replacement_buffer)
        self._prepare_assignments()

        tasks = []
        # Prepara a lista de tarefas para os produtores
        primary_datasets = datasets[: self.total_samples]
        for pair in primary_datasets:
            destination_split = self.assignments.pop() if self.assignments else "train"
            tasks.append({
                "dtm_url": pair.dtm_url,
                "ortho_url": pair.ortho_url,
                "split": destination_split,
            })

        # 3. Inicialização do Manager e Fila
        with multiprocessing.Manager() as manager:
            queue = manager.Queue() # Fila ilimitada (cuidado com RAM se produtor for muito mais rápido)
            
            # Limite a fila se quiser evitar que produtores encham a RAM com metadados
            # queue = manager.Queue(maxsize=10000) 

            # 4. Iniciando Consumidores (Processos Independentes)
            consumer_processes = []
            for i in range(n_consumers):
                p = multiprocessing.Process(
                    target=self.worker_consumer,
                    args=(queue, self.download_directory, self.batch_size, i)
                )
                p.start()
                consumer_processes.append(p)

            # 5. Iniciando Produtores (Via Pool)
            logger.info("Iniciando Produtores...")
            with ProcessPoolExecutor(max_workers=n_producers) as executor:
                futures = {
                    executor.submit(
                        self.worker_producer,
                        task,
                        self.tile_size,
                        self.stride_size,
                        self.download_directory,
                        self.temp_exchange_dir,
                        queue,
                        self.gdal_cache_max_mb
                    ): task for task in tasks
                }

                # Monitora produtores
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        # logger.debug(f"Produtor finalizou: {res}")
                    except Exception as e:
                        logger.error(f"Produtor falhou: {e}")

            logger.info("Todos os produtores finalizaram. Enviando sinal de parada para consumidores...")

            # 6. Encerramento (Poison Pill)
            # Envia um None para cada consumidor
            for _ in range(n_consumers):
                queue.put(None)

            # Aguarda consumidores terminarem de gravar
            for p in consumer_processes:
                p.join()

            # Limpeza Final do diretório de troca (deve estar vazio, mas por garantia)
            shutil.rmtree(self.temp_exchange_dir, ignore_errors=True)
            
        logger.info("Pipeline Finalizado com Sucesso!")
        