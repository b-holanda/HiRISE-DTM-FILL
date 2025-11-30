import torch
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as parquet
import numpy as np
from typing import List, Tuple, Iterator, Optional
from transformers import DPTImageProcessor

class StreamingHiRISeDataset(IterableDataset):
    """
    Dataset iterável para carregar dados HiRISe de arquivos Parquet em streaming.
    Versão blindada contra erros de Buffer Size e dados corrompidos.
    """

    def __init__(
        self,
        parquet_file_paths: List[str],
        image_processor: DPTImageProcessor,
        process_rank: int = 0,
        total_process_count: int = 1,
        image_tile_size: int = 512,
        estimated_rows_per_file: int = 500,
    ) -> None:
        super().__init__()
        self.parquet_file_paths = sorted(parquet_file_paths)
        self.image_processor = image_processor
        self.process_rank = process_rank
        self.total_process_count = total_process_count
        self.image_tile_size = image_tile_size
        self.estimated_rows_per_file = estimated_rows_per_file

    def _determine_worker_files(self) -> List[str]:
        files_assigned_to_process = self.parquet_file_paths[
            self.process_rank :: self.total_process_count
        ]
        worker_information = get_worker_info()

        if worker_information is None:
            return files_assigned_to_process
        else:
            return files_assigned_to_process[
                worker_information.id :: worker_information.num_workers
            ]

    def _safe_decode_buffer(self, buffer: bytes, name: str) -> Optional[np.ndarray]:
        """
        Tenta decodificar o buffer validando estritamente o tamanho.
        Retorna None se o tamanho for incompatível, evitando o crash.
        """
        num_pixels = self.image_tile_size * self.image_tile_size
        size_f16 = num_pixels * 2
        size_f32 = num_pixels * 4
        buffer_len = len(buffer)

        if buffer_len == size_f16:
            return (
                np.frombuffer(buffer, dtype=np.float16)
                .reshape(self.image_tile_size, self.image_tile_size)
                .astype(np.float32)
            )
        elif buffer_len == size_f32:
            return (
                np.frombuffer(buffer, dtype=np.float32)
                .reshape(self.image_tile_size, self.image_tile_size)
                .astype(np.float32)
            )
        else:
            print(f"⚠️  DADO CORROMPIDO ({name}): Recebido {buffer_len} bytes. Esperado {size_f32} (f32) ou {size_f16} (f16).")
            return None

    def _convert_bytes_to_tensors(
        self, ortho_bytes: bytes, dtm_bytes: bytes
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retorna None se houver falha na conversão.
        """
        
        orthophoto_numpy_array = self._safe_decode_buffer(ortho_bytes, "Ortho")
        if orthophoto_numpy_array is None:
            return None

        digital_terrain_model_numpy_array = self._safe_decode_buffer(dtm_bytes, "DTM")
        if digital_terrain_model_numpy_array is None:
            return None
        
        digital_terrain_model_numpy_array = digital_terrain_model_numpy_array.copy()

        orthophoto_rgb_array = np.stack(
            [orthophoto_numpy_array, orthophoto_numpy_array, orthophoto_numpy_array], axis=-1
        )

        processed_inputs = self.image_processor(orthophoto_rgb_array, return_tensors="pt")
        pixel_values_tensor = processed_inputs["pixel_values"].squeeze(0)
        
        digital_terrain_model_tensor = (
            torch.from_numpy(digital_terrain_model_numpy_array).float().unsqueeze(0)
        )

        return pixel_values_tensor, digital_terrain_model_tensor

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        files_assigned_to_worker = self._determine_worker_files()

        for file_path in files_assigned_to_worker:
            try:
                parquet_file = parquet.ParquetFile(file_path)

                for group_index in range(parquet_file.num_row_groups):
                    row_group_batch = parquet_file.read_row_group(group_index)
                    dataframe = row_group_batch.to_pandas()

                    for _, data_row in dataframe.iterrows():
                        result = self._convert_bytes_to_tensors(
                            data_row["ortho_bytes"], data_row["dtm_bytes"]
                        )
                        
                        # SE O RESULTADO FOR NONE (CORROMPIDO), PULA SEM CRASHAR
                        if result is None:
                            continue
                            
                        yield result

            except Exception as error:
                # Captura erros de leitura do arquivo Parquet em si
                print(f"❌ Erro crítico ao ler arquivo {file_path}: {error}")
                continue

    def __len__(self) -> int:
        return len(self.parquet_file_paths) * self.estimated_rows_per_file
