import torch
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as parquet
import numpy as np
from typing import List, Tuple, Iterator
from transformers import DPTImageProcessor


class StreamingHiRISeDataset(IterableDataset):
    """
    Dataset iterável para carregar dados HiRISe de arquivos Parquet em streaming.
    Suporta processamento distribuído (DDP) e múltiplos workers de DataLoader.
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
        """
        Inicializa o dataset de streaming.

        Parâmetros:
            parquet_file_paths (List[str]): Lista de caminhos para os arquivos .parquet.
            image_processor (DPTImageProcessor): Processador de imagens do HuggingFace para pré-processamento.
            process_rank (int): O rank (ID) do processo atual em um ambiente distribuído.
            total_process_count (int): O número total de processos (GPUs) no ambiente distribuído.
            image_tile_size (int): A dimensão (altura e largura) das imagens quadradas.
            estimated_rows_per_file (int): Estimativa de linhas por arquivo para cálculo de __len__.
        """
        super().__init__()
        self.parquet_file_paths = sorted(parquet_file_paths)
        self.image_processor = image_processor
        self.process_rank = process_rank
        self.total_process_count = total_process_count
        self.image_tile_size = image_tile_size
        self.estimated_rows_per_file = estimated_rows_per_file

    def _determine_worker_files(self) -> List[str]:
        """
        Calcula quais arquivos este worker específico deve processar.
        Considera tanto a divisão por GPU (rank) quanto a divisão por Worker do DataLoader.

        Retorno:
            List[str]: Subconjunto de caminhos de arquivos atribuídos a este worker.
        """
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

    def _convert_bytes_to_tensors(
        self, ortho_bytes: bytes, dtm_bytes: bytes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converte os bytes brutos do Parquet em tensores PyTorch processados.
        Método isolado para facilitar testes unitários sem necessidade de I/O de disco.

        Parâmetros:
            ortho_bytes (bytes): Bytes da imagem ortofoto.
            dtm_bytes (bytes): Bytes do modelo digital de terreno (DTM).

        Retorno:
            Tuple[torch.Tensor, torch.Tensor]: Tupla contendo (pixel_values, dtm_tensor).
        """
        orthophoto_numpy_array = np.frombuffer(ortho_bytes, dtype=np.float32).reshape(
            self.image_tile_size, self.image_tile_size
        )

        digital_terrain_model_numpy_array = np.frombuffer(dtm_bytes, dtype=np.float32).reshape(
            self.image_tile_size, self.image_tile_size
        )

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
        """
        Itera sobre os arquivos atribuídos, lendo grupos de linhas e gerando pares de tensores.

        Retorno:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]: Gerador que produz (imagem_entrada, profundidade_alvo).
        """
        files_assigned_to_worker = self._determine_worker_files()

        for file_path in files_assigned_to_worker:
            try:
                parquet_file = parquet.ParquetFile(file_path)

                for group_index in range(parquet_file.num_row_groups):
                    row_group_batch = parquet_file.read_row_group(group_index)
                    dataframe = row_group_batch.to_pandas()

                    for _, data_row in dataframe.iterrows():
                        yield self._convert_bytes_to_tensors(
                            data_row["ortho_bytes"], data_row["dtm_bytes"]
                        )

            except Exception as error:
                print(f"Erro crítico ao ler arquivo {file_path}: {error}")
                continue

    def __len__(self) -> int:
        """
        Retorna o tamanho estimado do dataset.
        Nota: Em IterableDatasets, isso é apenas uma estimativa para barras de progresso.

        Retorno:
            int: Número total estimado de amostras.
        """
        return len(self.parquet_file_paths) * self.estimated_rows_per_file
