import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from transformers import DPTForDepthEstimation, DPTImageProcessor
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import boto3

from marsfill.model.combined_loss import LossWeights, CombinedLoss
from marsfill.model.hirise_dataset import StreamingHiRISeDataset
from marsfill.utils import Logger, list_parquet_files

class AvailableDevices(Enum):
    GPU = "cuda"
    CPU = "cpu"

class AvailableModels(Enum):
    INTEL_DPT_LARGE = "Intel/dpt-large"

class MarsDepthTrainer:
    """
    Classe responsável por gerenciar o ciclo de vida do treinamento do modelo de estimativa de profundidade.
    Gerencia configuração de ambiente distribuído, carregamento de dados, loops de treino/validação e persistência.
    """

    def __init__(
        self,
        selected_model_name: AvailableModels,
        batch_size: int,
        learning_rate: float,
        total_epochs: int,
        weight_decay: float,
        loss_weights: LossWeights,
        storage_mode: str,
        dataset_root: str,
        dataset_prefix: str = "dataset/v1",
        output_prefix: str = "models",
        is_distributed: bool = False,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        injected_model: Optional[torch.nn.Module] = None,
        injected_optimizer: Optional[optim.Optimizer] = None,
        s3_client: Optional[boto3.client] = None,
        logger_instance: Optional[Logger] = None
    ) -> None:
        """
        Inicializa o treinador com as configurações de hiperparâmetros, caminhos e ambiente.

        Parâmetros:
            selected_model_name (AvailableModels): Enum indicando qual modelo base carregar.
            batch_size (int): Tamanho do lote de imagens por passo.
            learning_rate (float): Taxa de aprendizado para o otimizador.
            total_epochs (int): Número total de épocas de treinamento.
            weight_decay (float): Fator de decaimento de peso para regularização.
            loss_weights (LossWeights): Pesos para as diferentes componentes da função de perda.
            storage_mode (str): Modo de armazenamento, aceita "local" ou "s3".
            dataset_root (str): Caminho raiz local ou nome do bucket S3.
            dataset_prefix (str): Prefixo ou subpasta onde os dados estão localizados.
            output_prefix (str): Prefixo ou subpasta onde o modelo será salvo.
            is_distributed (bool): Define se o treinamento ocorre em ambiente distribuído manualmente.
            local_rank (int): Rank do processo na máquina local.
            global_rank (int): Rank global do processo no cluster.
            world_size (int): Número total de processos no cluster.
            injected_model (Optional[torch.nn.Module]): Modelo pré-instanciado para injeção de dependência (testes).
            injected_optimizer (Optional[optim.Optimizer]): Otimizador pré-instanciado para injeção de dependência.
            s3_client (Optional[boto3.client]): Cliente Boto3 pré-configurado.
            logger_instance (Optional[Logger]): Instância de logger customizada.
        """
        self.logger = logger_instance if logger_instance else Logger()
        
        self.is_external_distributed = is_distributed
        self.external_local_rank = local_rank
        self.external_global_rank = global_rank
        self.external_world_size = world_size
        
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.selected_model_name = selected_model_name
        
        self.storage_mode = storage_mode
        self.dataset_root = dataset_root
        self.s3_client = s3_client

        self._setup_processing_environment()
        self._resolve_io_paths(dataset_prefix, output_prefix)

        if self.is_master_process:
            self.logger.info(f"Modo de Armazenamento: {self.storage_mode.upper()}")
            self.logger.info(f"Fonte de Treino:      {self.training_uri}")
            self.logger.info(f"Fonte de Validação:   {self.validation_uri}")
            self.logger.info(f"Destino do Modelo:    {self.model_output_uri}")

        self.image_processor = DPTImageProcessor.from_pretrained(selected_model_name.value, do_rescale=False)
        self.loss_calculator = CombinedLoss(loss_weights=loss_weights).to(self.device)
        self.gradient_scaler = GradScaler("cuda") if self.device.type == "cuda" else GradScaler("cpu")

        if injected_model:
            raw_model = injected_model.to(self.device)
        else:
            raw_model = DPTForDepthEstimation.from_pretrained(selected_model_name.value).to(self.device) # type: ignore

        if injected_optimizer:
            self.optimizer = injected_optimizer
        else:
            self.optimizer = optim.AdamW(raw_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if self.is_distributed_mode:
            self.model = DistributedDataParallel(
                raw_model, 
                device_ids=[self.local_rank], 
                find_unused_parameters=True
            )
        else:
            self.model = raw_model

    def _resolve_io_paths(self, input_prefix: str, output_prefix: str) -> None:
        """
        Configura as URIs completas para leitura de dados e escrita do modelo baseando-se no modo de armazenamento.

        Parâmetros:
            input_prefix (str): Caminho relativo para entrada dos dados.
            output_prefix (str): Caminho relativo para saída do modelo.
        
        Retorno:
            None: Atualiza os atributos self.training_uri, self.validation_uri e self.model_output_uri.
        
        Levanta:
            ValueError: Se o storage_mode não for 'local' nem 's3'.
        """
        clean_input = input_prefix.strip("/")
        clean_output = output_prefix.strip("/")

        if self.storage_mode == "s3":
            base_uri = f"s3://{self.dataset_root}"
            self.training_uri = f"{base_uri}/{clean_input}/train"
            self.validation_uri = f"{base_uri}/{clean_input}/validation"
            self.model_output_uri = f"{base_uri}/{clean_output}/"
            
            if not self.s3_client:
                self.s3_client = boto3.client('s3')

        elif self.storage_mode == "local":
            base_path = Path(__file__).parent.parent.parent / Path(self.dataset_root)
            self.training_uri = str(base_path / clean_input / "train")
            self.validation_uri = str(base_path / clean_input / "validation")
            self.model_output_uri = Path(__file__).parent.parent.parent / str(base_path / clean_output)
            
        else:
            raise ValueError("Storage mode deve ser 'local' ou 's3'")

    def _setup_processing_environment(self) -> None:
        """
        Configura o ambiente de processamento, detectando se há suporte a CUDA e se a execução é distribuída (DDP).
        Define ranks locais e globais e inicializa o grupo de processos se necessário.

        Retorno:
            None: Configura self.device, self.is_distributed_mode e variáveis de rank.
        """
        if self.is_external_distributed:
            self.is_distributed_mode = True
            self.world_size = self.external_world_size
            self.global_rank = self.external_global_rank
            self.local_rank = self.external_local_rank
        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.is_distributed_mode = True
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.global_rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            self.is_distributed_mode = False
            self.world_size = 1
            self.global_rank = 0
            self.local_rank = 0

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device("cpu")
            
        self.is_master_process = (self.global_rank == 0)

        if self.is_distributed_mode and not distributed.is_initialized():
             distributed.init_process_group(backend='nccl')

    def _upload_file_to_s3(self, local_file_path: str, filename: str) -> None:
        """
        Realiza o upload de um arquivo local para o bucket S3 configurado.

        Parâmetros:
            local_file_path (str): Caminho absoluto do arquivo temporário local.
            filename (str): Nome do arquivo final no destino (chave S3).

        Retorno:
            None
        """
        bucket_name = self.dataset_root
        
        prefix = self.model_output_uri.replace(f"s3://{bucket_name}/", "")
        s3_key = f"{prefix.strip('/')}/{filename}"
        
        try:
            self.s3_client.upload_file(local_file_path, bucket_name, s3_key)
            self.logger.info(f"☁️  Upload S3 concluído: s3://{bucket_name}/{s3_key}")
        except Exception as error:
            self.logger.error(f"Erro ao fazer upload do modelo: {error}")

    def _save_model_checkpoint(self, filename: str = "marsfill_model.pth") -> None:
        """
        Salva o estado atual do modelo (state_dict). Suporta salvamento local direto ou upload para S3.
        Apenas o processo mestre executa esta ação.

        Parâmetros:
            filename (str): Nome do arquivo a ser salvo.

        Retorno:
            None
        """
        if not self.is_master_process:
            return

        model_to_save = self.model.module if self.is_distributed_mode else self.model
        
        if self.storage_mode == "local":
            output_path = Path(self.model_output_uri)
            output_path.mkdir(parents=True, exist_ok=True)
            
            full_path = output_path / filename
            torch.save(model_to_save.state_dict(), full_path)
            self.logger.info(f"💾 Modelo salvo localmente em: {full_path}")
            
        elif self.storage_mode == "s3":
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=True) as temporary_file:
                torch.save(model_to_save.state_dict(), temporary_file.name)
                self.logger.info(f"Salvamento temporário concluído. Iniciando upload para S3...")
                self._upload_file_to_s3(temporary_file.name, filename)

    def _execute_training_step(self, data_loader: DataLoader) -> float:
        """
        Executa uma época completa de treinamento iterando sobre o DataLoader fornecido.
        Calcula perda, gradientes e atualiza os pesos.

        Parâmetros:
            data_loader (DataLoader): Loader contendo os dados de treinamento.

        Retorno:
            float: A perda média (average loss) acumulada durante a época.
        """
        accumulated_loss = 0.0
        self.model.train()
        batch_counter = 0
        
        for batch_index, (input_images, ground_truth_depth) in enumerate(data_loader):
            input_images = input_images.to(self.device)
            ground_truth_depth = ground_truth_depth.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(self.device.type):
                outputs = self.model(input_images)
                predicted_depth_map = outputs.predicted_depth.unsqueeze(1)
                predicted_depth_resized = F.interpolate(
                    predicted_depth_map, size=ground_truth_depth.shape[2:],
                    mode="bilinear", align_corners=False
                )
                total_loss, _, _, _ = self.loss_calculator(predicted_depth_resized, ground_truth_depth)
            
            self.gradient_scaler.scale(total_loss).backward()
            self.gradient_scaler.step(optimizer=self.optimizer)
            self.gradient_scaler.update()
            
            accumulated_loss += total_loss.item()
            batch_counter += 1
            
            if (batch_index + 1) % 50 == 0 and self.is_master_process:
                self.logger.info(f"Step {batch_index + 1}, Loss: {total_loss.item():.4f}")
        
        return accumulated_loss / max(batch_counter, 1)

    def _execute_validation_step(self, data_loader: DataLoader) -> float:
        """
        Executa uma época completa de validação. Não atualiza gradientes.

        Parâmetros:
            data_loader (DataLoader): Loader contendo os dados de validação.

        Retorno:
            float: A perda média (average loss) calculada no conjunto de validação.
        """
        accumulated_loss = 0.0
        self.model.eval()
        batch_counter = 0
        with torch.no_grad():
            for input_images, ground_truth_depth in data_loader:
                input_images = input_images.to(self.device)
                ground_truth_depth = ground_truth_depth.to(self.device)
                with autocast(self.device.type):
                    outputs = self.model(input_images)
                    predicted_depth = outputs.predicted_depth.unsqueeze(1)
                    predicted_resized = F.interpolate(
                        predicted_depth, size=ground_truth_depth.shape[2:],
                        mode="bilinear", align_corners=False
                    )
                    total_loss, _, _, _ = self.loss_calculator(predicted_resized, ground_truth_depth)
                accumulated_loss += total_loss.item()
                batch_counter += 1
        return accumulated_loss / max(batch_counter, 1)

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Lista arquivos parquet e cria instâncias de DataLoader para treino e validação.

        Retorno:
            Tuple[DataLoader, DataLoader]: Uma tupla contendo (training_loader, validation_loader).
        
        Levanta:
            FileNotFoundError: Se nenhum arquivo for encontrado no diretório de treino.
        """
        training_files = list_parquet_files(self.training_uri)
        validation_files = list_parquet_files(self.validation_uri)

        if not training_files:
            message = f"Nenhum dado encontrado em {self.training_uri}"
            self.logger.error(message)
            raise FileNotFoundError(message)

        if self.is_master_process:
            self.logger.info(f"Arquivos Treino: {len(training_files)} | Validação: {len(validation_files)}")

        training_dataset = StreamingHiRISeDataset(training_files, self.image_processor, 
                                                  self.global_rank, self.world_size, 512)
        validation_dataset = StreamingHiRISeDataset(validation_files, self.image_processor, 
                                                    self.global_rank, self.world_size, 512)
        
        training_loader = DataLoader(training_dataset, batch_size=self.batch_size, num_workers=4, 
                                     pin_memory=(self.device.type=="cuda"), shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, num_workers=4, 
                                       pin_memory=(self.device.type=="cuda"), shuffle=False)
        return training_loader, validation_loader

    def run_training_loop(self) -> None:
        """
        Executa o loop principal de treinamento, orquestrando as épocas, validação, early stopping e salvamento do modelo.
        """
        if self.is_master_process:
            self.logger.info(f"Iniciando loop de treinamento...")
        
        training_loader, validation_loader = self.create_dataloaders()
        
        best_loss = float('inf')
        patience_limit = 5
        epochs_without_improvement = 0
        
        for epoch_index in range(self.total_epochs):
            if self.is_master_process:
                self.logger.info(f"\n--- ÉPOCA {epoch_index + 1}/{self.total_epochs} ---")
            
            training_loss = self._execute_training_step(training_loader)
            validation_loss = self._execute_validation_step(validation_loader)
            
            if self.is_master_process:
                self.logger.info(f"Treino: {training_loss:.4f} | Validação: {validation_loss:.4f}")
                
                if validation_loss < best_loss:
                    self.logger.info(f"Melhoria: {best_loss:.4f} -> {validation_loss:.4f}")
                    best_loss = validation_loss
                    epochs_without_improvement = 0
                    self._save_model_checkpoint("marsfill_model.pth")
                else:
                    epochs_without_improvement += 1
                    self.logger.info(f"Sem melhoria ({epochs_without_improvement}/{patience_limit})")
                
                if epochs_without_improvement >= patience_limit:
                    self.logger.info("Early Stopping ativado.")
                    break
        
        if self.is_distributed_mode and distributed.is_initialized():
            distributed.destroy_process_group()
