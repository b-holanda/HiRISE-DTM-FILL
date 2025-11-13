import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import DPTForDepthEstimation, DPTImageProcessor
from pathlib import Path
from enum import Enum
from marsfill.model.combined_loss import LossWights, CombinedLoss
from marsfill.model.hirise_dataset import HiRISeDataset
from marsfill.utils import Logger, load_dataset_files, check_dataset_files, parse_str_list_to_path_list

logger = Logger()

class AvaliabeDevices(Enum):
    GPU = "cuda"
    CPU = "cpu"

class AvaliableModels(Enum):
    INTEL_DPT_LARGE = "Intel/dpt-large"

class Train:
    def __init__(
            self,
            selected_model: AvaliableModels,
            batch_size: int,
            learning_rate: float,
            epochs: int,
            weight_decay: float,
            loss_weights: LossWights,
            # Adicione estes parâmetros
            is_distributed: bool = False,
            local_rank: int = 0,
            rank: int = 0,
            world_size: int = 1
        ) -> None:

        self._is_ddp_external = is_distributed
        self._external_local_rank = local_rank
        self._external_rank = rank
        self._external_world_size = world_size

        self._setup_device_and_ddp()

        if self._is_master:
            logger.info(f"Inicializando modelo: {selected_model.value}...")

        self._processor = DPTImageProcessor.from_pretrained(selected_model.value, do_rescale=False)
        self._loss_calculator = CombinedLoss(lossWeights=loss_weights, device=self._device).to(self._device)
        self._scaler = GradScaler("cuda") if self._device.type == "cuda" else GradScaler("cpu")
        self._batch_size = batch_size
        self._epochs = epochs

        model_raw = DPTForDepthEstimation.from_pretrained(selected_model.value).to(device=self._device) # type: ignore
        self._optmizer = optim.AdamW(model_raw.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if self._is_ddp:
            self._model = DDP(model_raw, device_ids=[self._local_rank], find_unused_parameters=False)
        else:
            self._model = model_raw

    def _setup_device_and_ddp(self) -> None:
        """Detecta o ambiente e configura DDP (Multi-GPU) ou GPU Única."""

        if hasattr(self, '_is_ddp_external') and self._is_ddp_external:
            self._is_ddp = True
            self._world_size = self._external_world_size
            self._rank = self._external_rank
            self._local_rank = self._external_local_rank

            if not torch.cuda.is_available():
                raise RuntimeError("DDP requer CUDA, mas CUDA não está disponível")

            self._device = torch.device(f"cuda:{self._local_rank}")
            torch.cuda.set_device(self._local_rank)
            self._is_master = (self._rank == 0)

            if self._is_master:
                logger.info(f"Modo DDP Ativado: Rank {self._rank}/{self._world_size} na GPU {self._local_rank}")

        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self._is_ddp = True
            self._world_size = int(os.environ['WORLD_SIZE'])
            self._rank = int(os.environ['RANK'])
            self._local_rank = int(os.environ.get('LOCAL_RANK', 0))

            if not torch.cuda.is_available():
                raise RuntimeError("DDP requer CUDA, mas CUDA não está disponível")

            self._device = torch.device(f"cuda:{self._local_rank}")
            torch.cuda.set_device(self._local_rank)
            self._is_master = (self._rank == 0)

            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

            if self._is_master:
                logger.info(f"Modo DDP Ativado: Rank {self._rank}/{self._world_size} na GPU {self._local_rank}")
            else:
                self._is_ddp = False
                self._world_size = 1
                self._rank = 0
                self._local_rank = 0
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._is_master = True
                logger.info(f"Modo Single Device Ativado. Dispositivo: {self._device}")

    def _train(self, loader: DataLoader):
        running_loss = 0.0
        self._model.train()
        
        for i, (pixel_values, dtm_targets) in enumerate(loader):
            pixel_values = pixel_values.to(self._device)
            dtm_targets = dtm_targets.to(self._device)
            
            self._optmizer.zero_grad()
            
            with autocast(self._device.type):
                outputs = self._model(pixel_values)
                predicted_depth = outputs.predicted_depth.unsqueeze(1)
                predicted_resized = F.interpolate(
                    predicted_depth,
                    size=dtm_targets.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
                total_loss, l1_loss, gradiend_loss, ssim_loss = self._loss_calculator(predicted_resized, dtm_targets)
            
            self._scaler.scale(total_loss).backward()
            self._scaler.step(optimizer=self._optmizer)
            self._scaler.update()
            
            running_loss += total_loss.item()
            
            if (i + 1) % 50 == 0 and self._is_master:
                logger.info(f"Batch {i + 1}/{len(loader)}, Loss: {total_loss.item():.4f}, L1: {l1_loss.item():.4f}, Gradent: {gradiend_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}")
        
        return running_loss / len(loader)

    def _validation(self, loader: DataLoader):
        running_vloss = 0.0
        self._model.eval()
        
        with torch.no_grad():
            for pixel_values, dtm_targets in loader:
                pixel_values = pixel_values.to(self._device)
                dtm_targets = dtm_targets.to(self._device)
                
                with autocast(self._device.type):
                    outputs = self._model(pixel_values)
                    predicted_depth = outputs.predicted_depth.unsqueeze(1)
                    predicted_resized = F.interpolate(
                        predicted_depth,
                        size=dtm_targets.shape[2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    total_loss, _, _, _ = self._loss_calculator(predicted_resized, dtm_targets)
                
                running_vloss += total_loss.item()
        
        return running_vloss / len(loader)

    def run(self) -> None:
        if self._is_master:
            logger.info(f"Iniciando treinamento no dispositivo: {self._device}")
            logger.info("Carregando datasets de treinamento e validation...")
        
        # Caminhos flexíveis: SageMaker ou local
        train_dir = Path(os.environ.get('SM_CHANNEL_TRAIN', 'datasets/train'))
        validation_dir = Path(os.environ.get('SM_CHANNEL_VALIDATION', 'datasets/validation'))
        
        train_ortho_files, train_dtm_files = load_dataset_files(
            path=train_dir,
            ortho="ORTHO_*.tif",
            dtm="DTM_*.tif"
        )
        validation_ortho_files, validation_dtm_files = load_dataset_files(
            path=validation_dir,
            ortho="ORTHO_*.tif",
            dtm="DTM_*.tif"
        )
        
        train_ortho_files, train_dtm_files = check_dataset_files(ortho_files=train_ortho_files, dtm_files=train_dtm_files)
        validation_ortho_files, validation_dtm_files = check_dataset_files(ortho_files=validation_ortho_files, dtm_files=validation_dtm_files)
        
        train_dataset = HiRISeDataset(
            ortho_files=parse_str_list_to_path_list(train_ortho_files),
            dtm_files=parse_str_list_to_path_list(train_dtm_files),
            processor=self._processor
        )
        validation_dataset = HiRISeDataset(
            ortho_files=parse_str_list_to_path_list(validation_ortho_files),
            dtm_files=parse_str_list_to_path_list(validation_dtm_files),
            processor=self._processor
        )
        
        train_sampler = None
        validation_sampler = None
        train_shuffle = True
        
        if self._is_ddp:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self._world_size, rank=self._rank, shuffle=True
            )
            validation_sampler = DistributedSampler(
                validation_dataset, num_replicas=self._world_size, rank=self._rank, shuffle=False
            )
            train_shuffle = False
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=train_shuffle,
            num_workers=4,
            pin_memory=True if self._device.type == "cuda" else False,
            sampler=train_sampler
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self._device.type == "cuda" else False,
            sampler=validation_sampler
        )
        
        if self._is_master:
            logger.info("Iniciando o loop de Fine-Tuning...")
        
        best_vloss = float('inf')
        epochs_no_improve = 0
        PATIENCE = 5
        
        for epoch in range(self._epochs):
            if self._is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            if self._is_master:
                logger.info(f"\n--- ÉPOCA {epoch+1}/{self._epochs} ---")
            
            avg_loss = self._train(loader=train_loader)
            avg_vloss = self._validation(loader=validation_loader)
            
            if self._is_master:
                logger.info(f"Fim da Época. Loss de Treino: {avg_loss:.4f} | Loss de Validação: {avg_vloss:.4f}")
                
                if avg_vloss < best_vloss:
                    logger.info(f"Perda de validação melhorou de {best_vloss:.4f} para {avg_vloss:.4f}.")
                    best_vloss = avg_vloss
                    epochs_no_improve = 0
                    
                    default_output_path = Path(__file__).parent.parent.parent / "outputs"
                    output_dir = Path(os.environ.get('SM_MODEL_DIR', default_output_path))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    model_save_path = output_dir / "marsfill_model.pth"
                    model_to_save = self._model.module if self._is_ddp else self._model
                    torch.save(model_to_save.state_dict(), model_save_path)
                    logger.info(f"Modelo salvo em {model_save_path}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"Sem melhoria na validação por {epochs_no_improve} épocas.")
                
                if epochs_no_improve >= PATIENCE:
                    logger.info(f"Parando o treinamento (Early Stopping) após {epoch+1} épocas.")
                    break

        if self._is_master:
            logger.info("Treinamento concluído.")

