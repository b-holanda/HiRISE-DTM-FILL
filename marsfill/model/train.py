import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

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
            selected_device: AvaliabeDevices, 
            selected_model: AvaliableModels,
            batch_size: int,
            learning_rate: float,
            epochs: int,
            weight_decay: float,
            loss_weights: LossWights
     ) -> None:

        logger.info(f"Inicializando modelo: {selected_model.value}...")

        self._device = torch.device(selected_device.value)
        self._processor = DPTImageProcessor.from_pretrained(selected_model.value, do_rescale=False)
        self._loss_calculator = CombinedLoss(lossWeights=loss_weights, device=self._device).to(self._device)
        self._scaler = GradScaler(selected_device.value)
        self._batch_size = batch_size
        self._epochs = epochs
        self._model = DPTForDepthEstimation.from_pretrained(selected_model.value).to(device=self._device) # type: ignore
        self._optmizer = optim.AdamW(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

            if (i + 1) % 50 == 0:
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
        logger.info(f"Iniciando treinamento no dispositivo: {self._device}")

        logger.info("Carregando datasets de treinamento e validatione...")

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
        validation_ortho_files, validation_dtm_files =  check_dataset_files(ortho_files=validation_ortho_files, dtm_files=validation_dtm_files)

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

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4, pin_memory=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4, pin_memory=True)

        logger.info("Iniciando o loop de Fine-Tuning...")
        best_vloss = float('inf')
        epochs_no_improve = 0
        PATIENCE = 5

        for epoch in range(self._epochs):
            logger.info(f"\n--- ÉPOCA {epoch+1}/{self._epochs} ---")

            avg_loss = self._train(loader=train_loader)
            avg_vloss = self._validation(loader=validation_loader)

            logger.info(f"Fim da Época. Loss de Treino: {avg_loss:.4f} | Loss de Validação: {avg_vloss:.4f}")

            if avg_loss < best_vloss:
                best_vloss = avg_vloss
                epochs_no_improve = 0

                torch.save(self._model.state_dict(), "/opt/ml/model/marsfill_model.pth")

                logger.info(f"Modelo salvo em /opt/ml/model/marsfill_model.pth (Melhor Loss Val: {best_vloss:.4f})")
            else:
                epochs_no_improve += 1
                logger.info(f"Sem melhoria na validação por {epochs_no_improve} épocas.")
            
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Parando o treinamento (Early Stopping) após {epoch+1} épocas.")
                break
        
        logger.info("Treinamento concluído.")
