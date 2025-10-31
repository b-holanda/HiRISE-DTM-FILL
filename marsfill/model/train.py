import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from transformers import DPTForDepthEstimation, DPTImageProcessor
from torchmetrics import StructuralSimilarityIndexMeasure
from pathlib import Path
from osgeo import gdal
from enum import Enum
import numpy as np
import glob
import os

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
            data_dir: Path,
            loss_weights: LossWights
     ) -> None:

        logger.info(f"Inicializando modelo: {selected_model.value}...")

        self._device = torch.device(selected_device.value)
        self._processor = DPTImageProcessor.from_pretrained(selected_model.value)
        self._loss_calculator = CombinedLoss(lossWeights=loss_weights, device=self._device).to(self._device)
        self._scaler = GradScaler(selected_device.value)
        self._batch_size = batch_size
        self._epochs = epochs
        self._data_dir = data_dir
        self._model = DPTForDepthEstimation.from_pretrained(selected_model.value).to(device=self._device)
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

    def _test(self, loader: DataLoader):
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

        logger.info("Carregando datasets de treinamento e teste...")

        train_dir = self._data_dir / "train"
        test_dir = self._data_dir / "test"

        train_ortho_files, train_dtm_files = load_dataset_files(
            path=train_dir, 
            ortho="ORTHO_*.tif", 
            dtm="DTM_*.tif"
        )

        test_ortho_files, test_dtm_files = load_dataset_files(
            path=test_dir, 
            ortho="ORTHO_*.tif", 
            dtm="DTM_*.tif"
        )

        train_ortho_files, train_dtm_files = check_dataset_files(ortho_files=train_ortho_files, dtm_files=train_dtm_files)
        test_ortho_files, test_dtm_files =  check_dataset_files(ortho_files=test_ortho_files, dtm_files=test_dtm_files)

        train_dataset = HiRISeDataset(
            ortho_files=parse_str_list_to_path_list(train_ortho_files), 
            dtm_files=parse_str_list_to_path_list(train_dtm_files), 
            processor=self._processor
        )
        test_dataset = HiRISeDataset(
            ortho_files=parse_str_list_to_path_list(test_ortho_files), 
            dtm_files=parse_str_list_to_path_list(test_dtm_files), 
            processor=self._processor
        )

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4, pin_memory=True)
