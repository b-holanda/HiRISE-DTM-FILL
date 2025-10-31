import torch
from torch.utils.data import Dataset
from typing import Any
from pathlib import Path

from transformers import DPTImageProcessor

from osgeo import gdal
import numpy as np

from marsfill.utils import Logger

gdal.UseExceptions()

logger = Logger()

class HiRISeDataset(Dataset):
    def __init__(self, ortho_files: list[Path], dtm_files: list[Path], processor: DPTImageProcessor) -> None:
        super().__init__()

        self._ortho_files = ortho_files
        self._dtm_files = dtm_files
        self._processor = processor

    def __len__(self):
        return len(self._ortho_files)
    
    def __getitem__(self, index) -> Any:
        try:
            ortho_path = self._ortho_files[index]
            ortho_ds = gdal.Open(ortho_path)
            ortho_array = ortho_ds.GetRasterBand(1).ReadAsArray()
            ortho_ds = None

            ortho_rgb_array = np.stack([ortho_array, ortho_array, ortho_array], axis=-1)
            
            dtm_path = self._dtm_files[index]
            dtm_ds = gdal.Open(dtm_path)
            dtm_array = dtm_ds.GetRasterBand(1).ReadAsArray()
            ortho_ds = None

            inputs = self._processor(ortho_rgb_array, return_tensors="pt")

            pixel_values = inputs["pixel_values"].squeeze(0)

            dtm_tensor = torch.from_numpy(dtm_array).float().unsqueeze(0)

            return pixel_values, dtm_tensor
        except Exception as e:
            logger.error(f"Erro ao carregar {self._ortho_files[index]} ou {self._dtm_files[index]}: {e}")

            return torch.zeros((3, 384, 384)), torch.zeros((1, 512, 512))
