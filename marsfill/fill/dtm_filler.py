from logging import Logger
import os
from pathlib import Path
import shutil
from typing import Tuple

import numpy as np
from osgeo import gdal
from tqdm import tqdm

from marsfill.model.eval import Evaluator 

logger = Logger(__name__)

class DTMFiller:
    def __init__(self,
        evaluator: Evaluator,
        padding_size: int,
        tile_size: int,
    ):
        self._evaluator = evaluator
        self._padding_size = padding_size
        self._tile_size = tile_size

    def fill(self, ortho_path: Path, dtm_path: Path, output_folder: Path) -> None:
        output_file_path = output_folder / f"predicted_{os.path.basename(dtm_path)}"

        if not os.path.exists(output_file_path):
            logger.info(f"Copiando DTM original para: {output_file_path}")
            shutil.copy(dtm_path, output_file_path)

        orthoimage_dataset = gdal.Open(str(ortho_path), gdal.GA_ReadOnly)
        output_dataset = gdal.Open(str(output_file_path), gdal.GA_Update)

        if not orthoimage_dataset or not output_dataset:
            logger.error("Erro ao abrir arquivos GDAL.")
            raise FileNotFoundError("Erro ao abrir arquivos.")
        
        orthoimage_band = orthoimage_dataset.GetRasterBand(1)
        output_band = output_dataset.GetRasterBand(1)

        orthoimage_width = orthoimage_dataset.RasterXSize
        orthoimage_height = orthoimage_dataset.RasterYSize

        nodata_val = output_band.GetNoDataValue()

        if nodata_val is None:
            nodata_val = -3.4028234663852886e+38

        logger.info(f"Iniciado preenchimento do arquivo: {os.path.basename(dtm_path)}")
        logger.info(f"Orthoimage base carregada: {os.path.basename(ortho_path)}")
        logger.info(f"📐 Dimensões: {orthoimage_width}x{orthoimage_height} | Padding: {self._padding_size}px")

        steps = []
        for y in range(0, orthoimage_height, self._tile_size):
            for x in range(0, orthoimage_width, self._tile_size):
                steps.append((x, y))

        logger.info(f"Processando {len(steps)} blocos...")

        for x_core, y_core in tqdm(steps, desc="Infilling"):
            w_core, h_core = self._get_read_window_size(orthoimage_width, orthoimage_height, x_core, y_core)
            
            x_read_start, x_read_end = self._get_vertical_points_of_read_window(orthoimage_width, x_core, w_core)
            
            y_read_start, y_read_end = self._get_horizontal_points_of_read_window(orthoimage_height, y_core, h_core)

            w_read = x_read_end - x_read_start
            h_read = y_read_end - y_read_start

            dtm_core = output_band.ReadAsArray(x_core, y_core, w_core, h_core)
  
            if dtm_core is None:
                continue

            mask_nodata_core = (dtm_core == nodata_val) | np.isnan(dtm_core)

            if not np.any(mask_nodata_core):
                continue

            normalized_orthoimage_tile_padded = self._crop_and_normalize_orthoimage(
                orthoimage_band, x_read_start, y_read_start, w_read, h_read
            )

            normalized_dtm_predicted_tile_padded = self._evaluator.predict(
                padding=normalized_orthoimage_tile_padded, 
                width=w_read, 
                height=h_read
            )

            dtm_predicted_tile = self._extract_predicted_dtm(
                normalized_dtm_predicted_tile_padded,
                dtm_core,
                x_core,
                y_core,
                x_read_start,
                y_read_start,
                h_core,
                w_core,
                ~mask_nodata_core
            )

            dtm_core[mask_nodata_core] = dtm_predicted_tile[mask_nodata_core]
            output_band.WriteArray(dtm_core, x_core, y_core)
        
        output_band.FlushCache()
        orthoimage_dataset = None
        output_dataset = None

        logger.info(f"Processamento concluído, arquivo salvo em: {output_file_path}")

    def _extract_predicted_dtm(self, 
            normalized_dtm_predicted_tile_padded: np.ndarray, 
            dtm_core: np.ndarray,
            x_core: int, 
            y_core: int, 
            x_read_start: int, 
            y_read_start: int, 
            h_core: int, 
            w_core: int, 
            valid_mask: np.ndarray
        ) -> np.ndarray:
        
        off_x = x_core - x_read_start
        off_y = y_core - y_read_start

        normalized_dtm_predicted_tile = normalized_dtm_predicted_tile_padded[off_y : off_y + h_core, off_x : off_x + w_core]

        if np.sum(valid_mask) > 10:
            return self._unormalize_dtm_predicted_tile(normalized_dtm_predicted_tile, dtm_core, valid_mask)

        return normalized_dtm_predicted_tile

    def _unormalize_dtm_predicted_tile(self, normalized_dtm_predicted_tile: np.ndarray, dtm_core: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        mu_real = np.mean(dtm_core[valid_mask])
        std_real = np.std(dtm_core[valid_mask])
        mu_pred = np.mean(normalized_dtm_predicted_tile[valid_mask])
        std_pred = np.std(normalized_dtm_predicted_tile[valid_mask])
        
        scale = std_real / (std_pred + 1e-8)
        offset = mu_real - (mu_pred * scale)
    
        return normalized_dtm_predicted_tile * scale + offset

    def _get_read_window_size(self, orthoimage_width: int, orthoimage_height: int, x_core: int, y_core: int) -> Tuple[int, int]:
        w_core = min(self._tile_size, orthoimage_width - x_core)
        h_core = min(self._tile_size, orthoimage_height - y_core)

        return w_core, h_core

    def _get_vertical_points_of_read_window(self, orthoimage_width: int, x_core: int, w_core: int) -> Tuple[int, int]:
        x_read_start = max(0, x_core - self._padding_size)
        x_read_end = min(orthoimage_width, x_core + w_core + self._padding_size)

        return x_read_start, x_read_end
    
    def _get_horizontal_points_of_read_window(self, orthoimage_height: int, y_core: int, h_core: int) -> Tuple[int, int]:
        y_read_start = max(0, y_core - self._padding_size)
        y_read_end = min(orthoimage_height, y_core + h_core + self._padding_size)

        return y_read_start, y_read_end

    def _crop_and_normalize_orthoimage(self, orthoimage_band: gdal.Band, x_read_start: int, y_read_start: int, w_read: int, h_read: int) -> np.ndarray:
        orthoimage_padded = orthoimage_band.ReadAsArray(x_read_start, y_read_start, w_read, h_read).astype(np.float32)

        return self._normalize_orthoimage_padded(orthoimage_padded)

    def _normalize_orthoimage_padded(self, orthoimage_padded: np.ndarray) -> np.ndarray:
        min_v, max_v = orthoimage_padded.min(), orthoimage_padded.max()
        ortho_norm = (orthoimage_padded - min_v) / (max_v - min_v + 1e-8)
        
        return np.stack([ortho_norm]*3, axis=-1)
