from logging import Logger
import os
from pathlib import Path
import shutil
from typing import Tuple

import numpy as np
from osgeo import gdal
from tqdm import tqdm
from scipy.ndimage import binary_dilation, gaussian_filter

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

    def fill(self, ortho_path: Path, dtm_path: Path, output_folder: Path) -> tuple[Path, Path]:
        output_file_path = output_folder / f"predicted_{os.path.basename(dtm_path).split(".")[0].lower()}.tif"
        output_file_mask_path = output_folder / f"mask_{os.path.basename(dtm_path).split(".")[0].lower()}.tif"

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

        driver = gdal.GetDriverByName('GTiff')
        mask_dataset = driver.Create(
            str(output_file_mask_path),
            orthoimage_width,
            orthoimage_height,
            1,
            gdal.GDT_Byte,
            options=['COMPRESS=LZW']
        )
        mask_dataset.SetGeoTransform(orthoimage_dataset.GetGeoTransform())
        mask_dataset.SetProjection(orthoimage_dataset.GetProjection())
        
        mask_band = mask_dataset.GetRasterBand(1)
        mask_band.SetNoDataValue(0)

        logger.info(f"Iniciado preenchimento do arquivo: {os.path.basename(dtm_path)}")
        logger.info(f"Orthoimage base carregada: {os.path.basename(ortho_path)}")
        logger.info(f"📐 Dimensões: {orthoimage_width}x{orthoimage_height} | Padding: {self._padding_size}px")

        steps = []
        for y in range(0, orthoimage_height, self._tile_size):
            for x in range(0, orthoimage_width, self._tile_size):
                steps.append((x, y))

        logger.info(f"Processando {len(steps)} blocos...")

        for x_tile, y_tile in tqdm(steps, desc="Infilling"):
            w_tile, h_tile = self._calculate_tile_size(orthoimage_width, orthoimage_height, x_tile, y_tile)
            
            x_box_start, x_box_end = self._calculate_vertical_bound_box_distance(orthoimage_width, x_tile, w_tile)
            
            y_box_start, y_box_end = self._calculate_horizontal_bound_box_distance(orthoimage_height, y_tile, h_tile)

            w_box = x_box_end - x_box_start
            h_box = y_box_end - y_box_start

            dtm_tile = output_band.ReadAsArray(x_tile, y_tile, w_tile, h_tile)
  
            if dtm_tile is None:
                continue

            mask_nodata_tile = (dtm_tile == nodata_val) | np.isnan(dtm_tile)

            mask_to_save = mask_nodata_tile.astype(np.uint8)
            mask_band.WriteArray(mask_to_save, x_tile, y_tile)

            if not np.any(mask_nodata_tile):
                continue

            orthoimage_box = self._crop_bounding_box(
                orthoimage_band, x_box_start, y_box_start, w_box, h_box
            )

            orthoimage_box_normalized = self._normalize_orthoimage(orthoimage_box)

            dtm_predicted_box_normalized = self._evaluator.predict(
                orthoimage=orthoimage_box_normalized, 
                width=w_box, 
                height=h_box
            )

            dtm_predicted_tile_normalized = self._crop_title_of_dtm_box(
                dtm_predicted_box_normalized,
                x_tile,
                y_tile,
                x_box_start,
                y_box_start,
                h_tile,
                w_tile
            )

            valid_mask = ~mask_nodata_tile
            dtm_predicted_tile = None

            if np.sum(valid_mask) > 10:
                dtm_predicted_tile = self._unormalize_dtm(dtm_predicted_tile_normalized, dtm_tile, valid_mask)
            else:
                dtm_predicted_tile = dtm_predicted_tile_normalized

            dtm_tile_raw = dtm_tile.copy()

            dtm_tile_raw[mask_nodata_tile] = dtm_predicted_tile[mask_nodata_tile]

            if np.any(mask_nodata_tile):
                dtm_tile = self._blend_seams(dtm_tile_raw, mask_nodata_tile)
            else:
                dtm_tile = dtm_tile_raw

            output_band.WriteArray(dtm_tile, x_tile, y_tile)

        output_band.FlushCache()
        mask_band.FlushCache()
        orthoimage_dataset = None
        output_dataset = None

        logger.info(f"Processamento concluído, arquivo salvo em: {output_file_path}")

        return output_file_path, output_file_mask_path

    def _blend_seams(self, dtm_filled: np.ndarray, mask_hole: np.ndarray, width: int = 5) -> np.ndarray:
        dilated_mask = binary_dilation(mask_hole, iterations=width)
        dtm_blurred = gaussian_filter(dtm_filled, sigma=2.0)

        blend_zone = dilated_mask ^ binary_dilation(mask_hole, iterations=0)
        blend_zone = dilated_mask & ~mask_hole

        blend_zone_inner = mask_hole & binary_dilation(~mask_hole, iterations=width)

        final_blend_zone = blend_zone | blend_zone_inner

        dtm_out = dtm_filled.copy()

        dtm_out[final_blend_zone] = dtm_blurred[final_blend_zone]

        return dtm_out

    def _crop_title_of_dtm_box(self, 
            dtm_box: np.ndarray, 
            x_tile: int, 
            y_tile: int, 
            x_box_start: int, 
            y_box_start: int, 
            h_tile: int, 
            w_tile: int,
        ) -> np.ndarray:
        
        off_x = x_tile - x_box_start
        off_y = y_tile - y_box_start

        return dtm_box[off_y : off_y + h_tile, off_x : off_x + w_tile]

    def _unormalize_dtm(self, normalized_dtm_predicted_tile: np.ndarray, dtm_tile: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        mu_real = np.mean(dtm_tile[valid_mask])
        std_real = np.std(dtm_tile[valid_mask])
        mu_pred = np.mean(normalized_dtm_predicted_tile[valid_mask])
        std_pred = np.std(normalized_dtm_predicted_tile[valid_mask])
        
        scale = std_real / (std_pred + 1e-8)
        offset = mu_real - (mu_pred * scale)
    
        return normalized_dtm_predicted_tile * scale + offset

    def _calculate_tile_size(self, orthoimage_width: int, orthoimage_height: int, x_tile: int, y_tile: int) -> Tuple[int, int]:
        w_tile = min(self._tile_size, orthoimage_width - x_tile)
        h_tile = min(self._tile_size, orthoimage_height - y_tile)

        return w_tile, h_tile

    def _calculate_vertical_bound_box_distance(self, orthoimage_width: int, x_tile: int, w_tile: int) -> Tuple[int, int]:
        x_box_start = max(0, x_tile - self._padding_size)
        x_box_end = min(orthoimage_width, x_tile + w_tile + self._padding_size)

        return x_box_start, x_box_end
    
    def _calculate_horizontal_bound_box_distance(self, orthoimage_height: int, y_tile: int, h_tile: int) -> Tuple[int, int]:
        y_box_start = max(0, y_tile - self._padding_size)
        y_box_end = min(orthoimage_height, y_tile + h_tile + self._padding_size)

        return y_box_start, y_box_end

    def _crop_bounding_box(self, orthoimage_band: gdal.Band, x_box_start: int, y_box_start: int, w_box: int, h_box: int) -> np.ndarray:
        orthoimage_box = orthoimage_band.ReadAsArray(x_box_start, y_box_start, w_box, h_box).astype(np.float32)

        return orthoimage_box

    def _normalize_orthoimage(self, orthoimage: np.ndarray) -> np.ndarray:
        min_v, max_v = orthoimage.min(), orthoimage.max()
        ortho_norm = (orthoimage - min_v) / (max_v - min_v + 1e-8)
        
        return np.stack([ortho_norm]*3, axis=-1)
