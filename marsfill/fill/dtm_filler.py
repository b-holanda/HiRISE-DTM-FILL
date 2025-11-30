import os
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, List
from osgeo import gdal
from tqdm import tqdm
from scipy.ndimage import binary_dilation, gaussian_filter

# Ajuste o import conforme sua estrutura. 
# Se Evaluator estiver em marsfill.model.eval, mantenha assim.
from marsfill.model.eval import Evaluator
from marsfill.utils import Logger

logger = Logger()

class DTMFiller:
    """
    Preenche lacunas em DTMs usando inferência de profundidade com tratamento robusto de NoData.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        padding_size: int,
        tile_size: int,
    ) -> None:
        self.depth_evaluator = evaluator
        self.context_padding_size = padding_size
        self.processing_tile_size = tile_size

    def fill(
        self,
        dtm_path: str | Path,
        ortho_path: str | Path,
        output_root: str,
        keep_local_output: bool = False,
    ) -> Tuple[str, str, Path, Path, Path]:
        
        working_directory = Path(output_root)
        working_directory.mkdir(parents=True, exist_ok=True)

        local_ortho = Path(ortho_path)
        local_dtm = Path(dtm_path)
        original_dtm_path = local_dtm

        # Prepara arquivos de trabalho (cópia)
        working_dtm_path, working_mask_path = self._prepare_working_files(
            local_dtm, working_directory
        )

        try:
            self._execute_filling_process(local_ortho, working_dtm_path, working_mask_path)
            
            return (
                str(working_dtm_path),
                str(working_mask_path),
                working_dtm_path,
                working_mask_path,
                original_dtm_path,
            )
        except Exception as e:
            logger.error(f"Erro no processo de preenchimento: {e}")
            raise

    def _prepare_working_files(
        self, source_dtm_path: Path, working_directory: Path
    ) -> Tuple[Path, Path]:
        base_filename = Path(source_dtm_path).stem
        working_dtm_path = working_directory / f"{base_filename}_filled.tif"
        working_mask_path = working_directory / f"{base_filename}_filled_mask.tif"

        # Sempre sobrescreve para garantir estado limpo a cada execução
        shutil.copy(source_dtm_path, working_dtm_path)
        return working_dtm_path, working_mask_path

    def _execute_filling_process(
        self, orthophoto_path: Path, dtm_path: Path, mask_path: Path
    ) -> None:
        dtm_dataset = gdal.Open(str(dtm_path), gdal.GA_Update)
        if not dtm_dataset:
            raise FileNotFoundError("Falha ao abrir DTM.")

        # Alinhamento de Ortofoto e DTM
        # Cria um arquivo temporário alinhado para garantir que pixels batam 1:1
        ortho_aligned_path = dtm_path.parent / "aligned_ortho_temp.tif"
        orthophoto_dataset = self._align_rasters(dtm_dataset, orthophoto_path, ortho_aligned_path)

        ortho_band = orthophoto_dataset.GetRasterBand(1)
        dtm_band = dtm_dataset.GetRasterBand(1)

        width = dtm_dataset.RasterXSize
        height = dtm_dataset.RasterYSize
        
        no_data_val = dtm_band.GetNoDataValue()

        # Cria dataset da máscara
        mask_dataset = self._create_mask_dataset(
            mask_path, width, height, dtm_dataset.GetGeoTransform(), dtm_dataset.GetProjection()
        )
        mask_band = mask_dataset.GetRasterBand(1)

        tile_coords = self._generate_processing_grid(width, height)
        
        logger.info(f"Iniciando preenchimento: {len(tile_coords)} tiles a processar.")
        
        count_fills = 0
        for x, y in tqdm(tile_coords, desc="Processando Tiles"):
            filled = self._process_single_tile(
                x, y, width, height, ortho_band, dtm_band, mask_band, no_data_val
            )
            if filled:
                count_fills += 1

        logger.info(f"Tiles com preenchimento ativo: {count_fills}")

        dtm_dataset.FlushCache()
        mask_dataset.FlushCache()
        
        # Limpeza temporária
        orthophoto_dataset = None
        if ortho_aligned_path.exists():
            try:
                os.remove(ortho_aligned_path)
            except:
                pass

    def _align_rasters(self, dtm_ds, ortho_path, output_path):
        """Reamostra ortofoto para bater pixel-a-pixel com DTM usando GDAL Warp."""
        dtm_gt = dtm_ds.GetGeoTransform()
        dtm_proj = dtm_ds.GetProjection()
        width = dtm_ds.RasterXSize
        height = dtm_ds.RasterYSize
        
        bounds = [
            dtm_gt[0],
            dtm_gt[3] + dtm_gt[5] * height,
            dtm_gt[0] + dtm_gt[1] * width,
            dtm_gt[3],
        ]
        
        try:
            gdal.Warp(
                destNameOrDestDS=str(output_path),
                srcDSOrSrcDSTab=str(ortho_path),
                format="GTiff",
                outputBounds=bounds,
                xRes=dtm_gt[1],
                yRes=abs(dtm_gt[5]),
                dstSRS=dtm_proj,
                resampleAlg="cubic",
            )
            return gdal.Open(str(output_path), gdal.GA_ReadOnly)
        except Exception as e:
            logger.warning(f"Alinhamento via Warp falhou, tentando abrir direto: {e}")
            return gdal.Open(str(ortho_path), gdal.GA_ReadOnly)

    def _process_single_tile(
        self, x, y, total_w, total_h, ortho_band, dtm_band, mask_band, no_data_val
    ) -> bool:
        w_tile = min(self.processing_tile_size, total_w - x)
        h_tile = min(self.processing_tile_size, total_h - y)

        dtm_data = dtm_band.ReadAsArray(x, y, w_tile, h_tile)
        
        # --- LÓGICA DE DETECÇÃO ROBUSTA ---
        # Resolve o problema de precisão de float (ex: -3.4e38 vs -3.40001e38)
        is_nan = np.isnan(dtm_data)
        
        if no_data_val is not None and no_data_val < -1e30:
            # Se o NoData for um número negativo muito grande (padrão GDAL),
            # usa comparação por limiar (< -1e30) para pegar todos os pixels inválidos.
            is_nodata = (dtm_data < -1e30)
        elif no_data_val is not None:
            # Para outros valores (ex: -9999), usa tolerância
            is_nodata = np.isclose(dtm_data, no_data_val, equal_nan=True)
        else:
            # Se não há metadado, assume padrão float32 mínimo
            is_nodata = (dtm_data < -1e30)

        missing_mask = is_nodata | is_nan
        # ----------------------------------

        # Se não há buraco neste tile, marca máscara como 0 e pula
        if not np.any(missing_mask):
            mask_band.WriteArray(np.zeros_like(missing_mask, dtype=np.uint8), x, y)
            return False

        # Escreve na máscara onde vamos preencher (1 = Preenchido)
        mask_band.WriteArray(missing_mask.astype(np.uint8), x, y)

        # Contexto expandido para inferência
        bbox = self._calculate_context_bounding_box(total_w, total_h, x, y, w_tile, h_tile)
        
        ortho_crop = ortho_band.ReadAsArray(
            bbox["x_start"], bbox["y_start"], bbox["width"], bbox["height"]
        ).astype(np.float32)

        # Inferência
        normalized_ortho = self._normalize_image(ortho_crop)
        predicted_depth_box = self.depth_evaluator.predict_depth(
            orthophoto_image=normalized_ortho,
            target_width=bbox["width"],
            target_height=bbox["height"],
        )

        # Recorta apenas a área de interesse (ROI) do contexto
        predicted_tile = self._crop_tile_from_context_box(
            predicted_depth_box, x, y, bbox["x_start"], bbox["y_start"], h_tile, w_tile
        )

        # Ajuste estatístico (Denormalização)
        valid_pixels_mask = ~missing_mask
        final_prediction = predicted_tile

        # Usa estatística local se houver pixels válidos suficientes
        if np.sum(valid_pixels_mask) > 10:
            final_prediction = self._denormalize_depth_prediction(
                predicted_tile, dtm_data, valid_pixels_mask
            )
        else:
            # Fallback para tiles 100% vazios (sem referência local)
            # Mantém a predição crua ou tenta pegar média global (melhoria futura)
            pass 

        # Merge e Blending
        merged_tile = dtm_data.copy()
        merged_tile[missing_mask] = final_prediction[missing_mask]
        
        blended_tile = self._blend_prediction_edges(merged_tile, missing_mask)

        dtm_band.WriteArray(blended_tile, x, y)
        return True

    def _create_mask_dataset(self, path, w, h, geo, proj):
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), w, h, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
        ds.SetGeoTransform(geo)
        ds.SetProjection(proj)
        ds.GetRasterBand(1).SetNoDataValue(0)
        return ds

    def _generate_processing_grid(self, w, h):
        return [(x, y) for y in range(0, h, self.processing_tile_size) for x in range(0, w, self.processing_tile_size)]

    def _calculate_context_bounding_box(self, tw, th, x, y, w, h):
        xs = max(0, x - self.context_padding_size)
        ys = max(0, y - self.context_padding_size)
        return {
            "x_start": xs,
            "y_start": ys,
            "width": min(tw, x + w + self.context_padding_size) - xs,
            "height": min(th, y + h + self.context_padding_size) - ys,
        }

    def _crop_tile_from_context_box(self, box, xt, yt, xs, ys, h, w):
        off_x, off_y = xt - xs, yt - ys
        return box[off_y : off_y + h, off_x : off_x + w]

    def _normalize_image(self, img):
        mn, mx = img.min(), img.max()
        norm = (img - mn) / (mx - mn + 1e-8)
        return np.stack([norm] * 3, axis=-1)

    def _denormalize_depth_prediction(self, pred, real, valid_mask):
        mu_r = np.mean(real[valid_mask])
        std_r = np.std(real[valid_mask])
        
        mu_p = np.mean(pred[valid_mask])
        std_p = np.std(pred[valid_mask])

        if std_p < 1e-6:
            scale = 0.0
        else:
            scale = std_r / (std_p + 1e-8)
            
        shift = mu_r - (mu_p * scale)

        out = pred * scale + shift
        
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=mu_r)
            
        return out

    def _blend_prediction_edges(self, dtm, mask, width=5):
        dilated = binary_dilation(mask, iterations=width)
        border_zone = dilated & (~mask)
        
        if not np.any(border_zone):
            return dtm

        blurred = gaussian_filter(dtm, sigma=2.0)
        
        out = dtm.copy()
        out[border_zone] = blurred[border_zone]
        return out
