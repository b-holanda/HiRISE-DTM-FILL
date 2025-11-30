import os
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, List
from osgeo import gdal
from tqdm import tqdm
from scipy.ndimage import binary_dilation, gaussian_filter

from marsfill.model.eval import Evaluator
from marsfill.utils import Logger

logger = Logger()


class DTMFiller:
    """
    Preenche lacunas (NoData) em DTMs via inferência do modelo de profundidade.

    Suporta leitura/escrita local; assume insumos já presentes no disco.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        padding_size: int,
        tile_size: int,
    ) -> None:
        """
        Args:
            evaluator: Avaliador com o modelo carregado.
            padding_size: Tamanho do padding de contexto em pixels.
            tile_size: Lado do tile processado em pixels.
        """
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
        """Executa o preenchimento e retorna URIs finais e caminhos locais usados."""
        return self.fill_missing_elevation_data(
            orthophoto_file_path=ortho_path,
            digital_terrain_model_path=dtm_path,
            output_destination=output_root,
            keep_local_output=keep_local_output,
        )

    def fill_missing_elevation_data(
        self,
        orthophoto_file_path: str | Path,
        digital_terrain_model_path: str | Path,
        output_destination: str,
        keep_local_output: bool = False,
    ) -> Tuple[str, str, Path, Path, Path]:
        """
        Executa o pipeline de preenchimento e salva no destino local.

        Args:
            orthophoto_file_path: Caminho ou URI da ortofoto.
            digital_terrain_model_path: Caminho ou URI do DTM.
            output_destination: Diretório local para saída.
            keep_local_output: Mantém diretório temporário (não usado no modo local).

        Returns:
            URIs finais (DTM e máscara) e caminhos locais usados no processamento.
        """
        working_directory = Path(output_destination)
        working_directory.mkdir(parents=True, exist_ok=True)

        local_ortho = Path(orthophoto_file_path)
        local_dtm = Path(digital_terrain_model_path)
        original_dtm_path = local_dtm

        working_dtm_path, working_mask_path = self._prepare_working_files(
            local_dtm, working_directory
        )

        try:
            self._execute_filling_process(local_ortho, working_dtm_path, working_mask_path)

            final_dtm_uri, final_mask_uri = self._finalize_output(
                working_dtm_path, working_mask_path, output_destination
            )

            return (
                final_dtm_uri,
                final_mask_uri,
                working_dtm_path,
                working_mask_path,
                original_dtm_path,
            )

        finally:
            if not keep_local_output:
                pass

    def _prepare_working_files(
        self, source_dtm_path: Path, working_directory: Path
    ) -> Tuple[Path, Path]:
        """
        Copia o DTM original para a área de trabalho e define caminhos de saída.

        Args:
            source_dtm_path: Caminho do DTM original.
            working_directory: Diretório temporário usado no processamento.

        Returns:
            Caminhos do DTM preenchido e da máscara dentro da área de trabalho.
        """
        base_filename = Path(source_dtm_path).stem
        working_dtm_path = working_directory / f"{base_filename}_filled.tif"
        working_mask_path = working_directory / f"{base_filename}_filled_mask.tif"

        if not os.path.exists(working_dtm_path):
            logger.info(f"Copiando DTM original para área de trabalho: {working_dtm_path}")
            shutil.copy(source_dtm_path, working_dtm_path)

        return working_dtm_path, working_mask_path

    def _execute_filling_process(
        self, orthophoto_path: Path, dtm_path: Path, mask_path: Path
    ) -> None:
        """
        Abre datasets, percorre tiles e escreve predições e máscara.

        Args:
            orthophoto_path: Caminho da ortofoto.
            dtm_path: Caminho do DTM a ser sobrescrito.
            mask_path: Caminho para salvar a máscara de lacunas.
        """
        dtm_dataset = gdal.Open(str(dtm_path), gdal.GA_Update)
        if not dtm_dataset:
            logger.error("Erro crítico ao abrir DTM.")
            raise FileNotFoundError("Falha ao abrir DTM.")

        # Reamostra a ortofoto para o grid do DTM, evitando offsets fora do raster
        ortho_aligned_path = dtm_path.parent / "aligned_ortho.tif"
        try:
            dtm_gt = dtm_dataset.GetGeoTransform()
            dtm_proj = dtm_dataset.GetProjection()
            width = dtm_dataset.RasterXSize
            height = dtm_dataset.RasterYSize
            bounds = [
                dtm_gt[0],
                dtm_gt[3] + dtm_gt[5] * height,
                dtm_gt[0] + dtm_gt[1] * width,
                dtm_gt[3],
            ]
            gdal.Warp(
                destNameOrDestDS=str(ortho_aligned_path),
                srcDSOrSrcDSTab=str(orthophoto_path),
                format="GTiff",
                outputBounds=bounds,
                xRes=dtm_gt[1],
                yRes=abs(dtm_gt[5]),
                dstSRS=dtm_proj,
                resampleAlg="cubic",
            )
            orthophoto_dataset = gdal.Open(str(ortho_aligned_path), gdal.GA_ReadOnly)
        except Exception as e:
            logger.error(f"Falha ao alinhar ortofoto ao DTM: {e}")
            orthophoto_dataset = gdal.Open(str(orthophoto_path), gdal.GA_ReadOnly)

        if not orthophoto_dataset or not dtm_dataset:
            logger.error("Erro crítico ao abrir arquivos GDAL.")
            raise FileNotFoundError("Falha ao abrir datasets.")

        ortho_band = orthophoto_dataset.GetRasterBand(1)
        dtm_band = dtm_dataset.GetRasterBand(1)

        width = dtm_dataset.RasterXSize
        height = dtm_dataset.RasterYSize

        no_data_val = dtm_band.GetNoDataValue()
        if no_data_val is None:
            no_data_val = -3.4028234663852886e38

        mask_dataset = self._create_mask_dataset(
            mask_path,
            width,
            height,
            dtm_dataset.GetGeoTransform(),
            dtm_dataset.GetProjection(),
        )
        mask_band = mask_dataset.GetRasterBand(1)

        logger.info(f"Iniciando preenchimento em: {dtm_path}")

        tile_coords = self._generate_processing_grid(width, height)

        for x, y in tqdm(tile_coords, desc="Processando Tiles"):
            self._process_single_tile(
                x, y, width, height, ortho_band, dtm_band, mask_band, no_data_val
            )

        dtm_dataset.FlushCache()
        mask_dataset.FlushCache()
        dtm_dataset = None
        mask_dataset = None

    def _process_single_tile(
        self,
        x: int,
        y: int,
        total_w: int,
        total_h: int,
        ortho_band: gdal.Band,
        dtm_band: gdal.Band,
        mask_band: gdal.Band,
        no_data_val: float,
    ) -> None:
        """
        Processa um único tile: leitura, inferência, blending e gravação.

        Args:
            x: Offset horizontal do tile.
            y: Offset vertical do tile.
            total_w: Largura total do raster.
            total_h: Altura total do raster.
            ortho_band: Banda da ortofoto.
            dtm_band: Banda do DTM.
            mask_band: Banda da máscara gerada.
            no_data_val: Valor NoData do DTM.
        """
        w_tile, h_tile = self._calculate_current_tile_dimensions(total_w, total_h, x, y)

        dtm_data = dtm_band.ReadAsArray(x, y, w_tile, h_tile)
        if dtm_data is None:
            return

        missing_mask = (dtm_data == no_data_val) | np.isnan(dtm_data)

        mask_band.WriteArray(missing_mask.astype(np.uint8), x, y)

        if not np.any(missing_mask):
            return

        bbox = self._calculate_context_bounding_box(total_w, total_h, x, y, w_tile, h_tile)

        ortho_crop = ortho_band.ReadAsArray(
            bbox["x_start"], bbox["y_start"], bbox["width"], bbox["height"]
        ).astype(np.float32)

        normalized_ortho = self._normalize_image(ortho_crop)
        predicted_depth_box = self.depth_evaluator.predict_depth(
            orthophoto_image=normalized_ortho,
            target_width=bbox["width"],
            target_height=bbox["height"],
        )

        predicted_tile = self._crop_tile_from_context_box(
            predicted_depth_box, x, y, bbox["x_start"], bbox["y_start"], h_tile, w_tile
        )

        valid_mask = ~missing_mask
        final_prediction = predicted_tile

        if np.sum(valid_mask) > 10:
            final_prediction = self._denormalize_depth_prediction(
                predicted_tile, dtm_data, valid_mask
            )

        merged_tile = dtm_data.copy()
        merged_tile[missing_mask] = final_prediction[missing_mask]

        blended_tile = self._blend_prediction_edges(merged_tile, missing_mask)

        dtm_band.WriteArray(blended_tile, x, y)

    def _finalize_output(
        self, working_dtm_path: Path, working_mask_path: Path, destination_root: str
    ) -> Tuple[str, str]:
        """
        Move os arquivos processados para destino final local.

        Args:
            working_dtm_path: Caminho local do DTM preenchido.
            working_mask_path: Caminho local da máscara.
            destination_root: Diretório de saída.

        Returns:
            URIs finais do DTM e da máscara.
        """
        logger.info(f"Arquivos salvos localmente em: {destination_root}")
        return str(working_dtm_path), str(working_mask_path)

    # --- Métodos Auxiliares de Processamento (Mesmos da versão anterior, apenas mantidos para integridade) ---

    def _create_mask_dataset(
        self, path: Path, w: int, h: int, geo: tuple, proj: str
    ) -> gdal.Dataset:
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), w, h, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
        ds.SetGeoTransform(geo)
        ds.SetProjection(proj)
        ds.GetRasterBand(1).SetNoDataValue(0)
        return ds

    def _generate_processing_grid(self, w: int, h: int) -> List[Tuple[int, int]]:
        return [
            (x, y)
            for y in range(0, h, self.processing_tile_size)
            for x in range(0, w, self.processing_tile_size)
        ]

    def _calculate_current_tile_dimensions(
        self, tw: int, th: int, x: int, y: int
    ) -> Tuple[int, int]:
        return min(self.processing_tile_size, tw - x), min(self.processing_tile_size, th - y)

    def _calculate_context_bounding_box(
        self, tw: int, th: int, x: int, y: int, w: int, h: int
    ) -> dict:
        xs = max(0, x - self.context_padding_size)
        ys = max(0, y - self.context_padding_size)
        return {
            "x_start": xs,
            "y_start": ys,
            "width": min(tw, x + w + self.context_padding_size) - xs,
            "height": min(th, y + h + self.context_padding_size) - ys,
        }

    def _crop_tile_from_context_box(
        self, box: np.ndarray, xt: int, yt: int, xs: int, ys: int, h: int, w: int
    ) -> np.ndarray:
        off_x, off_y = xt - xs, yt - ys
        return box[off_y : off_y + h, off_x : off_x + w]

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        mn, mx = img.min(), img.max()
        norm = (img - mn) / (mx - mn + 1e-8)
        return np.stack([norm] * 3, axis=-1)

    def _denormalize_depth_prediction(
        self, pred: np.ndarray, real: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        mu_r, std_r = np.mean(real[mask]), np.std(real[mask])
        mu_p, std_p = np.mean(pred[mask]), np.std(pred[mask])

        if not np.isfinite(std_p) or std_p < 1e-6:
            scale = 1.0
        else:
            scale = std_r / (std_p + 1e-8)
        shift = mu_r - (mu_p * scale)

        out = pred * scale + shift
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=mu_r, posinf=mu_r, neginf=mu_r)

        # Evita explosões: limita ao intervalo plausível dos dados reais
        clamp_std = std_r if np.isfinite(std_r) and std_r > 1e-6 else 1.0
        lower = mu_r - 5 * clamp_std
        upper = mu_r + 5 * clamp_std
        return np.clip(out, lower, upper)

    def _blend_prediction_edges(
        self, dtm: np.ndarray, mask: np.ndarray, width: int = 5
    ) -> np.ndarray:
        dilated = binary_dilation(mask, iterations=width)
        blurred = gaussian_filter(dtm, sigma=2.0)
        zone = (dilated & ~mask) | (mask & binary_dilation(~mask, iterations=width))
        out = dtm.copy()
        out[zone] = blurred[zone]
        return out
