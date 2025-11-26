import os
import shutil
import tempfile
import boto3
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Any
from osgeo import gdal
from tqdm import tqdm
from scipy.ndimage import binary_dilation, gaussian_filter

from marsfill.model.eval import Evaluator
from marsfill.utils import Logger

logger = Logger()

class DTMFiller:
    """
    Classe responsável por preencher lacunas (NoData) em arquivos DTM usando inferência de IA.
    Suporta salvamento local ou envio direto para S3 após o processamento.
    """

    def __init__(
        self,
        depth_evaluator: Evaluator,
        context_padding_size: int,
        processing_tile_size: int,
        s3_client: Optional[Any] = None
    ) -> None:
        """
        Inicializa o preenchedor de terreno.

        Parâmetros:
            depth_evaluator (Evaluator): Instância da classe avaliadora contendo o modelo de IA carregado.
            context_padding_size (int): Tamanho da borda extra (padding) para dar contexto ao modelo.
            processing_tile_size (int): Tamanho do bloco central (tile) onde a predição será efetivamente salva.
            s3_client (Optional[Any]): Cliente Boto3 para upload S3. Se None, cria um novo se necessário.
        """
        self.depth_evaluator = depth_evaluator
        self.context_padding_size = context_padding_size
        self.processing_tile_size = processing_tile_size
        self.s3_client = s3_client

    def _get_s3_client(self) -> Any:
        """
        Retorna o cliente S3 existente ou cria um novo.
        """
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        return self.s3_client

    def fill_missing_elevation_data(
        self, 
        orthophoto_file_path: Path, 
        digital_terrain_model_path: Path, 
        output_destination: str
    ) -> Tuple[str, str]:
        """
        Executa o pipeline de preenchimento e salva no destino (Local ou S3).

        Parâmetros:
            orthophoto_file_path (Path): Caminho local para o arquivo GeoTIFF da ortofoto.
            digital_terrain_model_path (Path): Caminho local para o arquivo GeoTIFF do DTM.
            output_destination (str): Caminho do diretório local (ex: 'data/output') ou URI S3 (ex: 's3://bucket/output').

        Retorno:
            Tuple[str, str]: Caminhos finais (ou URIs) do DTM preenchido e da máscara.
        """
        is_s3_output = str(output_destination).startswith("s3://")
        
        # Define onde o trabalho pesado do GDAL vai acontecer (sempre localmente)
        if is_s3_output:
            working_directory = Path(tempfile.mkdtemp())
        else:
            working_directory = Path(output_destination)
            working_directory.mkdir(parents=True, exist_ok=True)

        # 1. Preparação dos Arquivos de Trabalho
        working_dtm_path, working_mask_path = self._prepare_working_files(
            digital_terrain_model_path, working_directory
        )

        # 2. Execução do Processamento (GDAL)
        try:
            self._execute_filling_process(
                orthophoto_file_path, 
                working_dtm_path, 
                working_mask_path
            )
            
            # 3. Finalização (Upload S3 ou Manter Local)
            final_dtm_uri, final_mask_uri = self._finalize_output(
                working_dtm_path, 
                working_mask_path, 
                output_destination, 
                is_s3_output
            )
            
            return final_dtm_uri, final_mask_uri

        finally:
            # Limpeza de arquivos temporários se estivemos usando S3
            if is_s3_output and os.path.exists(working_directory):
                shutil.rmtree(working_directory)

    def _prepare_working_files(self, source_dtm_path: Path, working_directory: Path) -> Tuple[Path, Path]:
        """
        Cria os caminhos de trabalho e copia o DTM original.
        """
        base_filename = os.path.basename(source_dtm_path).split(".")[0].lower()
        working_dtm_path = working_directory / f"predicted_{base_filename}.tif"
        working_mask_path = working_directory / f"mask_{base_filename}.tif"

        if not os.path.exists(working_dtm_path):
            logger.info(f"Copiando DTM original para área de trabalho: {working_dtm_path}")
            shutil.copy(source_dtm_path, working_dtm_path)
            
        return working_dtm_path, working_mask_path

    def _execute_filling_process(self, orthophoto_path: Path, dtm_path: Path, mask_path: Path) -> None:
        """
        Contém a lógica principal de abertura do GDAL e iteração sobre os tiles.
        """
        orthophoto_dataset = gdal.Open(str(orthophoto_path), gdal.GA_ReadOnly)
        dtm_dataset = gdal.Open(str(dtm_path), gdal.GA_Update)

        if not orthophoto_dataset or not dtm_dataset:
            logger.error("Erro crítico ao abrir arquivos GDAL.")
            raise FileNotFoundError("Falha ao abrir datasets.")

        ortho_band = orthophoto_dataset.GetRasterBand(1)
        dtm_band = dtm_dataset.GetRasterBand(1)
        
        width = orthophoto_dataset.RasterXSize
        height = orthophoto_dataset.RasterYSize
        
        no_data_val = dtm_band.GetNoDataValue()
        if no_data_val is None:
            no_data_val = -3.4028234663852886e+38

        # Criação do dataset de máscara
        mask_dataset = self._create_mask_dataset(
            mask_path, width, height, 
            orthophoto_dataset.GetGeoTransform(), 
            orthophoto_dataset.GetProjection()
        )
        mask_band = mask_dataset.GetRasterBand(1)

        logger.info(f"Iniciando preenchimento em: {dtm_path}")
        
        tile_coords = self._generate_processing_grid(width, height)
        
        for x, y in tqdm(tile_coords, desc="Processando Tiles"):
            self._process_single_tile(
                x, y, width, height, 
                ortho_band, dtm_band, mask_band, 
                no_data_val
            )

        dtm_dataset.FlushCache()
        mask_dataset.FlushCache()
        dtm_dataset = None
        mask_dataset = None

    def _process_single_tile(
        self, x: int, y: int, total_w: int, total_h: int, 
        ortho_band: gdal.Band, dtm_band: gdal.Band, mask_band: gdal.Band, 
        no_data_val: float
    ) -> None:
        """
        Processa um único bloco: lê, infere e escreve.
        """
        w_tile, h_tile = self._calculate_current_tile_dimensions(total_w, total_h, x, y)
        
        dtm_data = dtm_band.ReadAsArray(x, y, w_tile, h_tile)
        if dtm_data is None: return

        missing_mask = (dtm_data == no_data_val) | np.isnan(dtm_data)
        
        # Salva máscara de onde era buraco
        mask_band.WriteArray(missing_mask.astype(np.uint8), x, y)

        if not np.any(missing_mask):
            return

        # Calcula contexto (padding)
        bbox = self._calculate_context_bounding_box(total_w, total_h, x, y, w_tile, h_tile)
        
        ortho_crop = ortho_band.ReadAsArray(
            bbox['x_start'], bbox['y_start'], bbox['width'], bbox['height']
        ).astype(np.float32)
        
        # Inferência
        normalized_ortho = self._normalize_image(ortho_crop)
        predicted_depth_box = self.depth_evaluator.predict(
            orthoimage=normalized_ortho, 
            width=bbox['width'], 
            height=bbox['height']
        )
        
        # Recorte do centro (remove padding)
        predicted_tile = self._crop_tile_from_context_box(
            predicted_depth_box, x, y, bbox['x_start'], bbox['y_start'], h_tile, w_tile
        )

        # Desnormalização e Blending
        valid_mask = ~missing_mask
        final_prediction = predicted_tile
        
        if np.sum(valid_mask) > 10:
            final_prediction = self._denormalize_depth_prediction(predicted_tile, dtm_data, valid_mask)

        merged_tile = dtm_data.copy()
        merged_tile[missing_mask] = final_prediction[missing_mask]
        
        blended_tile = self._blend_prediction_edges(merged_tile, missing_mask)
        
        dtm_band.WriteArray(blended_tile, x, y)

    def _finalize_output(
        self, 
        working_dtm_path: Path, 
        working_mask_path: Path, 
        destination_root: str, 
        is_s3: bool
    ) -> Tuple[str, str]:
        """
        Move os arquivos processados para o destino final (Upload S3 ou apenas retorna caminhos locais).
        """
        if not is_s3:
            # Se for local, os arquivos já estão no lugar certo (prepare_working_files usou output_dir)
            logger.info(f"Arquivos salvos localmente em: {destination_root}")
            return str(working_dtm_path), str(working_mask_path)
        
        # Lógica S3
        client = self._get_s3_client()
        bucket, prefix = self._parse_s3_uri(destination_root)
        
        dtm_filename = os.path.basename(working_dtm_path)
        mask_filename = os.path.basename(working_mask_path)
        
        dtm_key = f"{prefix.strip('/')}/{dtm_filename}"
        mask_key = f"{prefix.strip('/')}/{mask_filename}"
        
        logger.info(f"☁️ Iniciando upload para S3: {bucket}/{dtm_key}")
        client.upload_file(str(working_dtm_path), bucket, dtm_key)
        
        logger.info(f"☁️ Iniciando upload da máscara: {bucket}/{mask_key}")
        client.upload_file(str(working_mask_path), bucket, mask_key)
        
        return f"s3://{bucket}/{dtm_key}", f"s3://{bucket}/{mask_key}"

    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        """
        Extrai bucket e prefixo de uma URI S3.
        """
        parts = uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    # --- Métodos Auxiliares de Processamento (Mesmos da versão anterior, apenas mantidos para integridade) ---
    
    def _create_mask_dataset(self, path: Path, w: int, h: int, geo: tuple, proj: str) -> gdal.Dataset:
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(path), w, h, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
        ds.SetGeoTransform(geo)
        ds.SetProjection(proj)
        ds.GetRasterBand(1).SetNoDataValue(0)
        return ds

    def _generate_processing_grid(self, w: int, h: int) -> List[Tuple[int, int]]:
        return [(x, y) for y in range(0, h, self.processing_tile_size) for x in range(0, w, self.processing_tile_size)]

    def _calculate_current_tile_dimensions(self, tw: int, th: int, x: int, y: int) -> Tuple[int, int]:
        return min(self.processing_tile_size, tw - x), min(self.processing_tile_size, th - y)

    def _calculate_context_bounding_box(self, tw: int, th: int, x: int, y: int, w: int, h: int) -> dict:
        xs = max(0, x - self.context_padding_size)
        ys = max(0, y - self.context_padding_size)
        return {
            'x_start': xs, 'y_start': ys,
            'width': min(tw, x + w + self.context_padding_size) - xs,
            'height': min(th, y + h + self.context_padding_size) - ys
        }

    def _crop_tile_from_context_box(self, box: np.ndarray, xt: int, yt: int, xs: int, ys: int, h: int, w: int) -> np.ndarray:
        off_x, off_y = xt - xs, yt - ys
        return box[off_y : off_y + h, off_x : off_x + w]

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        mn, mx = img.min(), img.max()
        norm = (img - mn) / (mx - mn + 1e-8)
        return np.stack([norm]*3, axis=-1)

    def _denormalize_depth_prediction(self, pred: np.ndarray, real: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mu_r, std_r = np.mean(real[mask]), np.std(real[mask])
        mu_p, std_p = np.mean(pred[mask]), np.std(pred[mask])
        return pred * (std_r / (std_p + 1e-8)) + (mu_r - (mu_p * (std_r / (std_p + 1e-8))))

    def _blend_prediction_edges(self, dtm: np.ndarray, mask: np.ndarray, width: int = 5) -> np.ndarray:
        dilated = binary_dilation(mask, iterations=width)
        blurred = gaussian_filter(dtm, sigma=2.0)
        zone = (dilated & ~mask) | (mask & binary_dilation(~mask, iterations=width))
        out = dtm.copy()
        out[zone] = blurred[zone]
        return out
