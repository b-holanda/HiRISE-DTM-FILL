from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal
from scipy.ndimage import zoom

from marsfill.utils import Logger

logger = Logger()


def generate_organic_mask(
    shape: Tuple[int, int],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Gera uma máscara binária com forma orgânica (ameboide).
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    scale_factor = 4.0 
    small_h = max(2, int(h / scale_factor))
    small_w = max(2, int(w / scale_factor))
    
    noise = rng.random((small_h, small_w))
    
    zh = h / small_h
    zw = w / small_w
    
    # Interpolação cria formas suaves
    smooth_noise = zoom(noise, (zh, zw), order=3)
    smooth_noise = smooth_noise[:h, :w]
    
    threshold = np.mean(smooth_noise) + (np.std(smooth_noise) * 0.2)
    mask = smooth_noise > threshold
    
    return mask


def generate_holes_in_array(
    data: np.ndarray,
    nodata_value: float,
    num_holes: int = 5,
    min_radius: int = 20,
    max_radius: int = 80,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Cria lacunas orgânicas APENAS dentro da área válida do raster.
    """
    if min_radius > max_radius:
        raise ValueError("min_radius não pode ser maior que max_radius.")
    if data.ndim != 2:
        raise ValueError("O array de entrada deve ser 2D.")

    rng = np.random.default_rng(seed)
    out = data.copy()
    height, width = out.shape
    
    # Detecta onde existem dados válidos (não é NoData e não é NaN)
    # Nota: Usamos tolerância para float ou comparação direta
    if np.abs(nodata_value) > 1e30: # Se for valor muito grande negativo
        valid_mask = (data > -1e30) & (~np.isnan(data))
    else:
        valid_mask = (~np.isclose(data, nodata_value)) & (~np.isnan(data))

    # Encontra as coordenadas (y, x) de todos os pixels válidos
    valid_coords_y, valid_coords_x = np.where(valid_mask)
    
    if len(valid_coords_y) == 0:
        logger.warning("Nenhum pixel válido encontrado para gerar buracos.")
        return out

    logger.info(f"Gerando {num_holes} lacunas orgânicas em área válida...")

    for i in range(max(num_holes, 0)):
        # Escolhe um pixel central ALEATÓRIO que seja VÁLIDO
        idx = rng.integers(0, len(valid_coords_y))
        cy = valid_coords_y[idx]
        cx = valid_coords_x[idx]
        
        current_radius = rng.integers(min_radius, max_radius + 1)
        box_size = current_radius * 2
        
        # Gera máscara local
        seed_iter = (seed + i) if seed is not None else None
        local_mask = generate_organic_mask((box_size, box_size), seed=seed_iter)
        
        # Coordenadas da caixa
        y1 = max(0, cy - current_radius)
        y2 = min(height, cy + current_radius)
        x1 = max(0, cx - current_radius)
        x2 = min(width, cx + current_radius)
        
        # Coordenadas na máscara local
        my1 = max(0, -(cy - current_radius))
        my2 = my1 + (y2 - y1)
        mx1 = max(0, -(cx - current_radius))
        mx2 = mx1 + (x2 - x1)
        
        if (y2 > y1) and (x2 > x1):
            mask_slice = local_mask[my1:my2, mx1:mx2]
            
            # Recorta a região de interesse (ROI) do DTM
            roi = out[y1:y2, x1:x2]
            
            # Recorta também a máscara de validade para esta região
            roi_valid = valid_mask[y1:y2, x1:x2]
            
            # APLICAÇÃO SEGURA:
            # O pixel vira buraco se:
            # 1. A máscara orgânica diz que é buraco (mask_slice)
            # 2. O pixel original ERA válido (roi_valid) - evita expandir a borda preta
            final_hole_mask = mask_slice & roi_valid
            
            roi[final_hole_mask] = nodata_value
            out[y1:y2, x1:x2] = roi

    return out


def apply_holes_to_raster(
    input_path: Path | str,
    output_path: Path | str,
    num_holes: int = 5,
    min_radius: int = 20,
    max_radius: int = 80,
    nodata_value: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[Path, int]:
    """
    Função principal de aplicação de buracos.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    dataset = gdal.Open(str(input_path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Não foi possível abrir {input_path}")

    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    if array is None:
        raise RuntimeError("Falha ao ler banda do DTM.")

    detected_nodata = band.GetNoDataValue()
    nodata_val = nodata_value if nodata_value is not None else detected_nodata
    
    if nodata_val is None:
        # Tenta inferir se não houver metadado, assumindo valor muito baixo
        if np.nanmin(array) < -1e30:
            nodata_val = np.nanmin(array)
        else:
            nodata_val = -3.4028234663852886e38
    
    nodata_val = float(nodata_val)

    holed = generate_holes_in_array(
        data=array,
        nodata_value=nodata_val,
        num_holes=num_holes,
        min_radius=min_radius,
        max_radius=max_radius,
        seed=seed,
    )

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_path),
        dataset.RasterXSize,
        dataset.RasterYSize,
        1,
        band.DataType,
        options=["COMPRESS=LZW"],
    )
    out_ds.SetGeoTransform(dataset.GetGeoTransform())
    out_ds.SetProjection(dataset.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata_val)
    out_band.WriteArray(holed)
    out_band.FlushCache()

    out_band = None
    out_ds = None
    dataset = None

    logger.info(f"Saída salva em: {output_path}")
    return output_path, num_holes
