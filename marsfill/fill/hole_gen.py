from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter, zoom

from marsfill.utils import Logger

logger = Logger()


def generate_organic_mask(
    shape: Tuple[int, int],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Gera uma máscara binária com forma orgânica (ameboide) usando
    ruído gaussiano limiarizado.
    
    A lógica é: gerar ruído de baixa resolução -> dar zoom -> aplicar blur -> cortar.
    Isso cria bordas suaves e irregulares.
    """
    rng = np.random.default_rng(seed)
    
    # Gera ruído em resolução menor para criar formas maiores após o zoom
    h, w = shape
    # Fator de redução (quanto maior, mais "grosseiro" e suave o buraco)
    scale_factor = 4.0 
    small_h = max(2, int(h / scale_factor))
    small_w = max(2, int(w / scale_factor))
    
    noise = rng.random((small_h, small_w))
    
    # Redimensiona para o tamanho original (interpolação bicúbica gera suavidade)
    # zoom factor
    zh = h / small_h
    zw = w / small_w
    
    # O zoom já suaviza, mas o gaussian filter garante bordas mais naturais
    smooth_noise = zoom(noise, (zh, zw), order=3)
    
    # Garante que o tamanho bate exato com o shape pedido (pode variar por arredondamento)
    smooth_noise = smooth_noise[:h, :w]
    
    # Threshold: define o "recorte" da mancha. 
    # Valores > 0.5 viram buraco. Variações aqui mudam a "gordura" do buraco.
    threshold = np.mean(smooth_noise) + (np.std(smooth_noise) * 0.2)
    mask = smooth_noise > threshold
    
    return mask


def generate_holes_in_array(
    data: np.ndarray,
    nodata_value: float,
    num_holes: int = 5,
    min_radius: int = 20,  # Aumentei o default, buracos orgânicos precisam de espaço
    max_radius: int = 80,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Cria "buracos" de NoData orgânicos e realistas simulando falhas de fotogrametria.
    """
    if min_radius > max_radius:
        raise ValueError("min_radius não pode ser maior que max_radius.")
    if data.ndim != 2:
        raise ValueError("O array de entrada deve ser 2D.")

    rng = np.random.default_rng(seed)
    out = data.copy()
    height, width = out.shape

    # Se a imagem for muito pequena para o raio, ajusta
    max_radius = min(max_radius, min(height, width) // 2)

    logger.info(f"Gerando {num_holes} lacunas orgânicas...")

    for i in range(max(num_holes, 0)):
        # Define um tamanho aleatório para a "caixa" onde o buraco vai crescer
        current_radius = rng.integers(min_radius, max_radius + 1)
        box_size = current_radius * 2
        
        # Gera uma máscara orgânica local
        seed_iter = (seed + i) if seed is not None else None
        local_mask = generate_organic_mask((box_size, box_size), seed=seed_iter)
        
        # Escolhe uma posição aleatória
        # Permitimos que saia um pouco da borda para simular buracos cortados
        cy = rng.integers(0, height)
        cx = rng.integers(0, width)
        
        # Coordenadas da caixa no array principal (com clipping para não estourar array)
        y1 = max(0, cy - current_radius)
        y2 = min(height, cy + current_radius)
        x1 = max(0, cx - current_radius)
        x2 = min(width, cx + current_radius)
        
        # Coordenadas correspondentes dentro da máscara local
        my1 = max(0, -(cy - current_radius))
        my2 = my1 + (y2 - y1)
        mx1 = max(0, -(cx - current_radius))
        mx2 = mx1 + (x2 - x1)
        
        # Verifica se as fatias são válidas (tamanho > 0)
        if (y2 > y1) and (x2 > x1):
            mask_slice = local_mask[my1:my2, mx1:mx2]
            
            # Aplica o NoData onde a máscara for True
            roi = out[y1:y2, x1:x2]
            roi[mask_slice] = nodata_value
            out[y1:y2, x1:x2] = roi

    return out


def apply_holes_to_raster(
    input_path: Path | str,
    output_path: Path | str,
    num_holes: int = 5,
    min_radius: int = 20,
    max_radius: int = 60,
    nodata_value: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[Path, int]:
    """
    Lê um raster, insere buracos orgânicos de NoData e salva.
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
    
    # Fallback seguro se não houver metadado
    if nodata_val is None:
        nodata_val = -3.4028234663852886e38
    
    nodata_val = float(nodata_val)

    # Gera os buracos
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

    # Limpeza
    out_band = None
    out_ds = None
    dataset = None

    logger.info(f"Buracos orgânicos gerados: {num_holes} | Saída: {output_path}")
    return output_path, num_holes
