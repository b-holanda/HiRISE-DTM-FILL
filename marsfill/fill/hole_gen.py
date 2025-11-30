from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal

from marsfill.utils import Logger

logger = Logger()


def generate_holes_in_array(
    data: np.ndarray,
    nodata_value: float,
    num_holes: int = 5,
    min_radius: int = 5,
    max_radius: int = 15,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Cria "buracos" de NoData em um array 2D aplicando discos aleatórios.
    """
    if min_radius > max_radius:
        raise ValueError("min_radius não pode ser maior que max_radius.")
    if data.ndim != 2:
        raise ValueError("O array de entrada deve ser 2D.")

    rng = np.random.default_rng(seed)
    out = data.copy()
    height, width = out.shape

    for _ in range(max(num_holes, 0)):
        radius = rng.integers(min_radius, max_radius + 1)
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)

        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        out[mask] = nodata_value

    return out


def apply_holes_to_raster(
    input_path: Path | str,
    output_path: Path | str,
    num_holes: int = 5,
    min_radius: int = 5,
    max_radius: int = 15,
    nodata_value: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[Path, int]:
    """
    Lê um raster, insere buracos de NoData e salva um novo arquivo GeoTIFF.
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
        nodata_val = -3.4028234663852886e38  # default usado no projeto
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

    logger.info(f"Buracos gerados: {num_holes} | Saída: {output_path}")
    return output_path, num_holes
