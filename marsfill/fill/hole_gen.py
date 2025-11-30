import argparse
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

    Args:
        data: Matriz 2D com valores do DTM.
        nodata_value: Valor que representa NoData.
        num_holes: Número de buracos a aplicar.
        min_radius: Raio mínimo (em pixels).
        max_radius: Raio máximo (em pixels).
        seed: Semente opcional para reprodutibilidade.

    Returns:
        Novo array com os buracos aplicados.
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

    Args:
        input_path: Caminho do DTM original.
        output_path: Caminho de saída (GeoTIFF).
        num_holes: Quantidade de buracos a aplicar.
        min_radius: Raio mínimo dos buracos.
        max_radius: Raio máximo dos buracos.
        nodata_value: Valor NoData; se None, tenta ler do raster e, se ausente, usa -3.4e38.
        seed: Semente para reprodutibilidade.

    Returns:
        Tupla com (Path do arquivo de saída, buracos aplicados).
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gerador de buracos NoData em DTM.")
    parser.add_argument("--input", "-i", required=True, help="Caminho do DTM de entrada.")
    parser.add_argument("--output", "-o", required=True, help="Caminho de saída (GeoTIFF).")
    parser.add_argument("--holes", type=int, default=5, help="Número de buracos a inserir.")
    parser.add_argument("--min-radius", type=int, default=5, help="Raio mínimo (px).")
    parser.add_argument("--max-radius", type=int, default=15, help="Raio máximo (px).")
    parser.add_argument("--nodata", type=float, default=None, help="Valor NoData a aplicar.")
    parser.add_argument("--seed", type=int, default=None, help="Semente para reprodutibilidade.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    apply_holes_to_raster(
        input_path=args.input,
        output_path=args.output,
        num_holes=args.holes,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        nodata_value=args.nodata,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
