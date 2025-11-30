import argparse
from pathlib import Path

from marsfill.utils.profiler import get_profile
from marsfill.fill.hole_gen import apply_holes_to_raster
from marsfill.utils import Logger
from osgeo import gdal

gdal.UseExceptions()

logger = Logger()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Gerador de buracos NoData em DTM.")
    parser.add_argument("--input", "-i", required=True, help="Caminho do DTM de entrada.")
    parser.add_argument("--output", "-o", required=True, help="Caminho de saída (GeoTIFF).")
    parser.add_argument(
        "--profile",
        "-p",
        default="prod",
        help="Perfil de configuração (ex.: prod, test). Default: prod",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    profile = get_profile(args.profile)
    if not profile:
        raise RuntimeError(f"Perfil '{args.profile}' não encontrado.")

    cfg = profile.get("hole_gen", {})
    holes = cfg.get("holes", 5)
    # Suporta tanto min_radius/max_radius quanto min-radius/max-radius
    min_radius = cfg.get("min_radius", cfg.get("min-radius", 8))
    max_radius = cfg.get("max_radius", cfg.get("max-radius", 16))
    seed = cfg.get("seed", 42)
    nodata = cfg.get("nodata", -3.4028234663852886e38)

    apply_holes_to_raster(
        input_path=args.input,
        output_path=args.output,
        num_holes=holes,
        min_radius=min_radius,
        max_radius=max_radius,
        nodata_value=nodata,
        seed=seed,
    )

if __name__ == "__main__":
    main()
