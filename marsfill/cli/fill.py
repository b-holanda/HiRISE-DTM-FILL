import argparse
from pathlib import Path

from marsfill.fill.dtm_filler import DTMFiller
from marsfill.fill.filler_stats import FillerStats
from marsfill.model.eval import Evaluator
from marsfill.model.train import AvailableModels
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile

logger = Logger()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    """Ponto de entrada da CLI de preenchimento de DTMs."""
    parser = argparse.ArgumentParser(
        prog="Mars DTM Fill", description="CLI para preenchimento de lacunas em DTMs", epilog=""
    )

    parser.add_argument(
        "--dtm",
        "-d",
        required=True,
        help="Caminho do DTM (com buracos) a ser preenchido.",
    )
    parser.add_argument(
        "--ortho",
        "-o",
        required=True,
        help="Caminho da ortofoto correspondente.",
    )
    parser.add_argument(
        "--out_dir",
        "-O",
        required=True,
        help="Diretório de saída onde serão gravados DTM preenchido, máscara e métricas.",
    )
    parser.add_argument(
        "--profile", "-p", default="prod", help="Perfil de configuração [prod|test]"
    )
    parser.add_argument(
        "--gt",
        help="Caminho opcional para o DTM original (sem buracos) usado para métricas. "
        "Se omitido, usa o mesmo DTM de entrada.",
    )

    args = parser.parse_args()

    profile = get_profile(args.profile)
    if not profile:
        raise RuntimeError(f"Perfil '{args.profile}' não encontrado.")

    fill_cfg = profile.get("fill", {})
    model_cfg_path = fill_cfg.get("model_path", "models/marsfill_model.pth")
    local_base_dir = fill_cfg.get("local_base_dir", "data")

    base_path = PROJECT_ROOT / local_base_dir
    model_path = base_path / model_cfg_path

    dtm_path = Path(args.dtm)
    ortho_path = Path(args.ortho)
    output_dir = Path(args.out_dir)

    if not dtm_path.exists() or not ortho_path.exists():
        raise FileNotFoundError("DTM ou ortho não encontrados para preenchimento.")

    output_dir.mkdir(parents=True, exist_ok=True)

    padding_size = int(fill_cfg.get("padding_size", 128))
    tile_size = int(fill_cfg.get("tile_size", 512))

    evaluator = Evaluator(
        pretrained_model_name=AvailableModels.INTEL_DPT_LARGE, model_path_uri=str(model_path)
    )
    filler = DTMFiller(evaluator=evaluator, padding_size=padding_size, tile_size=tile_size)

    filled_uri, mask_uri, local_filled_path, local_mask_path, local_original_dtm = filler.fill(
        dtm_path=dtm_path,
        ortho_path=ortho_path,
        output_root=str(output_dir),
        keep_local_output=False,
    )

    # Métricas e gráficos (sempre calculados localmente)
    local_output_dir = local_filled_path.parent
    stats = FillerStats(output_dir=local_output_dir)

    gt_path = Path(args.gt) if args.gt else local_original_dtm
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth '{gt_path}' não encontrado para cálculo de métricas.")

    metrics, gt_arr, filled_arr, eval_mask = stats.calculate_metrics(
        gt_path=gt_path, filled_path=local_filled_path, mask_path=local_mask_path
    )
    stats.plot_results(eval_mask=eval_mask, filled_arr=filled_arr, gt_arr=gt_arr, metrics=metrics)

    logger.info(f"✅ Saída DTM: {filled_uri}")
    logger.info(f"✅ Saída Máscara: {mask_uri}")


if __name__ == "__main__":
    main()
