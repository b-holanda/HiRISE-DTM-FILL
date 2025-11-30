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
        "--pair",
        "-t",
        required=True,
        help="Nome da pasta em data/dataset/v1/test contendo dtm.IMG e ortho.JP2",
    )
    parser.add_argument(
        "--profile", "-p", default="prod", help="Perfil de configuração [prod|test]"
    )

    args = parser.parse_args()

    profile = get_profile(args.profile)
    if not profile:
        raise RuntimeError(f"Perfil '{args.profile}' não encontrado.")

    fill_cfg = profile.get("fill", {})
    model_cfg_path = fill_cfg.get("model_path", "models/marsfill_model.pth")
    dataset_prefix = fill_cfg.get("dataset_prefix", "dataset/v1")
    output_prefix = fill_cfg.get("output_prefix", "filled")
    local_base_dir = fill_cfg.get("local_base_dir", "data")

    base_path = PROJECT_ROOT / local_base_dir
    model_path = base_path / model_cfg_path
    dataset_root = base_path / dataset_prefix / "test" / args.pair
    output_root = base_path / output_prefix / args.pair

    dtm_path = dataset_root / "dtm.IMG"
    ortho_path = dataset_root / "ortho.JP2"
    output_dir = output_root

    if not dtm_path.exists() or not ortho_path.exists():
        raise FileNotFoundError(
            f"Arquivos do par não encontrados em {dataset_root}. Esperados dtm.IMG e ortho.JP2."
        )

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

    metrics, gt_arr, filled_arr, eval_mask = stats.calculate_metrics(
        gt_path=local_original_dtm, filled_path=local_filled_path, mask_path=local_mask_path
    )
    stats.plot_results(eval_mask=eval_mask, filled_arr=filled_arr, gt_arr=gt_arr, metrics=metrics)

    logger.info(f"✅ Saída DTM: {filled_uri}")
    logger.info(f"✅ Saída Máscara: {mask_uri}")


if __name__ == "__main__":
    main()
