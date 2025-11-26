import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Tuple
import boto3

from marsfill.fill.dtm_filler import DTMFiller
from marsfill.fill.filler_stats import FillerStats
from marsfill.model.eval import Evaluator
from marsfill.model.train import AvailableModels
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile

logger = Logger()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    clean = uri.replace("s3://", "")
    bucket, key = clean.split("/", 1)
    return bucket, key


def join_path(base: Any, *parts: str) -> Any:
    if isinstance(base, Path):
        for part in parts:
            base = base / part
        return base
    clean_parts = [str(base).rstrip("/")] + [str(p).strip("/") for p in parts]
    return "/".join(clean_parts)


def upload_artifacts_to_s3(local_dir: Path, remote_prefix: str, s3_client: Any) -> None:
    bucket, key_prefix = parse_s3_uri(remote_prefix)
    for file_path in local_dir.iterdir():
        if not file_path.is_file():
            continue
        destination_key = f"{key_prefix.rstrip('/')}/{file_path.name}"
        s3_client.upload_file(str(file_path), bucket, destination_key)
        logger.info(f"☁️  Enviado: s3://{bucket}/{destination_key}")


def main():
    parser = argparse.ArgumentParser(
        prog="Mars DTM Fill", description="CLI para preenchimento de lacunas em DTMs", epilog=""
    )

    parser.add_argument("--test", "-t", required=True, help="Identificador do teste (ex: a, b, c)")
    parser.add_argument(
        "--profile", "-p", default="prod", help="Perfil de configuração [prod|test]"
    )
    parser.add_argument(
        "--mode", "-m", required=True, choices=["local", "s3"], help="Fonte/destino dos dados"
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
    bucket_name = fill_cfg.get("s3_bucket") or profile.get("train", {}).get("s3_bucket")

    test_label = f"test-{args.test.lower()}"

    if args.mode == "s3":
        if not bucket_name:
            raise RuntimeError("Bucket S3 não configurado no perfil.")
        data_root = f"s3://{bucket_name}"
        model_path = join_path(data_root, model_cfg_path)
        dataset_root = join_path(data_root, dataset_prefix)
        output_root = join_path(data_root, output_prefix)
    else:
        base_path = PROJECT_ROOT / local_base_dir
        model_path = base_path / model_cfg_path
        dataset_root = base_path / dataset_prefix
        output_root = base_path / output_prefix

    dtm_path = join_path(dataset_root, "test", test_label, "dtm.IMG")
    ortho_path = join_path(dataset_root, "test", test_label, "ortho.JP2")
    output_dir = join_path(output_root, test_label)

    if isinstance(output_dir, Path):
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
        keep_local_output=(args.mode == "s3"),
    )

    # Métricas e gráficos (sempre calculados localmente)
    local_output_dir = local_filled_path.parent
    stats = FillerStats(output_dir=local_output_dir)

    metrics, gt_arr, filled_arr, eval_mask = stats.calculate_metrics(
        gt_path=local_original_dtm, filled_path=local_filled_path, mask_path=local_mask_path
    )
    stats.plot_results(eval_mask=eval_mask, filled_arr=filled_arr, gt_arr=gt_arr, metrics=metrics)

    if args.mode == "s3":
        remote_output_uri = join_path(output_root, test_label)
        s3_client = boto3.client("s3")
        upload_artifacts_to_s3(local_output_dir, str(remote_output_uri), s3_client)
        shutil.rmtree(local_output_dir, ignore_errors=True)

    logger.info(f"✅ Saída DTM: {filled_uri}")
    logger.info(f"✅ Saída Máscara: {mask_uri}")


if __name__ == "__main__":
    main()
