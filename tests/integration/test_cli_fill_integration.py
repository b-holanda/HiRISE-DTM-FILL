import sys
import types
from pathlib import Path

from marsfill.cli import fill as fill_cli


def test_fill_cli_stubbed_local(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset/v1/test/pair-a"
    output_root = tmp_path / "filled/dunes"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dtm_path = dataset_root / "dtm.IMG"
    ortho_path = dataset_root / "ortho.JP2"
    dtm_path.write_text("dtm")
    ortho_path.write_text("ortho")

    def fake_profile(name):
        return {
            "fill": {
                "model_path": "models/marsfill_model.pth",
                "padding_size": 1,
                "tile_size": 2,
                "local_base_dir": str(tmp_path),
            }
        }

    class FakeEvaluator:
        def __init__(self, *a, **k):
            self.args = (a, k)

    class FakeFiller:
        def __init__(self, *a, **k):
            pass

        def fill(self, dtm_path, ortho_path, output_root, keep_local_output=False):
            Path(output_root).mkdir(parents=True, exist_ok=True)
            # Retorna URIs/caminhos simulados
            return (
                str(Path(output_root) / "dtm_filled.tif"),
                str(Path(output_root) / "dtm_filled_mask.tif"),
                Path(output_root) / "dtm_filled.tif",
                Path(output_root) / "dtm_filled_mask.tif",
                Path(dtm_path),
            )

    class FakeStats:
        def __init__(self, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir

        def calculate_metrics(self, gt_path, filled_path, mask_path):
            metrics = {
                "rmse_m": 0.0,
                "mae_m": 0.0,
                "ssim": 0.0,
                "execution_time_s": 0.0,
                "evaluated_pixels": 0,
            }
            return metrics, None, None, None

        def plot_results(self, **kwargs):
            return None
        
        def generate_all_outputs(self, **kwargs):
            return None

    monkeypatch.setattr(fill_cli, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(fill_cli, "DTMFiller", FakeFiller)
    monkeypatch.setattr(fill_cli, "FillerStats", FakeStats)
    monkeypatch.setattr(
        fill_cli,
        "logger",
        types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dtm",
            str(dtm_path),
            "--ortho",
            str(ortho_path),
            "--out_dir",
            str(output_root),
            "--profile",
            "prod",
        ],
    )
    fill_cli.main()

    assert output_root.exists()
