import sys
import types
from pathlib import Path

from marsfill.cli import fill as fill_cli


def test_fill_cli_stubbed_local(monkeypatch, tmp_path):
    test_id = "a"
    dataset_root = tmp_path / "dataset/v1"
    output_root = tmp_path / "filled"
    dataset_root.mkdir(parents=True)

    def fake_profile(name):
        return {
            "fill": {
                "model_path": "models/marsfill_model.pth",
                "padding_size": 1,
                "tile_size": 2,
                "dataset_prefix": "dataset/v1",
                "output_prefix": "filled",
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
            # Retorna URIs/caminhos simulados
            return (
                str(Path(output_root) / "predicted_dtm.tif"),
                str(Path(output_root) / "mask_predicted_dtm.tif"),
                Path(output_root) / "predicted_dtm.tif",
                Path(output_root) / "mask_predicted_dtm.tif",
                Path(ortho_path),
            )

    class FakeStats:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def calculate_metrics(self, gt_path, filled_path, mask_path):
            return {"rmse": 0.0}, None, None, None

        def plot_results(self, **kwargs):
            return None

    monkeypatch.setattr(fill_cli, "get_profile", fake_profile)
    monkeypatch.setattr(fill_cli, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(fill_cli, "DTMFiller", FakeFiller)
    monkeypatch.setattr(fill_cli, "FillerStats", FakeStats)
    monkeypatch.setattr(fill_cli, "logger", types.SimpleNamespace(info=lambda *a, **k: None))

    monkeypatch.setattr(
        sys, "argv", ["prog", "--test", test_id, "--profile", "prod", "--mode", "local"]
    )
    fill_cli.main()

    expected_output_dir = output_root / f"test-{test_id}"
    assert expected_output_dir.exists()
