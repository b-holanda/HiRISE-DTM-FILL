import sys
import types
from pathlib import Path


def test_end_to_end_stubbed(monkeypatch, tmp_path):
    """
    Simula uma execução end-to-end usando stubs para dataset -> train -> fill.
    Garante que as CLIs aceitam parâmetros e encadeiam objetos principais.
    """
    # Dataset
    from marsfill.cli import dataset as dataset_cli
    from marsfill.cli import train as train_cli
    from marsfill.cli import fill as fill_cli

    # Stub dataset builder
    monkeypatch.setattr(
        dataset_cli,
        "DatasetBuilder",
        type("DB", (), {"__init__": lambda self, **k: None, "run": lambda self: None}),
    )
    monkeypatch.setattr(
        dataset_cli,
        "get_profile",
        lambda name: {"make": {"output": str(tmp_path / "dataset/v1/")}},
    )
    monkeypatch.setattr(dataset_cli, "logger", types.SimpleNamespace(info=lambda *a, **k: None))
    monkeypatch.setattr(sys, "argv", ["prog", "--profile", "prod"])
    dataset_cli.main()

    # Train
    class FakeTrainer:
        def __init__(self, **kwargs):
            return None

        def run_training_loop(self):
            return None

    class FakeTrainingCLI(train_cli.TrainingCLI):
        def __init__(self):
            fake_logger = types.SimpleNamespace(
                info=lambda *a, **k: None, error=lambda *a, **k: None
            )
            super().__init__(
                trainer_class=FakeTrainer,
                profile_loader=lambda name: {
                    "train": {
                        "local_data_dir": str(tmp_path),
                        "batch_size": 1,
                        "learning_rate": 1e-4,
                    }
                },
                logger_instance=fake_logger,
            )

    monkeypatch.setattr(train_cli, "TrainingCLI", FakeTrainingCLI)
    monkeypatch.setattr(
        train_cli,
        "Logger",
        lambda: types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--profile", "prod", "--mode", "local"])
    train_cli.main()

    # Fill
    test_id = "a"
    monkeypatch.setattr(
        fill_cli,
        "get_profile",
        lambda name: {
            "fill": {
                "model_path": "models/marsfill_model.pth",
                "padding_size": 1,
                "tile_size": 2,
                "dataset_prefix": "dataset/v1",
                "output_prefix": "filled",
                "local_base_dir": str(tmp_path),
            }
        },
    )
    monkeypatch.setattr(fill_cli, "Evaluator", lambda *a, **k: None)
    monkeypatch.setattr(
        fill_cli,
        "DTMFiller",
        type(
            "F",
            (),
            {
                "__init__": lambda self, **k: None,
                "fill": lambda self, **k: (
                    str(Path(tmp_path) / "pred.tif"),
                    str(Path(tmp_path) / "mask.tif"),
                    Path(tmp_path) / "pred.tif",
                    Path(tmp_path) / "mask.tif",
                    Path(tmp_path) / "gt.tif",
                ),
            },
        ),
    )
    monkeypatch.setattr(
        fill_cli,
        "FillerStats",
        lambda output_dir: types.SimpleNamespace(
            calculate_metrics=lambda **kw: ({}, None, None, None),
            plot_results=lambda **kw: None,
        ),
    )
    monkeypatch.setattr(fill_cli, "logger", types.SimpleNamespace(info=lambda *a, **k: None))
    monkeypatch.setattr(
        sys, "argv", ["prog", "--test", test_id, "--profile", "prod", "--mode", "local"]
    )
    fill_cli.main()
