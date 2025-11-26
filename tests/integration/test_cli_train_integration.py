import sys
import types

from marsfill.cli import train as train_cli


def test_train_cli_stubbed(monkeypatch, tmp_path):
    called = {}

    def fake_profile(name):
        return {
            "train": {
                "batch_size": 2,
                "learning_rate": 1e-4,
                "epochs": 1,
                "weight_decay": 0.01,
                "w_l1": 0.5,
                "w_grad": 0.3,
                "w_ssim": 0.2,
                "local_data_dir": str(tmp_path),
            }
        }

    class FakeTrainer:
        def __init__(self, **kwargs):
            called.update(kwargs)

        def run_training_loop(self):
            called["run"] = True

    class FakeTrainingCLI(train_cli.TrainingCLI):
        def __init__(self):
            fake_logger = types.SimpleNamespace(
                info=lambda *a, **k: None, error=lambda *a, **k: None
            )
            super().__init__(
                trainer_class=FakeTrainer, profile_loader=fake_profile, logger_instance=fake_logger
            )

    monkeypatch.setattr(train_cli, "TrainingCLI", FakeTrainingCLI)
    monkeypatch.setattr(
        train_cli,
        "Logger",
        lambda: types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--profile", "prod", "--mode", "local"])
    train_cli.main()

    assert called.get("run") is True
    assert called["storage_mode"] == "local"
