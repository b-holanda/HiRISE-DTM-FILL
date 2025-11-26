import sys
import types
from pathlib import Path

from marsfill.cli import dataset as dataset_cli


def test_dataset_cli_stubbed(monkeypatch, tmp_path, capsys):
    called = {}

    def fake_profile(name):
        return {
            "make": {
                "output": str(tmp_path / "dataset/v1/"),
                "samples": 1,
                "tile_size": 4,
                "stride": 2,
                "batch_size": 1,
            }
        }

    class FakeBuilder:
        def __init__(self, **kwargs):
            called.update(kwargs)

        def run(self):
            called["run"] = True

    monkeypatch.setattr(dataset_cli, "DatasetBuilder", FakeBuilder)
    monkeypatch.setattr(dataset_cli, "get_profile", fake_profile)
    monkeypatch.setattr(dataset_cli, "logger", types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))

    monkeypatch.setattr(sys, "argv", ["prog", "--profile", "prod", "--mode", "local"])
    dataset_cli.main()

    assert called.get("run") is True
    # Caminho local deve ser resolvido
    assert Path(called["download_directory"]).as_posix().startswith(tmp_path.as_posix())
