from pathlib import Path

import pytest

from marsfill.fill.dtm_filler import DTMFiller


class FakeEvaluator:
    def __init__(self):
        self.calls = []

    def predict_depth(self, orthophoto_image, target_height, target_width):
        self.calls.append((orthophoto_image.shape, target_height, target_width))
        import numpy as np

        return np.zeros((target_height, target_width), dtype=np.float32)


def test_fill_flow_local(monkeypatch, tmp_path):
    # Avoid heavy GDAL by mocking execution/finalization
    filler = DTMFiller(evaluator=FakeEvaluator(), padding_size=1, tile_size=2)

    called = {"execute": False, "finalize": False}

    def fake_execute(self, ortho_path, dtm_path, mask_path):
        called["execute"] = True

    def fake_finalize(self, working_dtm_path, working_mask_path, destination_root, is_s3):
        called["finalize"] = True
        return str(working_dtm_path), str(working_mask_path)

    monkeypatch.setattr(DTMFiller, "_execute_filling_process", fake_execute)
    monkeypatch.setattr(DTMFiller, "_finalize_output", fake_finalize)

    ortho = tmp_path / "o.jp2"
    dtm = tmp_path / "d.img"
    ortho.write_text("ortho")
    dtm.write_text("dtm")

    filled_uri, mask_uri, *_ = filler.fill(
        dtm_path=dtm, ortho_path=ortho, output_root=str(tmp_path)
    )

    assert called["execute"] and called["finalize"]
    assert filled_uri.endswith(".tif") and mask_uri.endswith(".tif")


def test_download_if_needed_s3(monkeypatch, tmp_path):
    filler = DTMFiller(evaluator=FakeEvaluator(), padding_size=1, tile_size=2)

    class FakeS3:
        def __init__(self):
            self.downloads = []

        def download_file(self, bucket, key, dest):
            self.downloads.append((bucket, key, dest))
            Path(dest).write_text("x")

    monkeypatch.setattr(filler, "_parse_s3_uri", lambda uri: ("bucket", "key"))
    monkeypatch.setattr(filler, "_get_s3_client", lambda: FakeS3())

    out = filler._download_if_needed("s3://bucket/key", tmp_path, "file.img")
    assert out.exists()
