from pathlib import Path


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

    monkeypatch.setattr(DTMFiller, "_execute_filling_process", fake_execute)

    ortho = tmp_path / "o.jp2"
    dtm = tmp_path / "d.img"
    ortho.write_text("ortho")
    dtm.write_text("dtm")

    filled_uri, mask_uri, *_ = filler.fill(
        dtm_path=dtm, ortho_path=ortho, output_root=str(tmp_path)
    )

    assert called["execute"]
    assert filled_uri.endswith(".tif") and mask_uri.endswith(".tif")
