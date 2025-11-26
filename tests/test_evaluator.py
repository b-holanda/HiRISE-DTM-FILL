import os
from pathlib import Path

import numpy as np
import torch

from marsfill.model import eval as eval_module


class FakeModel:
    def __init__(self):
        self.loaded_state = None
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict

    def __call__(self, **kwargs):
        class Output:
            def __init__(self):
                self.predicted_depth = torch.ones(1, 2, 2)
        return Output()


class FakeProcessor:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, images, return_tensors="pt"):
        arr = torch.tensor(images).permute(2, 0, 1).unsqueeze(0).float()
        return {"pixel_values": arr}

    @classmethod
    def from_pretrained(cls, name, do_rescale=False):
        return cls()


def test_evaluator_local_weights(tmp_path, monkeypatch):
    weights_path = tmp_path / "model.pth"
    torch.save({"layer": torch.tensor([1, 2, 3])}, weights_path)

    monkeypatch.setattr(eval_module.DPTForDepthEstimation, "from_pretrained", lambda *a, **k: FakeModel())
    monkeypatch.setattr(eval_module.DPTImageProcessor, "from_pretrained", FakeProcessor.from_pretrained)

    evaluator = eval_module.Evaluator(
        pretrained_model_name=eval_module.AvailableModels.INTEL_DPT_LARGE,
        model_path_uri=str(weights_path)
    )
    img = np.ones((2, 2, 3), dtype=np.float32)
    depth = evaluator.predict_depth(img, target_height=2, target_width=2)
    assert depth.shape == (2, 2)


def test_evaluator_s3_download(monkeypatch, tmp_path):
    downloaded = tmp_path / "downloaded.pth"
    torch.save({"x": torch.tensor([0])}, downloaded)

    class FakeS3:
        def __init__(self):
            self.calls = []

        def download_file(self, bucket, key, dest):
            self.calls.append((bucket, key, dest))
            # copy local weights into dest
            torch.save(torch.load(downloaded), dest)

    fake_s3 = FakeS3()
    monkeypatch.setattr(eval_module, "boto3", type("B", (), {"client": lambda *a, **k: fake_s3}))
    monkeypatch.setattr(eval_module.DPTForDepthEstimation, "from_pretrained", lambda *a, **k: FakeModel())
    monkeypatch.setattr(eval_module.DPTImageProcessor, "from_pretrained", FakeProcessor.from_pretrained)

    evaluator = eval_module.Evaluator(
        pretrained_model_name=eval_module.AvailableModels.INTEL_DPT_LARGE,
        model_path_uri="s3://bucket/model.pth"
    )
    assert fake_s3.calls  # download invoked
