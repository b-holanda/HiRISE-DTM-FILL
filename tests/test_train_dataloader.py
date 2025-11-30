import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from marsfill.model import train as train_module
from marsfill.model.combined_loss import LossWeights


def test_create_dataloaders_reads_parquet(monkeypatch, tmp_path):
    """
    Garante que o trainer monta DataLoaders que leem parquets e decodificam tiles float32.
    """

    tile_size = 2
    ortho = np.ones((tile_size, tile_size), dtype=np.float32)
    dtm = np.zeros((tile_size, tile_size), dtype=np.float32)

    def write_sample_parquet(target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_arrays(
            [
                pa.array([ortho.tobytes()]),
                pa.array([dtm.tobytes()]),
                pa.array(["pair"]),
                pa.array([0]),
                pa.array([0]),
            ],
            names=["ortho_bytes", "dtm_bytes", "pair_id", "tile_x", "tile_y"],
        )
        pq.write_table(table, target_dir / "sample.parquet")

    base_data_dir = tmp_path
    write_sample_parquet(base_data_dir / "dataset/v1/train")
    write_sample_parquet(base_data_dir / "dataset/v1/validation")

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, img, return_tensors="pt"):
            tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
            return {"pixel_values": tensor}

    class FakeCombinedLoss:
        def __init__(self, loss_weights):
            self.weights = loss_weights

        def to(self, device):
            return self

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            batch, _, height, width = x.shape
            return types.SimpleNamespace(predicted_depth=torch.zeros((batch, height, width)))

    monkeypatch.setattr(
        train_module, "DPTImageProcessor", type("StubProcessor", (), {"from_pretrained": FakeProcessor.from_pretrained})
    )
    monkeypatch.setattr(train_module, "CombinedLoss", FakeCombinedLoss)

    fake_logger = types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)

    trainer = train_module.MarsDepthTrainer(
        selected_model_name=train_module.AvailableModels.INTEL_DPT_LARGE,
        batch_size=1,
        learning_rate=1e-4,
        total_epochs=1,
        weight_decay=0.0,
        loss_weights=LossWeights(l1_weight=1.0, gradient_weight=1.0, ssim_weight=1.0),
        dataset_root=str(base_data_dir),
        dataset_prefix="dataset/v1",
        output_prefix="models",
        image_tile_size=tile_size,
        data_loader_workers=0,
        injected_model=FakeModel(),
        logger_instance=fake_logger,
    )

    train_loader, val_loader = trainer.create_dataloaders()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    for batch in (train_batch, val_batch):
        images, depths = batch
        assert images.shape == (1, 3, tile_size, tile_size)
        assert depths.shape == (1, 1, tile_size, tile_size)
