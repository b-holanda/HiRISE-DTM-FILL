import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from marsfill.model.hirise_dataset import StreamingHiRISeDataset


class FakeProcessor:
    def __call__(self, img, return_tensors="pt"):
        # Return a fixed tensor to avoid external dependencies
        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        return {"pixel_values": tensor}


def test_streaming_dataset_iter(tmp_path):
    tile_size = 2
    ortho = np.ones((tile_size, tile_size), dtype=np.float32)
    dtm = np.zeros((tile_size, tile_size), dtype=np.float32)
    table = pa.Table.from_arrays(
        [
            pa.array([ortho.tobytes()]),
            pa.array([dtm.tobytes()]),
            pa.array(["pair"])
        ],
        names=["ortho_bytes", "dtm_bytes", "pair_id"]
    )
    parquet_path = tmp_path / "sample.parquet"
    pq.write_table(table, parquet_path)

    dataset = StreamingHiRISeDataset(
        parquet_file_paths=[str(parquet_path)],
        image_processor=FakeProcessor(),
        process_rank=0,
        total_process_count=1,
        image_tile_size=tile_size
    )

    items = list(dataset)
    assert len(items) == 1
    x, y = items[0]
    assert x.shape == (3, tile_size, tile_size)
    assert y.shape == (1, tile_size, tile_size)
