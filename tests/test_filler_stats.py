import numpy as np

from marsfill.fill import filler_stats


def test_filler_stats_with_mock_gdal(monkeypatch, tmp_path):
    data_store = {}

    def write_array(path, array, nodata):
        data_store[str(path)] = (array, nodata)

    gt = np.ones((16, 16), dtype=np.float32)
    filled = np.ones((16, 16), dtype=np.float32)
    filled[0, 0] = 2.0
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[0, 0] = 1.0

    gt_path = tmp_path / "gt.tif"
    filled_path = tmp_path / "filled.tif"
    mask_path = tmp_path / "mask.tif"

    write_array(gt_path, gt, -9999)
    write_array(filled_path, filled, -9999)
    write_array(mask_path, mask, 0)

    class FakeBand:
        def __init__(self, array, nodata):
            self.array = array
            self.nodata = nodata

        def ReadAsArray(self):
            return self.array

        def GetNoDataValue(self):
            return self.nodata

    class FakeDataset:
        def __init__(self, array, nodata):
            self.array = array
            self.nodata = nodata

        def GetRasterBand(self, idx):
            return FakeBand(self.array, self.nodata)

    def fake_open(path):
        array, nodata = data_store[str(path)]
        return FakeDataset(array, nodata)

    monkeypatch.setattr(filler_stats, "gdal", type("G", (), {"Open": staticmethod(fake_open)}))

    stats = filler_stats.FillerStats(output_dir=tmp_path)
    metrics, *_ = stats.calculate_metrics(gt_path, filled_path, mask_path)
    assert "rmse_m" in metrics and metrics["rmse_m"] >= 0
