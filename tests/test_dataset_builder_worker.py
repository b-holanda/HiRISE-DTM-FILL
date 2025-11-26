import types
import numpy as np

from marsfill.dataset.build import DatasetBuilder


def test_worker_process_pair_local(monkeypatch, tmp_path):
    # Fake requests.get returning minimal image bytes
    class FakeResp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_get(url, stream=True, timeout=(15, 300)):
        return FakeResp(b"fake-bytes")

    monkeypatch.setattr("marsfill.dataset.build.requests.get", fake_get)

    # Fake GDAL objects
    class FakeBand:
        def __init__(self, data, nodata=None):
            self.data = data
            self.nodata = nodata

        def ReadAsArray(self, xoff, yoff, xsize, ysize):
            return self.data[yoff : yoff + ysize, xoff : xoff + xsize]

        def GetNoDataValue(self):
            return self.nodata

    class FakeDataset:
        def __init__(self, data):
            self.data = data
            self.RasterXSize = data.shape[1]
            self.RasterYSize = data.shape[0]

        def GetGeoTransform(self):
            return (0, 1, 0, 0, 0, -1)

        def GetProjection(self):
            return "EPSG:4326"

        def GetRasterBand(self, idx):
            return FakeBand(self.data, nodata=-9999)

    class FakeGdalModule:
        GA_Update = 1

        @staticmethod
        def FileFromMemBuffer(path, content):
            return None

        @staticmethod
        def Open(path, mode=None):
            data = np.ones((8, 8), dtype=np.float32)
            return FakeDataset(data)

        @staticmethod
        def Warp(*args, **kwargs):
            return None

        @staticmethod
        def Unlink(path):
            return None

    monkeypatch.setattr("marsfill.dataset.build.gdal", FakeGdalModule)

    pair = {
        "dtm_url": "http://example.com/dtm.IMG",
        "ortho_url": "http://example.com/ortho.JP2",
        "split": "train",
    }
    split, results = DatasetBuilder.worker_process_pair(
        pair_data=pair,
        tile_size=4,
        stride_size=4,
        download_directory=tmp_path,
        s3_bucket_name=None,
        s3_prefix="dataset/v1/",
    )
    assert split == "train"
    # Tiles should be generated because data has no nodata
    assert len(results) > 0
