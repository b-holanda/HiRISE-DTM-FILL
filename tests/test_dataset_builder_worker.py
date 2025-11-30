import multiprocessing
import numpy as np
import pyarrow.parquet as pq

from marsfill.dataset.build import DatasetBuilder


def test_producer_consumer_flow(monkeypatch, tmp_path):
    """
    Produtor deve gerar tiles e o consumidor consolidar em parquet no split correto.
    """

    class FakeResp:
        def __init__(self, content: bytes):
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

    # Fake GDAL + write helper
    class FakeBand:
        def __init__(self, data, nodata=-9999):
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
        @staticmethod
        def Open(path, mode=None):
            data = np.ones((4, 4), dtype=np.float32)
            return FakeDataset(data)

        @staticmethod
        def Warp(*args, **kwargs):
            return None

    monkeypatch.setattr("marsfill.dataset.build.gdal", FakeGdalModule)
    # Simplifica escrita de TIF em teste
    monkeypatch.setattr(
        "marsfill.dataset.build._save_tile_as_tif",
        lambda arr, path: path.write_bytes(arr.tobytes()),
    )

    pair = {"dtm_url": "http://example.com/dtm.IMG", "ortho_url": "http://example.com/ortho.JP2", "split": "train"}
    tile_size = 2

    temp_exchange = tmp_path / "exchange"
    temp_exchange.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Manager() as manager:
        queue = manager.Queue()

        # Produtor gera tiles e coloca metadados na fila
        DatasetBuilder.worker_producer(
            pair_data=pair,
            tile_size=tile_size,
            stride_size=2,
            download_directory=tmp_path,
            temp_exchange_dir=temp_exchange,
            queue=queue,
            gdal_cache_max_mb=64,
        )

        # Poison pill para encerrar consumidor
        queue.put(None)

        # Consumidor lê fila e grava parquet
        DatasetBuilder.worker_consumer(
            queue=queue,
            download_directory=tmp_path,
            batch_size=1,
            consumer_id=0,
        )

    # Verifica parquet no split correto
    parquets = list((tmp_path / "train").glob("*.parquet"))
    assert parquets, "Nenhum parquet gerado pelo consumidor"

    # Garante que os buffers gravados sejam float32 planos (tamanho fixo)
    table = pq.read_table(parquets[0])
    df = table.to_pandas()
    assert not df.empty
    row = df.iloc[0]
    expected_bytes = tile_size * tile_size * 4  # float32
    assert len(row["ortho_bytes"]) == expected_bytes
    assert len(row["dtm_bytes"]) == expected_bytes
    assert np.frombuffer(row["dtm_bytes"], dtype=np.float32).shape[0] == tile_size * tile_size

    # Não deve sobrar arquivos temporários de tiles
    assert not any(temp_exchange.rglob("*.tif"))


def test_prepare_assignments_train_validation(monkeypatch, tmp_path):
    """
    Garante distribuição 80/20 entre train/validation e ausência de split de teste.
    """
    monkeypatch.setattr("marsfill.dataset.build._configure_gdal_cache", lambda *a, **k: None)
    builder = DatasetBuilder(
        urls_to_scan=[],
        total_samples=10,
        tile_size=1,
        stride_size=1,
        download_directory=tmp_path,
    )

    builder._prepare_assignments()

    assert builder.assignments.count("train") == 8
    assert builder.assignments.count("validation") == 2
    assert "test" not in builder.assignments
