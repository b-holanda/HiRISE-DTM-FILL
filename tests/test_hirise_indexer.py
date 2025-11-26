
from marsfill.dataset.hirise_indexer import HirisePDSIndexerDFS


def test_indexer_finds_pair(monkeypatch):
    html_root = """
    <html><body><pre>
    <a href="ESP_012345_1234/">ESP_012345_1234/</a>
    </pre></body></html>
    """
    html_leaf = """
    <html><body><pre>
    <a href="DTEEC_012345_1234_012345_1234_A01.IMG">DTM</a>
    <a href="ESP_012345_1234_RED_A_01_ORTHO.JP2">ORTHO</a>
    </pre></body></html>
    """

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, timeout=10):
        return FakeResponse(html_root if "ESP_012345_1234/" not in url else html_leaf)

    indexer = HirisePDSIndexerDFS(["http://example.com/DTM/ESP/"])
    monkeypatch.setattr(indexer.session, "get", fake_get)

    pairs = indexer.index_pairs(max_pairs=1)
    assert len(pairs) == 1
    assert pairs[0].id == "ESP_012345_1234"
    assert pairs[0].dtm_url.endswith(".IMG")
    assert pairs[0].ortho_url.endswith(".JP2")
