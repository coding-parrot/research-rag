import pytest

from rag.errors import FetchError, ManifestError
from rag.ingest.fetch import PdfFetcher, StubDownloader, pin_digests, sha256_bytes
from rag.ingest.manifest import Manifest, Paper, load_manifest, save_manifest

VALID_YAML = """\
version: 1
source_repo: https://github.com/example/repo
papers:
  - id: mamba
    title: Mamba
    url: https://arxiv.org/pdf/2312.00752
    topic: ssm
    license: arxiv-nonexclusive
  - id: bert
    title: BERT
    url: https://arxiv.org/pdf/1810.04805
    topic: vectorization
    license: arxiv-nonexclusive
"""

PDF = b"%PDF-1.5 fake body"


@pytest.fixture()
def manifest_path(tmp_path):
    path = tmp_path / "corpus.yaml"
    path.write_text(VALID_YAML)
    return path


class TestManifest:
    def test_load(self, manifest_path):
        manifest = load_manifest(manifest_path)
        assert len(manifest) == 2
        assert manifest.get("mamba").topic == "ssm"

    def test_select_preserves_order_and_validates(self, manifest_path):
        manifest = load_manifest(manifest_path)
        assert [p.id for p in manifest.select(["bert"])] == ["bert"]
        with pytest.raises(ManifestError, match="unknown paper ids"):
            manifest.select(["nope"])

    def test_duplicate_id_rejected(self, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(VALID_YAML.replace("id: bert", "id: mamba"))
        with pytest.raises(ManifestError, match="duplicate"):
            load_manifest(path)

    def test_http_url_rejected(self, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(VALID_YAML.replace("https://arxiv.org/pdf/2312.00752", "http://insecure"))
        with pytest.raises(ManifestError, match="https"):
            load_manifest(path)

    def test_malformed_digest_rejected(self, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(
            VALID_YAML.replace("license: arxiv-nonexclusive", "license: x\n    sha256: nothex", 1)
        )
        with pytest.raises(ManifestError, match="sha256"):
            load_manifest(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            load_manifest(tmp_path / "absent.yaml")

    def test_round_trip(self, manifest_path, tmp_path):
        manifest = load_manifest(manifest_path)
        out = tmp_path / "saved.yaml"
        save_manifest(manifest, out)
        assert load_manifest(out) == manifest


class TestFetcher:
    def _manifest(self, digest=None):
        return Manifest(
            version=1,
            source_repo="",
            papers=(
                Paper(
                    id="mamba",
                    title="Mamba",
                    url="https://x/m.pdf",
                    topic="ssm",
                    license="l",
                    sha256=digest,
                ),
            ),
        )

    def test_downloads_and_reports_digest(self, tmp_path):
        downloader = StubDownloader({"https://x/m.pdf": PDF})
        fetcher = PdfFetcher(tmp_path, downloader)
        result = fetcher.fetch(self._manifest().papers[0])
        assert result.downloaded
        assert result.sha256 == sha256_bytes(PDF)
        assert result.path.read_bytes() == PDF

    def test_cache_hit_skips_download(self, tmp_path):
        downloader = StubDownloader({"https://x/m.pdf": PDF})
        fetcher = PdfFetcher(tmp_path, downloader)
        paper = self._manifest(sha256_bytes(PDF)).papers[0]
        fetcher.fetch(paper)
        second = fetcher.fetch(paper)
        assert not second.downloaded
        assert downloader.calls == ["https://x/m.pdf"]

    def test_pinned_digest_mismatch_raises(self, tmp_path):
        downloader = StubDownloader({"https://x/m.pdf": PDF})
        fetcher = PdfFetcher(tmp_path, downloader)
        paper = self._manifest("0" * 64).papers[0]
        with pytest.raises(FetchError, match="digest mismatch"):
            fetcher.fetch(paper)

    def test_corrupted_cache_redownloads(self, tmp_path):
        downloader = StubDownloader({"https://x/m.pdf": PDF})
        fetcher = PdfFetcher(tmp_path, downloader)
        paper = self._manifest(sha256_bytes(PDF)).papers[0]
        fetcher.fetch(paper)
        (tmp_path / paper.filename).write_bytes(b"%PDF- corrupted")
        result = fetcher.fetch(paper)
        assert result.downloaded
        assert result.path.read_bytes() == PDF

    def test_fetch_all_collects_failures(self, tmp_path):
        manifest = Manifest(
            version=1,
            source_repo="",
            papers=(
                Paper(id="ok", title="t", url="https://x/ok.pdf", topic="t", license="l"),
                Paper(id="bad", title="t", url="https://x/bad.pdf", topic="t", license="l"),
            ),
        )
        downloader = StubDownloader({"https://x/ok.pdf": PDF})
        results, failures = PdfFetcher(tmp_path, downloader).fetch_all(manifest.papers)
        assert [r.paper.id for r in results] == ["ok"]
        assert [pid for pid, _ in failures] == ["bad"]
        assert "bad" in failures[0][1]

    def test_pin_digests_only_fills_blanks(self, tmp_path):
        downloader = StubDownloader({"https://x/m.pdf": PDF})
        fetcher = PdfFetcher(tmp_path, downloader)
        manifest = self._manifest()
        results = [fetcher.fetch(manifest.papers[0])]
        pinned = pin_digests(manifest, results)
        assert pinned.papers[0].sha256 == sha256_bytes(PDF)
