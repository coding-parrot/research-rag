"""HTML blog ingestion: extraction, snapshot semantics, kind-aware manifest and health.

The fixture resembles a real Medium post: an article tag holding the content, an
h1 title, three h2 sections (one containing an h3), lists, a table, and the chrome
a saved page drags along (nav, footer, related-posts rail, scripts).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rag.chunking.section import ChunkReport, SectionChunker
from rag.config import ChunkConfig, Config, HeaderConfig
from rag.domain import BlockType, NormalizedDocument
from rag.errors import FetchError, ManifestError
from rag.ingest.fetch import StubDownloader, sha256_bytes
from rag.ingest.headers import DetectionReport, HeaderDetector
from rag.ingest.html import HtmlSnapshotFetcher, extract_html
from rag.ingest.manifest import Manifest, Paper, load_manifest, save_manifest
from rag.ingest.normalize import normalize
from rag.ingest.ocr.fake import FakeOcrEngine
from rag.ingest.service import DocumentReport, IngestService

BLOG_TITLE = "Improving Search Relevance in Hyperlocal Food Delivery"
BLOG_URL = "https://bytes.example.com/improving-search-relevance"

FIXTURE_HTML = f"""\
<html>
<head><title>Example Bytes</title><script>var tracker = init();</script></head>
<body>
<nav class="global-nav"><ul><li>Home</li><li>Engineering</li></ul></nav>
<div id="cookie-consent"><p>We use cookies to improve your browsing experience.</p></div>
<article>
  <h1>{BLOG_TITLE}</h1>
  <p>Search in hyperlocal food delivery is unforgiving. A user typing paneer expects
  the dish, the restaurant that spells it panner, and the combo box it hides inside,
  all ranked within a delivery radius of a few kilometres. This post describes how a
  small language model closed that gap for us in production.</p>
  <h2>The relevance problem</h2>
  <p>Classic lexical retrieval scores tokens, not intent. It cannot tell that a query
  for wrap should surface rolls, or that a misspelled brand still names one specific
  restaurant chain. Embedding models trained on web text miss the regional food
  vocabulary entirely, and both approaches ignore the delivery radius constraint that
  makes hyperlocal search a different problem from web search.</p>
  <ul>
    <li>Lexical match fails on dish synonyms and regional spellings</li>
    <li>Generic embeddings blur brand names into their food category</li>
  </ul>
  <h2>Model architecture</h2>
  <p>We fine-tuned a small decoder model on query and click pairs mined from months of
  search sessions. The model rescores the lexical candidate set rather than replacing
  it, which keeps latency inside the budget and lets the lexical layer keep handling
  exact brand matches, where it is already nearly perfect in our measurements.</p>
  <h3>Why hyperlocal is hard</h3>
  <p>The candidate set changes with every few hundred metres of movement, so scores
  must be comparable across wildly different result pools. We solved this by scoring
  each query and item pair independently instead of normalising across the pool,
  which also made offline evaluation reproducible across cities and times of day.</p>
  <table>
    <tr><th>Model</th><th>NDCG</th></tr>
    <tr><td>BM25 baseline</td><td>0.61</td></tr>
    <tr><td>Small LM reranker</td><td>0.74</td></tr>
  </table>
  <figcaption>Figure 1: Offline relevance metrics.</figcaption>
  <h2>Results and rollout</h2>
  <p>The reranker shipped behind an experiment flag to five percent of traffic first.
  Conversion on search sessions rose measurably while null result rates fell, and the
  latency budget held on commodity CPU inference. We then rolled it out to all cities
  over two weeks, watching the same dashboards the experiment had validated.</p>
  <div class="related-posts"><p>Read next: how we built the discovery feed.</p></div>
</article>
<footer class="site-footer"><p>Copyright Example Bytes 2025.</p></footer>
<script>analytics.track("view");</script>
</body>
</html>
"""

FIXTURE_BYTES = FIXTURE_HTML.encode()

H2_TITLES = ("The relevance problem", "Model architecture", "Results and rollout")
H3_TITLE = "Why hyperlocal is hard"


def _paper(sha256: str | None = None) -> Paper:
    return Paper(
        id="swiggy_search",
        title=BLOG_TITLE,
        url=BLOG_URL,
        topic="case_studies",
        license="publisher-owned; local snapshot for personal research",
        kind="html",
        sha256=sha256,
    )


def _blog_document() -> NormalizedDocument:
    """FIXTURE_HTML through the real extract -> normalize -> detect path."""
    extracted = extract_html(FIXTURE_HTML, "swiggy_search", "memory://blog.html")
    result = normalize(extracted, title=BLOG_TITLE)
    headings, _ = HeaderDetector(HeaderConfig()).detect(result)
    return replace(result.document, headings=headings)


class TestExtraction:
    def test_block_types_in_document_order(self):
        extracted = extract_html(FIXTURE_HTML, "blog", "memory://blog.html")
        assert [b.type for b in extracted.blocks] == [
            BlockType.TITLE,
            BlockType.TEXT,  # intro paragraph
            BlockType.SECTION_HEADER,  # h2: The relevance problem
            BlockType.TEXT,
            BlockType.LIST_ITEM,
            BlockType.LIST_ITEM,
            BlockType.SECTION_HEADER,  # h2: Model architecture
            BlockType.TEXT,
            BlockType.TEXT,  # h3 emitted as TEXT, not SECTION_HEADER
            BlockType.TEXT,
            BlockType.TABLE,
            BlockType.CAPTION,
            BlockType.SECTION_HEADER,  # h2: Results and rollout
            BlockType.TEXT,
        ]
        headers = [b.text for b in extracted.blocks if b.type is BlockType.SECTION_HEADER]
        assert headers == list(H2_TITLES)
        assert extracted.blocks[0].text == BLOG_TITLE
        assert extracted.blocks[8].text == H3_TITLE

    def test_list_table_and_caption_rendering(self):
        extracted = extract_html(FIXTURE_HTML, "blog", "memory://blog.html")
        items = [b.text for b in extracted.blocks if b.type is BlockType.LIST_ITEM]
        assert all(item.startswith("- ") for item in items)
        table = next(b for b in extracted.blocks if b.type is BlockType.TABLE)
        assert table.text.splitlines() == [
            "Model | NDCG",
            "BM25 baseline | 0.61",
            "Small LM reranker | 0.74",
        ]
        caption = next(b for b in extracted.blocks if b.type is BlockType.CAPTION)
        assert caption.text == "Figure 1: Offline relevance metrics."

    def test_chrome_is_stripped(self):
        extracted = extract_html(FIXTURE_HTML, "blog", "memory://blog.html")
        text = "\n".join(b.text for b in extracted.blocks)
        for chrome in ("Home", "cookies", "Read next", "Copyright", "analytics", "var tracker"):
            assert chrome not in text

    def test_engine_identity_and_pagelessness(self):
        from importlib.metadata import version

        extracted = extract_html(FIXTURE_HTML, "blog", "memory://blog.html")
        assert extracted.engine == "html"
        assert extracted.engine_version == version("beautifulsoup4")
        assert extracted.page_count == 1
        assert all(b.page == 1 for b in extracted.blocks)
        assert [b.order for b in extracted.blocks] == list(range(len(extracted.blocks)))


class TestHeadingDepth:
    def test_h3_stays_inside_its_h2_section(self):
        """The mapping decision: default HeaderDetector + SectionChunker must keep
        the h3 inside its h2 chunk instead of splitting the section at it."""
        chunks, report = SectionChunker(ChunkConfig()).chunk(_blog_document())
        titles = {c.section_title for c in chunks}
        assert set(H2_TITLES) <= titles
        assert H3_TITLE not in titles
        architecture = next(c for c in chunks if c.section_title == "Model architecture")
        assert H3_TITLE in architecture.text
        assert report.sections_detected >= 4  # frontmatter + three h2 sections


class TestSnapshotFetcher:
    def test_existing_snapshot_is_used_and_digest_reported(self, tmp_path):
        (tmp_path / "swiggy_search.html").write_bytes(FIXTURE_BYTES)
        downloader = StubDownloader({})
        result = HtmlSnapshotFetcher(tmp_path, downloader).fetch(_paper())
        assert not result.downloaded
        assert result.sha256 == sha256_bytes(FIXTURE_BYTES)
        assert result.newly_pinned  # no pin yet: this digest is what gets pinned
        assert downloader.calls == []

    def test_snapshot_matching_pin_is_used(self, tmp_path):
        (tmp_path / "swiggy_search.html").write_bytes(FIXTURE_BYTES)
        paper = _paper(sha256=sha256_bytes(FIXTURE_BYTES))
        result = HtmlSnapshotFetcher(tmp_path, StubDownloader({})).fetch(paper)
        assert not result.downloaded
        assert not result.newly_pinned

    def test_drifted_snapshot_raises(self, tmp_path):
        (tmp_path / "swiggy_search.html").write_bytes(b"<html>re-saved differently</html>")
        paper = _paper(sha256=sha256_bytes(FIXTURE_BYTES))
        with pytest.raises(FetchError, match="edited or re-saved"):
            HtmlSnapshotFetcher(tmp_path, StubDownloader({})).fetch(paper)

    def test_missing_snapshot_with_blocked_download_names_the_path(self, tmp_path):
        fetcher = HtmlSnapshotFetcher(tmp_path, StubDownloader({}))  # every URL fails
        with pytest.raises(FetchError) as excinfo:
            fetcher.fetch(_paper())
        message = str(excinfo.value)
        assert str(tmp_path / "swiggy_search.html") in message
        assert "browser" in message  # tells the human what to do, not just what broke

    def test_missing_snapshot_downloads_when_publisher_allows(self, tmp_path):
        downloader = StubDownloader({BLOG_URL: FIXTURE_BYTES})
        result = HtmlSnapshotFetcher(tmp_path, downloader).fetch(_paper())
        assert result.downloaded
        assert result.sha256 == sha256_bytes(FIXTURE_BYTES)
        assert result.path.read_bytes() == FIXTURE_BYTES


KIND_YAML = """\
version: 1
source_repo: https://github.com/example/repo
papers:
  - id: meta_test_generation
    title: Automated Unit Test Improvement using Large Language Models at Meta
    arxiv_id: "2402.09171"
    topic: case_studies
    url: https://arxiv.org/pdf/2402.09171
    license: arxiv-nonexclusive
  - id: swiggy_search
    title: Improving Search Relevance
    kind: html
    topic: case_studies
    url: https://bytes.example.com/improving-search-relevance
    license: publisher-owned; local snapshot for personal research
"""


class TestManifestKind:
    def test_kind_parses_and_defaults_to_pdf(self, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(KIND_YAML)
        manifest = load_manifest(path)
        assert manifest.get("meta_test_generation").kind == "pdf"
        assert manifest.get("swiggy_search").kind == "html"

    def test_filename_is_kind_aware(self, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(KIND_YAML)
        manifest = load_manifest(path)
        assert manifest.get("meta_test_generation").filename == "meta_test_generation.pdf"
        assert manifest.get("swiggy_search").filename == "swiggy_search.html"

    def test_save_round_trips_kind(self, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(KIND_YAML)
        manifest = load_manifest(path)
        out = tmp_path / "saved.yaml"
        save_manifest(manifest, out)
        assert load_manifest(out) == manifest
        assert "kind: html" in out.read_text()
        assert "kind: pdf" not in out.read_text()  # the default stays implicit

    def test_unknown_kind_rejected(self, tmp_path):
        path = tmp_path / "corpus.yaml"
        path.write_text(KIND_YAML.replace("kind: html", "kind: docx"))
        with pytest.raises(ManifestError, match="kind"):
            load_manifest(path)


class TestKindAwareHealth:
    def test_two_section_blog_is_healthy_where_a_paper_is_not(self):
        headers = DetectionReport(
            doc_id="d", accepted=2, rejected=0, by_source={}, disagreements=0, used_outline=False
        )
        chunks = ChunkReport(
            doc_id="d",
            sections_detected=3,
            sections_before_merge=3,
            chunks_emitted=3,
            sections_split=0,
            sections_merged=0,
            boundaries_dropped=0,
            max_chunk_chars=500,
            median_chunk_chars=400,
        )
        blog = DocumentReport(doc_id="d", ok=True, kind="html", headers=headers, chunks=chunks)
        paper = DocumentReport(doc_id="d", ok=True, kind="pdf", headers=headers, chunks=chunks)
        assert blog.healthy
        assert not paper.healthy  # pdf semantics unchanged: fewer than 3 sections


class TestEndToEnd:
    def _config(self, tmp_path) -> Config:
        return Config.model_validate(
            {
                "paths": {
                    "data": str(tmp_path / "data"),
                    "corpus_manifest": str(tmp_path / "corpus.yaml"),
                    "evals": str(tmp_path / "evals"),
                },
                "ocr": {"engine": "fake"},
                "embed": {"provider": "fake", "model": "fake", "dimension": 32},
                "index": {"store": "inmemory", "bm25": True},
                "generate": {"provider": "fake"},
            }
        )

    def test_blog_ingests_through_the_service(self, tmp_path):
        config = self._config(tmp_path)
        snapshot_dir = tmp_path / "data" / "html"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "swiggy_search.html").write_bytes(FIXTURE_BYTES)
        manifest_path = tmp_path / "corpus.yaml"
        manifest = Manifest(version=1, source_repo="", papers=(_paper(),))
        save_manifest(manifest, manifest_path)

        # FakeOcrEngine with no documents: the OCR path must never run for html.
        service = IngestService(config, FakeOcrEngine({}))
        chunks, report = service.ingest(manifest, manifest_path=manifest_path)

        (doc,) = report.documents
        assert doc.ok and doc.kind == "html" and doc.healthy

        titles = {c.section_title for c in chunks}
        assert set(H2_TITLES) <= titles
        for chunk in chunks:
            assert chunk.doc_title == BLOG_TITLE
            assert chunk.citation_label.startswith(BLOG_TITLE)
            assert chunk.citation_label.endswith("p.1")  # blogs are pageless

        # The snapshot digest was pinned back into the manifest, kind intact.
        pinned = load_manifest(manifest_path).get("swiggy_search")
        assert pinned.sha256 == sha256_bytes(FIXTURE_BYTES)
        assert pinned.kind == "html"

    def test_missing_snapshot_is_a_failed_document_not_an_abort(self, tmp_path):
        config = self._config(tmp_path)
        manifest_path = tmp_path / "corpus.yaml"
        manifest = Manifest(version=1, source_repo="", papers=(_paper(),))
        save_manifest(manifest, manifest_path)

        service = IngestService(
            config,
            FakeOcrEngine({}),
            html_fetcher=HtmlSnapshotFetcher(tmp_path / "data" / "html", StubDownloader({})),
        )
        chunks, report = service.ingest(manifest, manifest_path=manifest_path)

        assert chunks == []
        (doc,) = report.documents
        assert not doc.ok and doc.kind == "html"
        assert str(tmp_path / "data" / "html" / "swiggy_search.html") in doc.error
        assert doc in report.unhealthy  # what trips the CLI exit code
