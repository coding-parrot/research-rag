"""Regression tests for ingest hardening.

Each test pins a reviewed failure mode: fetch failures vanishing from the ingest
report, unhealthy or subset ingests corrupting the staged chunk store, corrupt
PDFs or Surya API drift aborting a whole run, degenerate table text replacing
recognised content, output-neutral knobs invalidating the OCR cache, digest pins
landing in the wrong manifest file, and CLI errors that point users at the wrong
next command.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from rag.cli import app as cli_app
from rag.config import Config, OcrConfig
from rag.domain import BlockType
from rag.errors import IndexError_, OcrError
from rag.index.base import ChunkStore
from rag.ingest.fetch import PdfFetcher, StubDownloader, sha256_bytes
from rag.ingest.manifest import Manifest, Paper, load_manifest, save_manifest
from rag.ingest.ocr.cached import CachedOcrEngine, save_ocr_document
from rag.ingest.ocr.fake import FakeOcrEngine, build_document
from rag.ingest.ocr.surya import SuryaOcrEngine
from rag.ingest.service import IngestService
from tests.conftest import PAPER_MARKUP

PDF = b"%PDF-1.5 fake body"

# A structureless document: no headers, so it chunks into whole-paper blob parts
# and must come out of ingest flagged unhealthy.
BLOB_MARKUP = "This paper has no detectable section structure at all. " + " ".join(
    f"Sentence {i} rambles on with more prose and never a heading in sight." for i in range(40)
)


def _manifest(*ids: str) -> Manifest:
    return Manifest(
        version=1,
        source_repo="",
        papers=tuple(
            Paper(id=i, title=i.title(), url=f"https://x/{i}.pdf", topic="t", license="l")
            for i in ids
        ),
    )


def _config(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "paths": {
                "data": str(tmp_path / "data"),
                "corpus_manifest": str(tmp_path / "corpus.yaml"),
                "evals": str(tmp_path / "evals"),
            },
            "ocr": {"engine": "fake"},
            "chunk": {"max_chunk_tokens": 128, "part_overlap_tokens": 16, "min_chunk_chars": 80},
            "embed": {"provider": "fake", "model": "fake", "dimension": 32},
            "index": {"store": "inmemory", "bm25": True},
            "generate": {"provider": "fake"},
        }
    )


def _write_cli_env(tmp_path: Path, papers: dict[str, str]) -> Path:
    """A full offline CLI environment: config, manifest, OCR fixtures, local PDFs.

    PDFs are pre-placed so the fetcher cache-hits without a network; the fake OCR
    engine replays the fixtures written here.
    """
    data = tmp_path / "data"
    (data / "pdfs").mkdir(parents=True)
    (data / "ocr").mkdir(parents=True)
    entries = []
    for doc_id, markup in papers.items():
        save_ocr_document(build_document(doc_id, markup), data / "ocr" / f"{doc_id}.json")
        (data / "pdfs" / f"{doc_id}.pdf").write_bytes(b"%PDF-1.5 " + doc_id.encode())
        entries.append(
            {
                "id": doc_id,
                "title": doc_id.title(),
                "url": f"https://example.org/{doc_id}.pdf",
                "topic": "t",
                "license": "l",
            }
        )
    (tmp_path / "corpus.yaml").write_text(
        yaml.safe_dump({"version": 1, "source_repo": "", "papers": entries})
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data": str(data),
                    "corpus_manifest": str(tmp_path / "corpus.yaml"),
                    "evals": str(tmp_path / "evals"),
                },
                "ocr": {"engine": "fake"},
                "chunk": {
                    "max_chunk_tokens": 128,
                    "part_overlap_tokens": 16,
                    "min_chunk_chars": 80,
                },
                "embed": {"provider": "fake", "model": "fake", "dimension": 32},
                "index": {"store": "inmemory", "bm25": True},
                "generate": {"provider": "fake"},
            }
        )
    )
    return config_path


class TestFetchFailuresInReport:
    def test_fetch_failure_becomes_failed_document(self, tmp_path):
        config = _config(tmp_path)
        downloader = StubDownloader({"https://x/good.pdf": PDF})
        fetcher = PdfFetcher(tmp_path / "pdfs", downloader)
        ocr = FakeOcrEngine.from_markup("good", PAPER_MARKUP)
        service = IngestService(config, ocr, fetcher)

        chunks, report = service.ingest(
            _manifest("good", "bad"), manifest_path=tmp_path / "corpus.yaml"
        )

        bad = next(d for d in report.documents if d.doc_id == "bad")
        assert not bad.ok
        assert "bad" in bad.error
        assert bad in report.failed
        assert bad in report.unhealthy  # what trips the CLI exit code
        assert all(c.doc_id != "bad" for c in chunks)


class TestIngestIsolation:
    def test_non_ocr_error_fails_one_document_not_the_run(self, tmp_path):
        class ExplodingOcr:
            """Delegates to the fake, but raises a non-OcrError for one document."""

            name = "fake"
            version = "fake-1"

            def __init__(self, inner: FakeOcrEngine, bad_id: str) -> None:
                self._inner = inner
                self._bad = bad_id

            def read(self, pdf_path: Path, doc_id: str) -> Any:
                if doc_id == self._bad:
                    raise RuntimeError("data format error in corrupt PDF")
                return self._inner.read(pdf_path, doc_id)

        config = _config(tmp_path)
        downloader = StubDownloader(
            {"https://x/good.pdf": PDF, "https://x/corrupt.pdf": b"%PDF-1.5 truncated"}
        )
        fetcher = PdfFetcher(tmp_path / "pdfs", downloader)
        ocr = ExplodingOcr(FakeOcrEngine.from_markup("good", PAPER_MARKUP), "corrupt")
        service = IngestService(config, ocr, fetcher)

        chunks, report = service.ingest(
            _manifest("good", "corrupt"), manifest_path=tmp_path / "corpus.yaml"
        )

        corrupt = next(d for d in report.documents if d.doc_id == "corrupt")
        assert not corrupt.ok
        assert "data format error" in corrupt.error
        good = next(d for d in report.documents if d.doc_id == "good")
        assert good.healthy
        assert {c.doc_id for c in chunks} == {"good"}


class TestManifestPinning:
    def test_pins_write_to_the_loaded_manifest_path(self, tmp_path):
        config = _config(tmp_path)  # config default points at tmp_path/corpus.yaml
        loaded_path = tmp_path / "elsewhere" / "corpus.yaml"
        loaded_path.parent.mkdir()
        save_manifest(_manifest("good"), loaded_path)
        manifest = load_manifest(loaded_path)

        downloader = StubDownloader({"https://x/good.pdf": PDF})
        service = IngestService(
            config,
            FakeOcrEngine.from_markup("good", PAPER_MARKUP),
            PdfFetcher(tmp_path / "pdfs", downloader),
        )
        service.ingest(manifest, manifest_path=loaded_path)

        assert load_manifest(loaded_path).get("good").sha256 == sha256_bytes(PDF)
        assert not (tmp_path / "corpus.yaml").exists()  # config path left alone

    def test_save_manifest_leaves_no_tmp_file(self, tmp_path):
        out = tmp_path / "corpus.yaml"
        save_manifest(_manifest("good"), out)
        assert load_manifest(out) == _manifest("good")
        assert not out.with_suffix(".tmp").exists()


class TestSuryaRasterise:
    def test_corrupt_pdf_open_is_wrapped_into_ocr_error(self, monkeypatch, tmp_path):
        module = types.ModuleType("pypdfium2")

        class PdfiumError(Exception):
            pass

        def raise_on_open(path: str) -> Any:
            raise PdfiumError("data format error")

        module.PdfiumError = PdfiumError  # type: ignore[attr-defined]
        module.PdfDocument = raise_on_open  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdfium2", module)

        engine = SuryaOcrEngine(OcrConfig())
        with pytest.raises(OcrError, match="could not open"):
            engine._rasterise(tmp_path / "corrupt.pdf")


class TestSuryaBatch:
    def test_recognition_receives_layout_results(self):
        """The 0.22 API is layout-aware: recognition must get the layout output."""
        engine = SuryaOcrEngine(OcrConfig())
        seen: dict[str, Any] = {}
        layout_results = [SimpleNamespace(bboxes=[])]

        def recognition(images: list[Any], layout_results: Any = None) -> list[Any]:
            seen["layout_results"] = layout_results
            return [SimpleNamespace(blocks=[]) for _ in images]

        predictors: dict[str, Any] = {
            "manager": object(),
            "layout": lambda images: layout_results,
            "recognition": recognition,
        }
        blocks = engine._read_batch([object()], [1], predictors)
        assert blocks == []
        assert seen["layout_results"] == layout_results

    def test_api_mismatch_wraps_into_ocr_error_naming_the_adapter(self):
        engine = SuryaOcrEngine(OcrConfig())

        def recognition(images: list[Any], **kwargs: Any) -> list[Any]:
            raise TypeError("unexpected keyword argument 'layout_results'")

        predictors: dict[str, Any] = {
            "manager": object(),
            "layout": lambda images: [SimpleNamespace(bboxes=[])],
            "recognition": recognition,
        }
        with pytest.raises(OcrError, match=r"surya-ocr .*rag\.ingest\.ocr\.surya"):
            engine._read_batch([object()], [1], predictors)

    def test_short_recognition_results_fail_the_document_loudly(self):
        engine = SuryaOcrEngine(OcrConfig())
        predictors: dict[str, Any] = {
            "manager": object(),
            "layout": lambda images: [SimpleNamespace(bboxes=[]) for _ in images],
            "recognition": lambda images, layout_results=None: [
                SimpleNamespace(blocks=[]) for _ in images[:-1]
            ],
        }
        with pytest.raises(OcrError, match="4 pages, 4 layout, 3 recognition"):
            engine._read_batch([object()] * 4, [1, 2, 3, 4], predictors)


def _ocr_block(**overrides: Any) -> SimpleNamespace:
    base = dict(
        label="Text",
        html="<p>body text</p>",
        polygon=[[0, 0], [100, 0], [100, 20], [0, 20]],
        confidence=0.9,
        reading_order=0,
        skipped=False,
        error=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestSuryaPageBlocks:
    def test_blocks_map_label_text_and_order(self):
        engine = SuryaOcrEngine(OcrConfig())
        recognised = SimpleNamespace(
            blocks=[
                _ocr_block(label="Section-header", html="<h2>3 Experiments</h2>", reading_order=0),
                _ocr_block(label="Text", html="<p>We evaluate on two tasks.</p>", reading_order=1),
            ]
        )
        blocks = engine._page_blocks(2, recognised)
        assert [b.type for b in blocks] == [BlockType.SECTION_HEADER, BlockType.TEXT]
        assert blocks[0].text == "3 Experiments"
        assert blocks[1].page == 2
        assert blocks[0].bbox is not None and blocks[0].bbox.x1 == 100.0

    def test_skipped_and_errored_blocks_dropped(self):
        engine = SuryaOcrEngine(OcrConfig())
        recognised = SimpleNamespace(
            blocks=[
                _ocr_block(skipped=True),
                _ocr_block(error="unreadable"),
                _ocr_block(html="<p>kept</p>", reading_order=2),
            ]
        )
        blocks = engine._page_blocks(1, recognised)
        assert [b.text for b in blocks] == ["kept"]

    def test_low_confidence_blocks_dropped(self):
        engine = SuryaOcrEngine(OcrConfig(min_block_confidence=0.5))
        recognised = SimpleNamespace(blocks=[_ocr_block(confidence=0.2)])
        assert engine._page_blocks(1, recognised) == []

    def test_table_html_renders_as_pipe_rows(self):
        engine = SuryaOcrEngine(OcrConfig())
        table_html = "<table><tr><th>metric</th><th>ours</th></tr><tr><td>BLEU</td><td>27.4</td></tr></table>"
        recognised = SimpleNamespace(blocks=[_ocr_block(label="Table", html=table_html)])
        blocks = engine._page_blocks(1, recognised)
        assert blocks[0].text == "metric | ours\nBLEU | 27.4"

    def test_table_label_without_table_markup_falls_back_to_text(self):
        engine = SuryaOcrEngine(OcrConfig())
        recognised = SimpleNamespace(
            blocks=[_ocr_block(label="Table", html="<p>caption-ish content</p>")]
        )
        blocks = engine._page_blocks(1, recognised)
        assert blocks[0].text == "caption-ish content"


class TestOcrCacheKey:
    def test_batch_size_change_does_not_invalidate(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(PDF)
        inner = FakeOcrEngine.from_markup("paper", "Body text.")
        CachedOcrEngine(inner, tmp_path, OcrConfig(engine="fake", batch_size=4)).read(pdf, "paper")
        CachedOcrEngine(inner, tmp_path, OcrConfig(engine="fake", batch_size=8)).read(pdf, "paper")
        assert inner.calls == ["paper"]  # second read was a cache hit


class TestChunkStoreHint:
    def test_hint_is_caller_supplied(self, tmp_path):
        with pytest.raises(IndexError_, match="run `rag ingest` first"):
            ChunkStore.load(tmp_path, hint="run `rag ingest` first")

    def test_default_hint_still_points_at_index(self, tmp_path):
        with pytest.raises(IndexError_, match="rag index"):
            ChunkStore.load(tmp_path)


class TestCliIngestStaging:
    def test_unhealthy_documents_are_not_staged(self, tmp_path):
        config_path = _write_cli_env(tmp_path, {"alpha": PAPER_MARKUP, "blob": BLOB_MARKUP})
        result = CliRunner().invoke(cli_app, ["ingest", "-c", str(config_path)])
        assert result.exit_code == 1
        assert "not staged" in result.output
        staged = ChunkStore.load(tmp_path / "data" / "index" / "staged")
        assert len(staged) > 0
        assert {c.doc_id for c in staged.chunks} == {"alpha"}

    def test_partial_ingest_merges_into_staged_store(self, tmp_path):
        config_path = _write_cli_env(tmp_path, {"alpha": PAPER_MARKUP, "beta": PAPER_MARKUP})
        runner = CliRunner()
        full = runner.invoke(cli_app, ["ingest", "-c", str(config_path)])
        assert full.exit_code == 0, full.output
        partial = runner.invoke(cli_app, ["ingest", "-c", str(config_path), "-p", "beta"])
        assert partial.exit_code == 0, partial.output
        staged = ChunkStore.load(tmp_path / "data" / "index" / "staged")
        assert {c.doc_id for c in staged.chunks} == {"alpha", "beta"}


class TestCliIndexHint:
    def test_missing_staged_store_points_at_ingest(self, tmp_path):
        config_path = _write_cli_env(tmp_path, {"alpha": PAPER_MARKUP})
        result = CliRunner().invoke(cli_app, ["index", "-c", str(config_path)])
        assert result.exit_code != 0
        assert "rag ingest" in str(result.exception)


class TestCliHeaders:
    def test_exits_nonzero_when_no_documents_scored(self, tmp_path):
        data = tmp_path / "data"
        (data / "pdfs").mkdir(parents=True)  # deliberately empty: no PDFs fetched
        (tmp_path / "corpus.yaml").write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "source_repo": "",
                    "papers": [
                        {
                            "id": "bert",
                            "title": "BERT",
                            "url": "https://example.org/bert.pdf",
                            "topic": "t",
                            "license": "l",
                        }
                    ],
                }
            )
        )
        labels_path = tmp_path / "labels.yaml"
        labels_path.write_text(
            yaml.safe_dump(
                {"documents": [{"doc_id": "bert", "sections": ["Introduction", "Method"]}]}
            )
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "data": str(data),
                        "corpus_manifest": str(tmp_path / "corpus.yaml"),
                        "evals": str(tmp_path / "evals"),
                    }
                }
            )
        )
        result = CliRunner().invoke(
            cli_app, ["headers", "-c", str(config_path), "--labels", str(labels_path)]
        )
        assert result.exit_code == 1
        assert "no documents were scored" in result.output


class TestCliEvalJudge:
    def test_judge_client_receives_the_secrets_api_key(self, monkeypatch, tmp_path):
        import rag.app as rag_app
        import rag.eval.datasets as datasets_mod
        import rag.eval.judge as judge_mod
        import rag.eval.runner as runner_mod
        import rag.generate.client as client_mod

        captured: dict[str, Any] = {}

        def fake_build_client(provider: str, **kwargs: Any) -> Any:
            captured["provider"] = provider
            captured.update(kwargs)
            return object()

        report = SimpleNamespace(
            run_id="r", config_hash="h", checks=[], aggregates={}, notes=[], passed=True
        )

        class FakeRunner:
            def __init__(self, config: Any, judge: Any = None) -> None:
                captured["judge"] = judge

            def run(self, pipeline: Any, golden: Any, **kwargs: Any) -> Any:
                return report

        monkeypatch.setattr(rag_app, "load_index", lambda config: "bundle")
        monkeypatch.setattr(rag_app, "build_pipeline", lambda config, bundle: "pipeline")
        monkeypatch.setattr(datasets_mod, "load_golden", lambda path: [])
        monkeypatch.setattr(
            judge_mod, "Judge", lambda client, model, effort: SimpleNamespace(client=client)
        )
        monkeypatch.setattr(runner_mod, "EvalRunner", FakeRunner)
        monkeypatch.setattr(runner_mod, "save_report", lambda report, path: None)
        monkeypatch.setattr(client_mod, "build_client", fake_build_client)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.chdir(tmp_path)  # keep any repo .env out of Secrets

        result = CliRunner().invoke(cli_app, ["eval", "--judge"])
        assert result.exit_code == 0, result.output
        assert captured["api_key"] == "sk-test-key"
        assert captured["judge"] is not None
