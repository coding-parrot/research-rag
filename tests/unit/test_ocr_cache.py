from pathlib import Path

import pytest

from rag.config import OcrConfig
from rag.domain import BlockType
from rag.errors import OcrError
from rag.ingest.ocr.base import map_surya_label
from rag.ingest.ocr.cached import CachedOcrEngine, load_ocr_document, save_ocr_document
from rag.ingest.ocr.fake import FakeOcrEngine, build_document


@pytest.fixture()
def pdf(tmp_path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.5 fake content")
    return path


class TestLabelMapping:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Section-header", BlockType.SECTION_HEADER),
            ("SectionHeader", BlockType.SECTION_HEADER),
            ("section_header", BlockType.SECTION_HEADER),
            ("Text", BlockType.TEXT),
            ("Page-footer", BlockType.PAGE_FOOTER),
            ("SomeFutureLabel", BlockType.OTHER),
        ],
    )
    def test_mapping(self, label, expected):
        assert map_surya_label(label) is expected


class TestSerialisation:
    def test_round_trip(self, tmp_path):
        document = build_document("d", "# 1. Intro\n\nBody text.\n\n@page\n\nMore body.")
        path = tmp_path / "d.json"
        save_ocr_document(document, path)
        loaded = load_ocr_document(path)
        assert loaded == document

    def test_wrong_format_version_rejected(self, tmp_path):
        document = build_document("d", "Body.")
        path = tmp_path / "d.json"
        save_ocr_document(document, path)
        content = path.read_text().replace('"format": 1', '"format": 99')
        path.write_text(content)
        with pytest.raises(ValueError, match="format"):
            load_ocr_document(path)


class TestCachedEngine:
    def test_second_read_hits_cache(self, tmp_path, pdf):
        inner = FakeOcrEngine.from_markup("paper", "# 1. Intro\n\nBody text here.")
        engine = CachedOcrEngine(inner, tmp_path, OcrConfig(engine="fake"))

        first = engine.read(pdf, "paper")
        second = engine.read(pdf, "paper")
        assert first == second
        assert inner.calls == ["paper"]  # inner engine ran exactly once
        assert engine.hits == 1 and engine.misses == 1

    def test_config_change_invalidates(self, tmp_path, pdf):
        inner = FakeOcrEngine.from_markup("paper", "Body text.")
        engine_a = CachedOcrEngine(inner, tmp_path, OcrConfig(engine="fake", dpi=150))
        engine_a.read(pdf, "paper")
        engine_b = CachedOcrEngine(inner, tmp_path, OcrConfig(engine="fake", dpi=200))
        engine_b.read(pdf, "paper")
        assert inner.calls == ["paper", "paper"]  # dpi change forced a re-run

    def test_pdf_change_invalidates(self, tmp_path, pdf):
        inner = FakeOcrEngine.from_markup("paper", "Body text.")
        config = OcrConfig(engine="fake")
        engine = CachedOcrEngine(inner, tmp_path, config)
        engine.read(pdf, "paper")
        pdf.write_bytes(b"%PDF-1.5 different bytes")
        engine.read(pdf, "paper")
        assert len(inner.calls) == 2

    def test_cache_disabled(self, tmp_path, pdf):
        inner = FakeOcrEngine.from_markup("paper", "Body text.")
        engine = CachedOcrEngine(inner, tmp_path, OcrConfig(engine="fake", cache=False))
        engine.read(pdf, "paper")
        engine.read(pdf, "paper")
        assert len(inner.calls) == 2

    def test_unknown_doc_raises(self, tmp_path, pdf):
        inner = FakeOcrEngine.from_markup("paper", "Body.")
        engine = CachedOcrEngine(inner, tmp_path, OcrConfig(engine="fake"))
        with pytest.raises(OcrError, match="no document"):
            engine.read(pdf, "other")
