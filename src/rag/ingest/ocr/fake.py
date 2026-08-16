"""Deterministic fake OCR engine.

This is what the whole test suite runs against. Two modes:

- `FakeOcrEngine.from_fixtures(dir)` replays cached JSON captured from a real Surya
  run, so tests exercise realistic, messy layout output.
- `FakeOcrEngine.from_markup(...)` builds a document from a tiny markup language, so
  a test that needs "a paper with three sections where section 2 is huge" can say so
  in four lines instead of hand-writing block dictionaries.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag.domain import BBox, Block, BlockType, OcrDocument
from rag.errors import OcrError
from rag.ingest.ocr.base import blocks_in_reading_order
from rag.ingest.ocr.cached import load_ocr_document

# "# 3. Experiments" -> section header; "@page" -> page break; anything else -> text.
_HEADER_RE = re.compile(r"^#\s+(.*)$")
_PAGE_RE = re.compile(r"^@page\s*$")
_TYPED_RE = re.compile(r"^\[(\w+)]\s*(.*)$", re.DOTALL)


class FakeOcrEngine:
    """An `OcrEngine` that never loads a model."""

    def __init__(self, documents: dict[str, OcrDocument], *, version: str = "fake-1") -> None:
        self._documents = dict(documents)
        self._version = version
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def version(self) -> str:
        return self._version

    def read(self, pdf_path: Path, doc_id: str) -> OcrDocument:
        self.calls.append(doc_id)
        if doc_id not in self._documents:
            raise OcrError(f"FakeOcrEngine has no document for {doc_id!r}")
        return self._documents[doc_id]

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_fixtures(cls, fixture_dir: Path) -> FakeOcrEngine:
        """Load every `*.json` in a directory as a document, keyed by its doc_id."""
        documents: dict[str, OcrDocument] = {}
        for path in sorted(Path(fixture_dir).glob("*.json")):
            document = load_ocr_document(path)
            documents[document.doc_id] = document
        if not documents:
            raise OcrError(f"no OCR fixtures found in {fixture_dir}")
        return cls(documents)

    @classmethod
    def from_markup(cls, doc_id: str, markup: str) -> FakeOcrEngine:
        """Build one document from compact markup.

            # 1. Introduction        -> a SECTION_HEADER block
            @page                    -> start a new page
            [caption] Figure 1: ...  -> an explicitly typed block
            anything else            -> a TEXT block

        Blank lines separate blocks.
        """
        return cls({doc_id: build_document(doc_id, markup)})


def build_document(
    doc_id: str, markup: str, *, source_path: str = "memory://fake.pdf"
) -> OcrDocument:
    """Parse the markup language into an `OcrDocument`."""
    blocks: list[Block] = []
    page = 1
    order = 0
    y = 0.0

    for raw in markup.strip().split("\n\n"):
        chunk = raw.strip()
        if not chunk:
            continue
        if _PAGE_RE.match(chunk):
            page += 1
            y = 0.0
            continue

        block_type = BlockType.TEXT
        text = chunk

        if match := _HEADER_RE.match(chunk):
            block_type, text = BlockType.SECTION_HEADER, match.group(1).strip()
        elif match := _TYPED_RE.match(chunk):
            label, text = match.group(1).lower(), match.group(2).strip()
            block_type = _LABELS.get(label, BlockType.TEXT)

        height = 20.0 if block_type is BlockType.SECTION_HEADER else 12.0 * (1 + text.count("\n"))
        blocks.append(
            Block(
                type=block_type,
                text=text,
                page=page,
                order=order,
                bbox=BBox(x0=50.0, y0=y, x1=550.0, y1=y + height),
                confidence=0.99,
            )
        )
        order += 1
        y += height + 6.0

    return OcrDocument(
        doc_id=doc_id,
        source_path=source_path,
        page_count=page,
        blocks=blocks_in_reading_order(blocks),
        engine="fake",
        engine_version="fake-1",
    )


_LABELS = {
    "title": BlockType.TITLE,
    "header": BlockType.SECTION_HEADER,
    "text": BlockType.TEXT,
    "list": BlockType.LIST_ITEM,
    "table": BlockType.TABLE,
    "figure": BlockType.FIGURE,
    "caption": BlockType.CAPTION,
    "formula": BlockType.FORMULA,
    "pageheader": BlockType.PAGE_HEADER,
    "pagefooter": BlockType.PAGE_FOOTER,
    "footnote": BlockType.FOOTNOTE,
}
