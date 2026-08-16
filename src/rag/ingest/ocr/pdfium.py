"""Text-extraction engine on pypdfium2. No models, no downloads, milliseconds per paper.

arXiv PDFs are born-digital, so their embedded text layer is intact and OCR is
unnecessary for a working pipeline. The trade-off against Surya is layout awareness:
two-column reading order can interleave, and there are no typed layout regions. To
keep header detection strong without layout, lines that look like numbered section
headings are promoted to SECTION_HEADER blocks here at extraction time, which feeds
the detector's layout signal; the PDF outline (bookmarks) remains the highest-trust
signal and works unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag.config import OcrConfig
from rag.domain import Block, BlockType, OcrDocument
from rag.errors import OcrError
from rag.ingest.ocr.base import blocks_in_reading_order
from rag.observability import get_logger, timed

log = get_logger("ocr.pdfium")

# A line that reads like a numbered heading: "3 Experiments", "3.1 Setup". Kept in
# sync with the detector's regex in spirit, but deliberately simpler: this only has
# to nominate candidates, the detector still scores and merges them.
_HEADING_LINE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{1,90})\s*$")
_WELL_KNOWN_LINE = re.compile(
    r"^\s*(Abstract|Introduction|Related Work|Background|Method(?:s|ology)?|Approach|"
    r"Experiments?|Results?|Evaluation|Discussion|Limitations?|Conclusions?|References|"
    r"Acknowledge?ments?|Appendix(?:\s+[A-Z])?)\s*$",
    re.IGNORECASE,
)


class PdfiumTextEngine:
    """Extract the embedded text layer, one block per paragraph-ish run."""

    def __init__(self, config: OcrConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "pypdfium"

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("pypdfium2")
        except Exception:  # pragma: no cover - install-dependent
            return "unknown"

    def read(self, pdf_path: Path, doc_id: str) -> OcrDocument:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise OcrError("pypdfium2 is required for ingest") from exc

        try:
            doc = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            raise OcrError(f"could not open {pdf_path}: {exc}") from exc

        blocks: list[Block] = []
        try:
            with timed(log, "pdfium.read", doc_id=doc_id):
                page_count = len(doc)
                for index in range(page_count):
                    text = self._page_text(doc[index])
                    blocks.extend(_page_to_blocks(text, page=index + 1))
        except OcrError:
            raise
        except Exception as exc:
            raise OcrError(f"text extraction failed for {pdf_path}: {exc}") from exc
        finally:
            doc.close()

        ordered = blocks_in_reading_order(blocks)
        log.info(
            "text extraction complete",
            fields={
                "doc_id": doc_id,
                "pages": page_count,
                "blocks": len(ordered),
                "headers": sum(1 for b in ordered if b.type is BlockType.SECTION_HEADER),
            },
        )
        return OcrDocument(
            doc_id=doc_id,
            source_path=str(pdf_path),
            page_count=page_count,
            blocks=ordered,
            engine=self.name,
            engine_version=self.version,
        )

    @staticmethod
    def _page_text(page: Any) -> str:
        textpage = page.get_textpage()
        try:
            return str(textpage.get_text_bounded() or "")
        finally:
            textpage.close()


def _page_to_blocks(text: str, *, page: int) -> list[Block]:
    """Split a page's text into blocks, promoting heading-shaped lines.

    Splitting at heading lines is what preserves section boundaries through
    normalisation: block boundaries become hard paragraph breaks, so a heading can
    never be soft-wrapped into the sentence before it.
    """
    blocks: list[Block] = []
    buffer: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal order
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            blocks.append(Block(type=BlockType.TEXT, text=body, page=page, order=order))
            order += 1

    for line in text.splitlines():
        if _HEADING_LINE.match(line) or _WELL_KNOWN_LINE.match(line):
            flush()
            blocks.append(
                Block(
                    type=BlockType.SECTION_HEADER,
                    text=line.strip(),
                    page=page,
                    order=order,
                    confidence=0.9,
                )
            )
            order += 1
        else:
            buffer.append(line)
    flush()
    return blocks
