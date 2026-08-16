"""Surya OCR engine, written against surya-ocr 0.22.x (see the pin in pyproject).

Pipeline per document: rasterise pages with pypdfium2, run layout detection, then
run recognition WITH the layout results. In 0.22.x recognition is layout-aware:
`RecognitionPredictor(images, layout_results=...)` returns one `PageOCRResult` per
page whose `.blocks` each carry the layout label, the reading order, the region
polygon, and the recognised content as HTML. That one integrated call replaces the
box-overlap heuristics older adapters needed to marry layout regions to text lines,
and table blocks arrive with their HTML structure intact, so no separate table
recognition pass is required.

The point of using Surya rather than a plain text extractor is reading order. On a
two-column arXiv paper `page.get_text()` interleaves the columns, which silently
destroys section boundaries before the chunker ever sees them.

Imports are lazy. Nothing in this module is imported unless someone actually runs
ingest, so the test suite never pulls torch into the process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import OcrConfig
from rag.domain import BBox, Block, BlockType, OcrDocument
from rag.errors import OcrError
from rag.ingest.ocr.base import blocks_in_reading_order, map_surya_label
from rag.observability import get_logger, timed

log = get_logger("ocr.surya")


class SuryaOcrEngine:
    """Real OCR. Slow, heavy, and always wrapped in `CachedOcrEngine` in practice."""

    def __init__(self, config: OcrConfig) -> None:
        self._config = config
        self._predictors: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "surya"

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("surya-ocr")
        except Exception:  # pragma: no cover - depends on install layout
            return "unknown"

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load(self) -> dict[str, Any]:
        """Load predictors once per process. Costs tens of seconds and GBs of RAM."""
        if self._predictors is not None:
            return self._predictors
        try:
            from surya.inference import SuryaInferenceManager
            from surya.layout import LayoutPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise OcrError(
                "surya-ocr is not installed. Install the ingest extra: pip install -e '.[ingest]'"
            ) from exc

        with timed(log, "surya.load"):
            manager = SuryaInferenceManager()
            predictors: dict[str, Any] = {
                "manager": manager,
                "layout": LayoutPredictor(manager),
                "recognition": RecognitionPredictor(manager),
            }

        self._predictors = predictors
        return predictors

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def read(self, pdf_path: Path, doc_id: str) -> OcrDocument:
        images = self._rasterise(pdf_path)
        if not images:
            raise OcrError(f"{pdf_path} produced no pages")

        predictors = self._load()
        blocks: list[Block] = []

        with timed(log, "surya.read", doc_id=doc_id, pages=len(images)):
            for start in range(0, len(images), self._config.batch_size):
                batch = images[start : start + self._config.batch_size]
                page_numbers = list(range(start + 1, start + len(batch) + 1))
                blocks.extend(self._read_batch(batch, page_numbers, predictors))

        ordered = blocks_in_reading_order(blocks)
        log.info(
            "ocr complete",
            fields={
                "doc_id": doc_id,
                "pages": len(images),
                "blocks": len(ordered),
                "headers": sum(1 for b in ordered if b.type is BlockType.SECTION_HEADER),
            },
        )
        return OcrDocument(
            doc_id=doc_id,
            source_path=str(pdf_path),
            page_count=len(images),
            blocks=ordered,
            engine=self.name,
            engine_version=self.version,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _rasterise(self, pdf_path: Path) -> list[Any]:
        """PDF pages to PIL images at the configured DPI."""
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise OcrError("pypdfium2 is required for ingest") from exc

        scale = self._config.dpi / 72.0
        # The open call is guarded separately from rendering: a truncated download
        # still starts with %PDF- and passes the fetch magic check, then fails here.
        # Wrapping it into OcrError keeps the one-bad-document isolation contract:
        # the service fails this document and carries on with the rest of the run.
        try:
            doc = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            raise OcrError(f"could not open {pdf_path}: {exc}") from exc
        try:
            return [doc[i].render(scale=scale).to_pil() for i in range(len(doc))]
        except Exception as exc:
            raise OcrError(f"could not rasterise {pdf_path}: {exc}") from exc
        finally:
            doc.close()

    def _read_batch(
        self, images: list[Any], page_numbers: list[int], predictors: dict[str, Any]
    ) -> list[Block]:
        try:
            layout_results = list(predictors["layout"](images))
            recognition_results = list(
                predictors["recognition"](images, layout_results=layout_results)
            )
        except Exception as exc:
            # Surya's call signatures have drifted across releases. An API mismatch
            # must name the installed version and this adapter, not surface as a
            # bare TypeError from deep inside a batch.
            raise OcrError(
                f"surya predictor call failed (surya-ocr {self.version}): {exc}; "
                f"the installed surya API may not match the rag.ingest.ocr.surya adapter"
            ) from exc

        # A predictor that returns fewer (or offset) results than pages would let
        # zip silently drop or misassign trailing pages while the page count in the
        # report still comes from rasterisation. Fail the document loudly instead.
        if not len(images) == len(layout_results) == len(recognition_results):
            raise OcrError(
                f"surya returned mismatched batch results: {len(images)} pages, "
                f"{len(layout_results)} layout, {len(recognition_results)} recognition"
            )

        blocks: list[Block] = []
        for page, recognised in zip(page_numbers, recognition_results, strict=True):
            blocks.extend(self._page_blocks(page, recognised))
        return blocks

    def _page_blocks(self, page: int, recognised: Any) -> list[Block]:
        """One `PageOCRResult` to typed blocks.

        Each `BlockOCRResult` already carries everything we need: the layout label,
        `reading_order`, the region polygon, and the content as HTML. Blocks that
        recognition skipped or failed are dropped individually rather than failing
        the page: `skipped` marks regions recognition chose not to read (figures),
        `error` marks regions it could not read.
        """
        raw_blocks = list(getattr(recognised, "blocks", []) or [])
        blocks: list[Block] = []

        for fallback_order, raw in enumerate(raw_blocks):
            if getattr(raw, "skipped", False) or getattr(raw, "error", None):
                continue
            confidence = float(getattr(raw, "confidence", 1.0) or 1.0)
            if confidence < self._config.min_block_confidence:
                continue

            block_type = map_surya_label(str(getattr(raw, "label", "")))
            html = str(getattr(raw, "html", "") or "")
            text = (
                _table_html_to_pipes(html) if block_type is BlockType.TABLE else _html_to_text(html)
            )
            if not text.strip():
                continue

            order = getattr(raw, "reading_order", None)
            blocks.append(
                Block(
                    type=block_type,
                    text=text,
                    page=page,
                    order=int(order) if order is not None else fallback_order,
                    bbox=_polygon_to_bbox(getattr(raw, "polygon", None)),
                    confidence=confidence,
                )
            )
        return blocks


def _polygon_to_bbox(polygon: Any) -> BBox | None:
    """Surya polygons are [[x, y], ...] corner lists; we keep the enclosing box."""
    if not polygon:
        return None
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


def _html_to_text(html: str) -> str:
    """Flatten a block's HTML content to text, preserving line structure."""
    if not html:
        return ""
    if "<" not in html:
        return html.strip()
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return soup.get_text(separator=" ").strip()


def _table_html_to_pipes(html: str) -> str:
    """Render a table block's HTML as pipe-delimited rows.

    Keeps the table retrievable as text (a table-lookup question must be able to
    match cell contents), while staying readable inside a prompt. Returns "" when
    the HTML has no usable cells so the block is dropped rather than emitting
    separator garbage.
    """
    if not html:
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows: list[str] = []
    for tr in soup.find_all("tr"):
        cells = [cell.get_text(separator=" ").strip() for cell in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    if rows:
        return "\n".join(rows)
    # Table label without table markup: fall back to the flat text.
    return _html_to_text(html)
