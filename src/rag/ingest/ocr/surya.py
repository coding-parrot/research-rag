"""Surya OCR engine.

Pipeline per document: rasterise pages with pypdfium2, run layout detection to get
typed regions and reading order, run text recognition, and optionally run table
recognition over TABLE regions.

The point of using Surya here rather than a plain text extractor is reading order.
On a two-column arXiv paper `page.get_text()` interleaves the columns, which silently
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
            if self._config.detect_tables:
                from surya.table_rec import TableRecPredictor

                predictors["table"] = TableRecPredictor(manager)

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
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            return [doc[i].render(scale=scale).to_pil() for i in range(len(doc))]
        except Exception as exc:
            raise OcrError(f"could not rasterise {pdf_path}: {exc}") from exc
        finally:
            doc.close()

    def _read_batch(
        self, images: list[Any], page_numbers: list[int], predictors: dict[str, Any]
    ) -> list[Block]:
        layout_results = predictors["layout"](images)
        recognition_results = predictors["recognition"](images)

        blocks: list[Block] = []
        for image, page, layout, recognised in zip(
            images, page_numbers, layout_results, recognition_results, strict=False
        ):
            blocks.extend(self._merge_page(page, layout, recognised, image, predictors))
        return blocks

    def _merge_page(
        self,
        page: int,
        layout: Any,
        recognised: Any,
        image: Any,
        predictors: dict[str, Any],
    ) -> list[Block]:
        """Assign recognised text lines to layout regions by box overlap.

        Layout gives us typed regions and their reading order; recognition gives us
        text lines. Neither alone is enough, so we assign each line to the region it
        sits inside and keep the region's reading position.
        """
        lines = list(getattr(recognised, "text_lines", []) or [])
        regions = sorted(
            getattr(layout, "bboxes", []) or [],
            key=lambda r: getattr(r, "position", 0),
        )

        blocks: list[Block] = []
        for order, region in enumerate(regions):
            confidence = float(getattr(region, "confidence", 1.0) or 1.0)
            if confidence < self._config.min_block_confidence:
                continue

            box = _to_bbox(getattr(region, "bbox", None))
            block_type = map_surya_label(str(getattr(region, "label", "")))
            text = _text_in_region(lines, box)

            if block_type is BlockType.TABLE and "table" in predictors:
                text = self._table_text(image, box, predictors) or text

            if not text.strip():
                continue

            blocks.append(
                Block(
                    type=block_type,
                    text=text,
                    page=page,
                    order=order,
                    bbox=box,
                    confidence=confidence,
                )
            )
        return blocks

    def _table_text(self, image: Any, box: BBox | None, predictors: dict[str, Any]) -> str:
        """Render a table region as pipe-delimited rows so it survives chunking as text."""
        if box is None:
            return ""
        try:
            crop = image.crop((box.x0, box.y0, box.x1, box.y1))
            result = predictors["table"]([crop])[0]
            cells = getattr(result, "cells", []) or []
            if not cells:
                return ""
            rows: dict[int, list[tuple[int, str]]] = {}
            for cell in cells:
                row = int(getattr(cell, "row_id", 0))
                col = int(getattr(cell, "col_id", 0))
                rows.setdefault(row, []).append((col, str(getattr(cell, "text", "")).strip()))
            return "\n".join(
                " | ".join(text for _, text in sorted(cols)) for _, cols in sorted(rows.items())
            )
        except Exception as exc:  # a bad table must not fail the whole document
            log.warning("table recognition failed", fields={"error": str(exc)})
            return ""


def _to_bbox(raw: Any) -> BBox | None:
    if raw is None:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _text_in_region(lines: list[Any], box: BBox | None) -> str:
    """Join the recognised lines whose centre falls inside a layout region."""
    if box is None:
        return ""
    inside: list[tuple[float, str]] = []
    for line in lines:
        line_box = _to_bbox(getattr(line, "bbox", None))
        if line_box is None:
            continue
        cx = (line_box.x0 + line_box.x1) / 2
        cy = (line_box.y0 + line_box.y1) / 2
        if box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1:
            inside.append((line_box.y0, str(getattr(line, "text", ""))))
    return "\n".join(text for _, text in sorted(inside)).strip()
