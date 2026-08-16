"""The OCR seam.

Everything downstream of ingest depends on `OcrEngine`, never on Surya. That is what
lets the entire test suite run without torch, without model weights, and without the
several minutes per paper that real OCR costs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.domain import Block, BlockType, OcrDocument

# Surya's layout label vocabulary mapped onto ours. Labels we do not model become
# OTHER rather than raising, so a Surya upgrade that adds a class does not break ingest.
SURYA_LABEL_MAP: dict[str, BlockType] = {
    "Title": BlockType.TITLE,
    "Section-header": BlockType.SECTION_HEADER,
    "SectionHeader": BlockType.SECTION_HEADER,
    "Text": BlockType.TEXT,
    "Text-inline-math": BlockType.TEXT,
    "List-item": BlockType.LIST_ITEM,
    "ListItem": BlockType.LIST_ITEM,
    "Table": BlockType.TABLE,
    "TableOfContents": BlockType.OTHER,
    "Figure": BlockType.FIGURE,
    "Picture": BlockType.FIGURE,
    "Caption": BlockType.CAPTION,
    "Equation": BlockType.FORMULA,
    "Formula": BlockType.FORMULA,
    "Page-header": BlockType.PAGE_HEADER,
    "PageHeader": BlockType.PAGE_HEADER,
    "Page-footer": BlockType.PAGE_FOOTER,
    "PageFooter": BlockType.PAGE_FOOTER,
    "Footnote": BlockType.FOOTNOTE,
    "Handwriting": BlockType.TEXT,
    "Code": BlockType.TEXT,
    "Form": BlockType.OTHER,
}


def map_surya_label(label: str) -> BlockType:
    """Normalise a Surya layout label, tolerating case and separator drift."""
    if label in SURYA_LABEL_MAP:
        return SURYA_LABEL_MAP[label]
    key = label.replace("_", "-").replace(" ", "-").lower()
    for known, mapped in SURYA_LABEL_MAP.items():
        if known.replace("_", "-").lower() == key:
            return mapped
    return BlockType.OTHER


@runtime_checkable
class OcrEngine(Protocol):
    """Turns a PDF into ordered, typed layout blocks."""

    @property
    def name(self) -> str:
        """Engine identifier, part of the cache key."""
        ...

    @property
    def version(self) -> str:
        """Engine version, part of the cache key. A bump invalidates the cache."""
        ...

    def read(self, pdf_path: Path, doc_id: str) -> OcrDocument:
        """Extract layout blocks in reading order."""
        ...


def blocks_in_reading_order(blocks: list[Block]) -> tuple[Block, ...]:
    """Sort by (page, order) and renumber `order` to be globally contiguous.

    Surya returns per-page reading order. Chunking needs one monotonic sequence
    across the whole document, and every offset downstream depends on it.
    """
    ordered = sorted(blocks, key=lambda b: (b.page, b.order))
    from dataclasses import replace

    return tuple(replace(b, order=i) for i, b in enumerate(ordered))
