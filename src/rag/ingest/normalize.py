"""Turn OCR blocks into one reading-ordered text with offsets we can trust.

`NormalizedDocument.text` is the single source of truth. Every heading and every
chunk stores offsets into it, so any claim about provenance is checkable by slicing
the string. Nothing downstream ever re-derives text from blocks.

What this stage removes:
  - page furniture (running headers and footers)
  - repeated boilerplate that layout misclassified as body text
  - hyphenation introduced by line wrapping

What it deliberately keeps: tables, captions and formulas. On a research corpus those
carry answers, and dropping them is how a table-lookup question becomes unanswerable.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from rag.domain import Block, BlockType, NormalizedDocument, OcrDocument, PageSpan

BLOCK_SEPARATOR = "\n\n"

# "informa-\ntion" -> "information". Requires lowercase either side so we do not
# join legitimate hyphenated compounds split across lines ("state-\nof-the-art").
_HYPHEN_BREAK = re.compile(r"([a-z])-\s*\n\s*([a-z])")
_SOFT_NEWLINE = re.compile(r"(?<![.!?:;])\n(?!\n)")
_WHITESPACE = re.compile(r"[ \t ]+")  # noqa: RUF001 - NBSP is deliberate
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# Bare page numbers, and "3 of 12" style footers that layout sometimes types as text.
_PAGE_NUMBER_ONLY = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE
)

# A block repeated on at least this fraction of pages is running furniture.
BOILERPLATE_PAGE_RATIO = 0.5
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MAX_CHARS = 200


@dataclass(frozen=True, slots=True)
class BlockSpan:
    """A block plus where its text landed in the normalised document."""

    block: Block
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.block.text


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """The normalised document plus the block spans header detection needs."""

    document: NormalizedDocument
    spans: tuple[BlockSpan, ...]
    dropped_blocks: int = 0

    def spans_of_type(self, block_type: BlockType) -> tuple[BlockSpan, ...]:
        return tuple(s for s in self.spans if s.block.type is block_type)


def normalize(ocr: OcrDocument, *, title: str | None = None) -> NormalizationResult:
    """Build a `NormalizedDocument` from OCR output."""
    kept = _drop_chrome(ocr.blocks)
    kept = _drop_boilerplate(kept, page_count=ocr.page_count)

    parts: list[str] = []
    spans: list[BlockSpan] = []
    page_bounds: dict[int, list[int]] = {}
    cursor = 0

    for block in kept:
        text = clean_text(block.text)
        if not text:
            continue
        if parts:
            parts.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)

        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(BlockSpan(block=block, start=start, end=cursor))

        bounds = page_bounds.setdefault(block.page, [start, cursor])
        bounds[0] = min(bounds[0], start)
        bounds[1] = max(bounds[1], cursor)

    text = "".join(parts)
    document = NormalizedDocument(
        doc_id=ocr.doc_id,
        title=title or _infer_title(kept) or ocr.doc_id,
        text=text,
        headings=(),  # filled in by the header detector
        page_spans=_page_spans(page_bounds, total=len(text)),
        metadata={
            "source_path": ocr.source_path,
            "engine": ocr.engine,
            "engine_version": ocr.engine_version,
            "page_count": str(ocr.page_count),
        },
    )
    return NormalizationResult(
        document=document,
        spans=tuple(spans),
        dropped_blocks=len(ocr.blocks) - len(spans),
    )


def clean_text(raw: str) -> str:
    """Normalise one block's text.

    Order matters: de-hyphenate before collapsing newlines, or the hyphen pattern
    no longer has a newline to match on.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SOFT_NEWLINE.sub(" ", text)  # unwrap soft-wrapped lines, keep real breaks
    text = _WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _drop_chrome(blocks: tuple[Block, ...]) -> list[Block]:
    """Remove page furniture and bare page numbers."""
    return [
        b
        for b in blocks
        if not b.type.is_chrome
        and b.type is not BlockType.FIGURE  # a figure region has no recoverable text
        and not _PAGE_NUMBER_ONLY.match(b.text.strip())
    ]


def _drop_boilerplate(blocks: list[Block], *, page_count: int) -> list[Block]:
    """Remove short blocks repeated across most pages.

    Layout detection reliably tags the running header on a well-formed paper, but on
    a scanned or unusual PDF the venue string ends up as body TEXT on every page.
    Left in, it becomes a high-frequency token that pollutes every embedding.
    """
    if page_count < BOILERPLATE_MIN_PAGES:
        return blocks

    pages_by_text: dict[str, set[int]] = {}
    for block in blocks:
        key = _boilerplate_key(block.text)
        if key and len(block.text) <= BOILERPLATE_MAX_CHARS:
            pages_by_text.setdefault(key, set()).add(block.page)

    threshold = max(BOILERPLATE_MIN_PAGES, int(page_count * BOILERPLATE_PAGE_RATIO))
    repeated = {key for key, pages in pages_by_text.items() if len(pages) >= threshold}
    if not repeated:
        return blocks

    return [
        b
        for b in blocks
        # Never drop a section header, even a repeated one: losing a boundary is
        # far more damaging than keeping one stray line of furniture.
        if b.type is BlockType.SECTION_HEADER or _boilerplate_key(b.text) not in repeated
    ]


def _boilerplate_key(text: str) -> str:
    """Normalise a block for repetition comparison.

    Deliberately does *not* fold digits: "Page 3" style footers are already caught
    by the page-number filter, and digit folding makes genuinely distinct blocks
    that differ only in a number collapse into one key and vanish together.
    """
    return _WHITESPACE.sub(" ", text.strip().lower())


def _infer_title(blocks: list[Block]) -> str | None:
    """First TITLE block, else the first substantial block on page 1."""
    for block in blocks:
        if block.type is BlockType.TITLE and block.text.strip():
            return clean_text(block.text).split("\n")[0][:200]
    for block in blocks:
        if block.page == 1 and len(block.text.strip()) > 20:
            return clean_text(block.text).split("\n")[0][:200]
    return None


def _page_spans(page_bounds: dict[int, list[int]], *, total: int) -> tuple[PageSpan, ...]:
    """Contiguous, non-overlapping [start, end) span per page.

    Blocks from adjacent pages can interleave slightly once reading order is applied,
    so we take each page's first offset as its boundary and let each page run until
    the next one starts. That guarantees `page_at` is total over [0, total).
    """
    if not page_bounds:
        return ()
    ordered = sorted(page_bounds.items())
    spans: list[PageSpan] = []
    for i, (page, (start, end)) in enumerate(ordered):
        next_start = ordered[i + 1][1][0] if i + 1 < len(ordered) else total
        spans.append(PageSpan(page=page, start=start if i else 0, end=max(end, next_start)))
    return tuple(spans)


def page_of_offset(spans: tuple[PageSpan, ...], offset: int) -> int:
    for span in spans:
        if span.start <= offset < span.end:
            return span.page
    return spans[-1].page if spans else 1


def block_type_counts(spans: tuple[BlockSpan, ...]) -> Counter[str]:
    """Diagnostic used by the ingest report."""
    return Counter(span.block.type.value for span in spans)
