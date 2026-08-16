"""Core domain types.

Everything here is frozen and dependency-free. No I/O, no models, no third-party
imports beyond the standard library. That keeps the type layer importable in any
test without pulling in torch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

# --------------------------------------------------------------------------- #
# OCR layer
# --------------------------------------------------------------------------- #


class BlockType(StrEnum):
    """Layout region classes we care about.

    Mirrors Surya's layout labels, collapsed to the set that changes our behaviour.
    Anything Surya emits that we do not model maps to OTHER.
    """

    TITLE = "title"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FORMULA = "formula"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    FOOTNOTE = "footnote"
    OTHER = "other"

    @property
    def is_body(self) -> bool:
        """True if this block contributes to the readable body of the document."""
        return self in _BODY_BLOCKS

    @property
    def is_chrome(self) -> bool:
        """True if this block is running page furniture we strip before chunking."""
        return self in _CHROME_BLOCKS


_BODY_BLOCKS = frozenset(
    {
        BlockType.TITLE,
        BlockType.SECTION_HEADER,
        BlockType.TEXT,
        BlockType.LIST_ITEM,
        BlockType.TABLE,
        BlockType.CAPTION,
        BlockType.FORMULA,
    }
)

_CHROME_BLOCKS = frozenset(
    {
        BlockType.PAGE_HEADER,
        BlockType.PAGE_FOOTER,
    }
)


@dataclass(frozen=True, slots=True)
class BBox:
    """Bounding box in PDF page coordinates, origin top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(frozen=True, slots=True)
class Block:
    """One layout region with its recognised text, in document reading order."""

    type: BlockType
    text: str
    page: int  # 1-indexed, matches what a human reads off the PDF
    order: int  # global reading order across the whole document
    bbox: BBox | None = None
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class OcrDocument:
    """Raw output of the OCR stage, before normalisation.

    Cached to disk keyed by (pdf sha256, engine, engine_version, config hash), so
    a re-run never pays for Surya twice.
    """

    doc_id: str
    source_path: str
    page_count: int
    blocks: tuple[Block, ...]
    engine: str
    engine_version: str


# --------------------------------------------------------------------------- #
# Headings
# --------------------------------------------------------------------------- #


class HeaderSource(StrEnum):
    """Where a heading was detected. Ordered by trust, highest first."""

    OUTLINE = "outline"  # PDF bookmarks: the author's own section tree
    LAYOUT = "layout"  # Surya SECTION_HEADER region
    REGEX = "regex"  # numbered-heading pattern over reading-ordered text
    FONT = "font"  # font size/weight outlier

    @property
    def trust(self) -> float:
        return _SOURCE_TRUST[self]


_SOURCE_TRUST: Mapping[HeaderSource, float] = {
    HeaderSource.OUTLINE: 1.0,
    HeaderSource.LAYOUT: 0.8,
    HeaderSource.REGEX: 0.6,
    HeaderSource.FONT: 0.3,
}


@dataclass(frozen=True, slots=True)
class Heading:
    """A detected section heading, anchored to an offset in the normalised text."""

    title: str
    level: int  # 1 == top level ("3. Experiments"), 2 == "3.1 Setup", ...
    char_start: int  # offset of the heading itself in NormalizedDocument.text
    page: int
    number: str | None = None  # "3" or "3.1", None for unnumbered headings
    sources: frozenset[HeaderSource] = field(default_factory=frozenset)
    confidence: float = 0.0

    @property
    def label(self) -> str:
        """Human-facing section label, e.g. '3.1 Selective Scan'."""
        return f"{self.number} {self.title}".strip() if self.number else self.title

    @property
    def parent_number(self) -> str | None:
        """'3.1.2' -> '3.1'. Top-level headings have no parent."""
        if not self.number or "." not in self.number:
            return None
        return self.number.rsplit(".", 1)[0]


@dataclass(frozen=True, slots=True)
class PageSpan:
    """Half-open [start, end) character range of one page within the document text."""

    page: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """Reading-ordered text plus the structure we recovered from it.

    `text` is the single source of truth for offsets. Every heading and every chunk
    indexes into it, so a chunk's provenance is always checkable by slicing.
    """

    doc_id: str
    title: str
    text: str
    headings: tuple[Heading, ...]
    page_spans: tuple[PageSpan, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def page_at(self, offset: int) -> int:
        """Page number containing a character offset. Clamps rather than raising."""
        for span in self.page_spans:
            if span.start <= offset < span.end:
                return span.page
        return self.page_spans[-1].page if self.page_spans else 1


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


def make_chunk_id(doc_id: str, char_start: int, char_end: int, part_index: int) -> str:
    """Position-addressed chunk id: (doc, offsets, part), not the chunk text.

    Deterministic across runs and machines given identical ingest output, which is
    what makes snapshot tests and cross-run eval comparisons possible. It is NOT
    content-addressed: if re-OCR or a normalisation change alters the text while
    leaving these offsets intact, the id stays the same for different text.
    Anything that persists ids across ingest runs must also key on an ingest hash
    (`Config.ingest_hash`). Truncated to 16 hex chars: at our corpus size that is
    far past collision risk and stays readable in logs.
    """
    # The text is deliberately not part of the hash material: including it would
    # churn every id on any normalisation change. The trade-off is that the id
    # names a location within one ingest output, not the content at it.
    raw = f"{doc_id}:{char_start}:{char_end}:{part_index}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One section (or one part of an oversized section) of one paper.

    Offsets are trimmed to the non-whitespace extent of the span, so for a chunk
    without a repeated header (every `part_index == 0` chunk) `text` is exactly
    `document.text[char_start:char_end]`. Parts after the first re-attach the
    section header so they stay self-describing; for those the invariant is
    `text == header_line + "\\n\\n" + document.text[char_start:char_end]`.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    char_start: int
    char_end: int
    section_title: str
    page_start: int
    page_end: int
    section_number: str | None = None
    parent_section: str | None = None
    part_index: int = 0  # 0-based index within a split section
    part_count: int = 1  # 1 when the section was not split

    @property
    def section_label(self) -> str:
        base = (
            f"{self.section_number} {self.section_title}".strip()
            if self.section_number
            else self.section_title
        )
        if self.part_count > 1:
            return f"{base} (part {self.part_index + 1}/{self.part_count})"
        return base

    @property
    def citation_label(self) -> str:
        """What a reader sees, e.g. 'Mamba, section 3.2 Selective Scan, p.7'."""
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        return f"{self.doc_title}, section {self.section_label}, {pages}"

    @property
    def was_split(self) -> bool:
        return self.part_count > 1


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Scored:
    """A chunk with the score and rank a retriever assigned it."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str


# --------------------------------------------------------------------------- #
# Guardrail decisions
# --------------------------------------------------------------------------- #


class Action(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"


@dataclass(frozen=True, slots=True)
class Decision:
    """The typed result of one guardrail rule.

    Guardrails never raise and never silently mutate. They return one of these, so
    every rule is unit-testable in isolation and countable in evals.
    """

    rule_id: str
    action: Action
    reason: str = ""
    evidence: str = ""

    @property
    def allowed(self) -> bool:
        return self.action is not Action.DENY

    @classmethod
    def allow(cls, rule_id: str, reason: str = "") -> Decision:
        return cls(rule_id=rule_id, action=Action.ALLOW, reason=reason)

    @classmethod
    def deny(cls, rule_id: str, reason: str, evidence: str = "") -> Decision:
        return cls(rule_id=rule_id, action=Action.DENY, reason=reason, evidence=evidence)

    @classmethod
    def modify(cls, rule_id: str, reason: str, evidence: str = "") -> Decision:
        return cls(rule_id=rule_id, action=Action.MODIFY, reason=reason, evidence=evidence)


def first_denial(decisions: Sequence[Decision]) -> Decision | None:
    return next((d for d in decisions if d.action is Action.DENY), None)


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


class AnswerStatus(StrEnum):
    OK = "ok"
    BLOCKED_INPUT = "blocked_input"
    NO_RESULTS = "no_results"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BLOCKED_OUTPUT = "blocked_output"

    @property
    def is_answer(self) -> bool:
        return self is AnswerStatus.OK


@dataclass(frozen=True, slots=True)
class Citation:
    """A claim-level citation, validated against the retrieved set before it ships."""

    chunk_id: str
    quote: str
    label: str = ""


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    llm_calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
        )


@dataclass(frozen=True, slots=True)
class Answer:
    """The one thing the pipeline returns. Never a bare string."""

    status: AnswerStatus
    text: str
    citations: tuple[Citation, ...] = ()
    retrieved: tuple[Scored, ...] = ()
    decisions: tuple[Decision, ...] = ()
    usage: Usage = field(default_factory=Usage)
    trace_id: str = ""

    @property
    def sources(self) -> tuple[str, ...]:
        """Distinct citation labels, in first-cited order."""
        seen: dict[str, None] = {}
        for c in self.citations:
            seen.setdefault(c.label or c.chunk_id, None)
        return tuple(seen)
