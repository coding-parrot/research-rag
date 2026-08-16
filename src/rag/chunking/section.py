"""Section chunking. The only chunking strategy in this system.

A chunk is a section. `1. Introduction`, `2. Method` and `3. Experiments` become
three chunks, delimited by their headings, with `3.1` and `3.2` living inside
chunk 3 (configurable via `max_depth`).

Two policies keep that honest on a real paper:

  Oversized sections split. An Experiments section can run 12k characters while a
  Conclusion runs 400. A 12k-character chunk destroys the retrieval precision that
  section chunking exists to provide, and `all-MiniLM-L6-v2` truncates at 256 tokens
  anyway, so most of that chunk would never reach the index. Parts inherit the full
  section header, so a mid-section part is still self-describing.

  Undersized sections merge forward. A 200-character section is a heading with a
  sentence under it; on its own it is retrieval noise.

Both are reported per document so you can see how often they fire.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from rag.config import ChunkConfig
from rag.domain import Chunk, NormalizedDocument, make_chunk_id
from rag.observability import get_logger

log = get_logger("chunking")

# Preferred split points inside an oversized section, best first. We split on the
# strongest boundary available rather than mid-sentence.
_PARAGRAPH_BREAK = re.compile(r"\n\n+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


@dataclass(frozen=True, slots=True)
class Section:
    """A heading and the span of text it owns, before size policies are applied."""

    title: str
    char_start: int
    char_end: int
    number: str | None
    parent: str | None
    level: int

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True, slots=True)
class ChunkReport:
    """Per-document chunking diagnostics.

    `sections_detected == 1` on a research paper means header detection failed and
    the whole paper became one chunk. That is the number to alert on.
    """

    doc_id: str
    sections_detected: int
    chunks_emitted: int
    sections_split: int
    sections_merged: int
    max_chunk_chars: int
    median_chunk_chars: int

    @property
    def looks_healthy(self) -> bool:
        return self.sections_detected >= 3 and self.chunks_emitted >= 3


class SectionChunker:
    """Turns a `NormalizedDocument` into section chunks.

    Pure: no I/O, no model, no embedder. Given the same document and config it
    returns byte-identical chunks with identical ids, which is what makes snapshot
    tests and cross-run eval comparison possible.
    """

    def __init__(self, config: ChunkConfig) -> None:
        self._config = config

    def chunk(self, document: NormalizedDocument) -> tuple[tuple[Chunk, ...], ChunkReport]:
        sections = self._sections(document)
        sections, merged = self._merge_small(sections)

        chunks: list[Chunk] = []
        split_count = 0
        for section in sections:
            parts = self._split_large(document.text, section)
            if len(parts) > 1:
                split_count += 1
            chunks.extend(self._to_chunks(document, section, parts))

        lengths = sorted(len(c.text) for c in chunks)
        report = ChunkReport(
            doc_id=document.doc_id,
            sections_detected=len(sections),
            chunks_emitted=len(chunks),
            sections_split=split_count,
            sections_merged=merged,
            max_chunk_chars=lengths[-1] if lengths else 0,
            median_chunk_chars=lengths[len(lengths) // 2] if lengths else 0,
        )
        if not report.looks_healthy:
            log.warning(
                "document chunked into too few sections; check header detection",
                fields={"doc_id": document.doc_id, "sections": report.sections_detected},
            )
        return tuple(chunks), report

    # ------------------------------------------------------------------ #
    # Step 1: headings to sections
    # ------------------------------------------------------------------ #

    def _sections(self, document: NormalizedDocument) -> list[Section]:
        """Cut the document at each heading at or above `max_depth`.

        Deeper headings are left inside their parent, which is what makes `3.1` and
        `3.2` part of the `3. Experiments` chunk at the default depth of 1.
        """
        boundaries = [h for h in document.headings if h.level <= self._config.max_depth]
        text_length = len(document.text)

        if not boundaries:
            # No usable structure. One chunk, and the report will flag it.
            return [
                Section(
                    title=self._config.frontmatter_section_title,
                    char_start=0,
                    char_end=text_length,
                    number=None,
                    parent=None,
                    level=1,
                )
            ]

        sections: list[Section] = []

        # Text before the first heading is the title block and abstract. It answers
        # a lot of questions, so it becomes a chunk rather than being discarded.
        if boundaries[0].char_start > 0:
            sections.append(
                Section(
                    title=self._config.frontmatter_section_title,
                    char_start=0,
                    char_end=boundaries[0].char_start,
                    number=None,
                    parent=None,
                    level=1,
                )
            )

        for i, heading in enumerate(boundaries):
            end = boundaries[i + 1].char_start if i + 1 < len(boundaries) else text_length
            if end <= heading.char_start:
                continue
            sections.append(
                Section(
                    title=heading.title,
                    char_start=heading.char_start,
                    char_end=end,
                    number=heading.number,
                    parent=heading.parent_number,
                    level=heading.level,
                )
            )
        return sections

    # ------------------------------------------------------------------ #
    # Step 2: merge undersized sections forward
    # ------------------------------------------------------------------ #

    def _merge_small(self, sections: list[Section]) -> tuple[list[Section], int]:
        """Fold a too-short section into the one that follows it.

        Forward rather than backward because a short section is usually a heading
        immediately followed by its real content under the next heading (a stray
        boundary), and because merging forward keeps the earlier section's title,
        which is the one a reader would cite.
        """
        floor = self._config.min_chunk_chars
        if floor <= 0 or len(sections) < 2:
            return sections, 0

        merged: list[Section] = []
        pending: Section | None = None
        count = 0
        last_index = len(sections) - 1

        for index, original in enumerate(sections):
            current = original
            if pending is not None:
                # Absorb the short section into this one, keeping the short one's
                # title: it is the heading a reader would cite for this span.
                current = _span(pending, char_end=current.char_end)
                pending = None
                count += 1

            if current.length < floor and index < last_index:
                pending = current
                continue
            merged.append(current)

        if pending is not None:
            # The final section was short with nothing after it. Fold it backwards.
            if merged:
                merged[-1] = _span(merged[-1], char_end=pending.char_end)
                count += 1
            else:
                merged.append(pending)

        return merged, count

    # ------------------------------------------------------------------ #
    # Step 3: split oversized sections
    # ------------------------------------------------------------------ #

    def _split_large(self, text: str, section: Section) -> list[tuple[int, int]]:
        """Split a section into [start, end) parts that fit the token budget.

        Splits at paragraph breaks where possible, sentence boundaries otherwise,
        and only falls back to a hard character cut when a single sentence exceeds
        the budget on its own.
        """
        limit = self._config.max_chunk_chars
        if section.length <= limit:
            return [(section.char_start, section.char_end)]

        overlap = min(self._config.part_overlap_chars, limit // 2)
        body = text[section.char_start : section.char_end]
        candidates = _split_points(body)

        parts: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(body):
            target = cursor + limit
            if target >= len(body):
                parts.append((section.char_start + cursor, section.char_end))
                break

            cut = _best_cut(candidates, lower=cursor + limit // 2, upper=target)
            if cut is None:
                cut = target  # one very long sentence; hard cut rather than a huge chunk
            parts.append((section.char_start + cursor, section.char_start + cut))
            next_cursor = cut - overlap
            cursor = next_cursor if next_cursor > cursor else cut

        return parts

    # ------------------------------------------------------------------ #
    # Step 4: build chunks
    # ------------------------------------------------------------------ #

    def _to_chunks(
        self, document: NormalizedDocument, section: Section, parts: Sequence[tuple[int, int]]
    ) -> list[Chunk]:
        header_line = _header_line(section)
        chunks: list[Chunk] = []

        for index, (start, end) in enumerate(parts):
            body = document.text[start:end].strip()
            if not body:
                continue

            # Parts after the first lose the heading, so re-attach it. Without this
            # a retrieved part 3 of 5 has no idea which section it belongs to, and
            # neither does the embedding.
            text = body
            if index > 0 and self._config.repeat_header_in_parts and header_line:
                text = f"{header_line}\n\n{body}"

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, start, end, index),
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    text=text,
                    char_start=start,
                    char_end=end,
                    section_title=section.title,
                    section_number=section.number,
                    parent_section=section.parent,
                    page_start=document.page_at(start),
                    page_end=document.page_at(max(start, end - 1)),
                    part_index=index,
                    part_count=len(parts),
                )
            )
        return chunks


# --------------------------------------------------------------------------- #
# Split-point helpers
# --------------------------------------------------------------------------- #


def _span(section: Section, *, char_end: int) -> Section:
    """Copy a section with a new end offset, keeping its identity fields."""
    return Section(
        title=section.title,
        char_start=section.char_start,
        char_end=char_end,
        number=section.number,
        parent=section.parent,
        level=section.level,
    )


def _split_points(body: str) -> list[int]:
    """Offsets we are willing to cut at, sorted, paragraph breaks first in priority.

    We return a single sorted list and let `_best_cut` choose the latest point that
    fits, which naturally prefers paragraph breaks because they are rarer.
    """
    points = {match.end() for match in _PARAGRAPH_BREAK.finditer(body)}
    points.update(match.start() for match in _SENTENCE_END.finditer(body))
    return sorted(points)


def _best_cut(points: Sequence[int], *, lower: int, upper: int) -> int | None:
    """Latest split point in (lower, upper]. None when there is none."""
    index = bisect_right(points, upper) - 1
    if index < 0:
        return None
    cut = points[index]
    return cut if cut > lower else None


def _header_line(section: Section) -> str:
    if section.number:
        return f"{section.number} {section.title}".strip()
    return section.title.strip()
