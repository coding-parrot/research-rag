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
from rag.domain import Chunk, Heading, NormalizedDocument, make_chunk_id
from rag.errors import ConfigError
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

    `sections_detected == 1` on a research paper means the document collapsed to a
    single chunkable section. Compare it against `sections_before_merge` to tell
    the two causes apart: if both are 1, header detection failed; if the pre-merge
    count is healthy, the min-chunk floor merged the sections away. Those need
    different fixes, so the pair is the thing to alert on.
    """

    doc_id: str
    sections_detected: int  # after the merge policy: what chunking worked from
    sections_before_merge: int  # straight from header detection, before merging
    chunks_emitted: int
    sections_split: int
    sections_merged: int
    boundaries_dropped: int  # headings that shared a char_start with a stronger one
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
        # Pydantic bounds max_chunk_tokens >= 64 but chars_per_token is any float
        # above zero, so the character budget can round to 0 (e.g. 64 * 0.01). A
        # zero budget would leave the split loop in _split_large unable to advance,
        # so reject the product here rather than hanging mid-ingest.
        if config.max_chunk_chars < 1:
            raise ConfigError(
                "max_chunk_tokens * chars_per_token must yield a budget of at least "
                f"1 character, got {config.max_chunk_chars}"
            )
        self._config = config

    def chunk(self, document: NormalizedDocument) -> tuple[tuple[Chunk, ...], ChunkReport]:
        sections, dropped_boundaries = self._sections(document)
        sections_before_merge = len(sections)
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
            sections_before_merge=sections_before_merge,
            chunks_emitted=len(chunks),
            sections_split=split_count,
            sections_merged=merged,
            boundaries_dropped=dropped_boundaries,
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

    def _sections(self, document: NormalizedDocument) -> tuple[list[Section], int]:
        """Cut the document at each heading at or above `max_depth`.

        Deeper headings are left inside their parent, which is what makes `3.1` and
        `3.2` part of the `3. Experiments` chunk at the default depth of 1.

        Returns the sections plus the number of headings dropped because another
        heading claimed the same offset.
        """
        boundaries = [h for h in document.headings if h.level <= self._config.max_depth]
        text_length = len(document.text)

        # Detection signals can anchor two distinct headings at the same offset
        # (e.g. two outline bookmarks resolving to one text position). Keep one per
        # offset: highest confidence wins, deeper level breaks ties (the more
        # specific heading), first-seen order breaks the rest so output stays
        # deterministic. Without this the zero-length span guard below would drop
        # whichever heading dict order happened to disfavour, silently.
        by_start: dict[int, Heading] = {}
        dropped = 0
        for heading in boundaries:
            kept = by_start.get(heading.char_start)
            if kept is None:
                by_start[heading.char_start] = heading
                continue
            dropped += 1
            if (heading.confidence, heading.level) > (kept.confidence, kept.level):
                by_start[heading.char_start] = heading
        boundaries = list(by_start.values())
        if dropped:
            log.warning(
                "dropped headings sharing an offset with a stronger heading",
                fields={"doc_id": document.doc_id, "dropped": dropped},
            )

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
            ], dropped

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
                # Backstop only: same-offset headings were deduplicated above, so
                # this now fires solely on out-of-order or past-the-end headings.
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
        return sections, dropped

    # ------------------------------------------------------------------ #
    # Step 2: merge undersized sections forward
    # ------------------------------------------------------------------ #

    def _merge_small(self, sections: list[Section]) -> tuple[list[Section], int]:
        """Fold a too-short section into the one that follows it.

        Forward rather than backward because a short section is usually a heading
        immediately followed by its real content under the next heading (a stray
        boundary), and because merging forward keeps the earlier section's title,
        which is the one a reader would cite.

        A short section absorbs at most ONE follower. If the merged span is still
        under the floor it is emitted anyway: re-pending it would let one tiny
        leading section chain-absorb the rest of the paper and stamp its title on
        every section it swallowed.

        A short FINAL section has no follower, so it folds backwards into the
        previous kept section instead, keeping that section's title.
        """
        floor = self._config.min_chunk_chars
        if floor <= 0 or len(sections) < 2:
            return sections, 0

        merged: list[Section] = []
        pending: Section | None = None
        count = 0

        for original in sections:
            if pending is not None:
                # Absorb the short section into this one, keeping the short one's
                # title: it is the heading a reader would cite for this span. Emit
                # immediately, even below the floor: one absorption per short
                # section is the cap.
                merged.append(_span(pending, char_end=original.char_end))
                pending = None
                count += 1
                continue

            if original.length < floor:
                pending = original
                continue
            merged.append(original)

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
        # The constructor guarantees limit >= 1, which is what makes the cursor
        # below always advance: every accepted cut is strictly past the cursor.
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

        # Tighten each part to its non-whitespace extent and drop whitespace-only
        # parts BEFORE numbering. Tightening the offsets (instead of stripping the
        # text) keeps `text == document.text[char_start:char_end]` true, and
        # numbering only the surviving parts keeps part_index/part_count free of
        # holes: a citation label must never advertise a part that does not exist.
        kept: list[tuple[int, int]] = []
        for start, end in parts:
            raw = document.text[start:end]
            stripped = raw.strip()
            if not stripped:
                continue
            lead = len(raw) - len(raw.lstrip())
            kept.append((start + lead, start + lead + len(stripped)))

        chunks: list[Chunk] = []
        for index, (start, end) in enumerate(kept):
            body = document.text[start:end]

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
                    part_count=len(kept),
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
