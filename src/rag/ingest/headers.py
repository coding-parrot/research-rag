"""Layered section-header detection.

Section headers are the only chunk delimiter this system has. There is no fallback
strategy, so a missed boundary is not a degraded chunk, it is a wrong one: two
unrelated sections fused into a single embedding. That makes this the highest-stakes
module in the pipeline, and the reason detection is layered rather than a regex.

Four signals, merged by agreement:

  OUTLINE  PDF bookmarks. Most arXiv PDFs come from pdflatex with hyperref, so the
           bookmark tree is literally the author's own \\section commands. Highest
           precision available and it costs nothing.
  LAYOUT   Surya SECTION_HEADER regions. The signal that works when there are no
           bookmarks, including on scans.
  REGEX    Numbered-heading pattern over reading-ordered text. Catches what layout
           misses, and its disagreements with LAYOUT are the debugging surface.
  FONT     Font-size outliers. Noisy on arXiv, off by default.

Confidence is the sum of contributing source trusts, capped at 1.0, so two mutually
corroborating weak signals can clear the bar that neither clears alone.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.config import HeaderConfig
from rag.domain import BlockType, HeaderSource, Heading, NormalizedDocument
from rag.ingest.normalize import BlockSpan, NormalizationResult, page_of_offset
from rag.observability import get_logger

log = get_logger("headers")

# "3", "3.1", "3.1.2" followed by a title. The title must start with a capital and
# must not run on for a paragraph, which is what separates a heading from a sentence
# that happens to begin with a figure number.
_NUMBERED = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(?P<title>[A-Z][^\n]{1,120}?)\s*$",
    re.MULTILINE,
)

# Unnumbered sections that every paper has and that carry real answers.
_WELL_KNOWN = re.compile(
    r"^(?P<title>Abstract|Introduction|Related Work|Background|Method(?:s|ology)?|"
    r"Approach|Model|Architecture|Experiments?|Experimental Setup|Setup|Results?|"
    r"Evaluation|Analysis|Ablations?|Ablation Study|Discussion|Limitations?|"
    r"Conclusions?|Future Work|Acknowledge?ments?|References|Bibliography|"
    r"Appendix(?:\s+[A-Z])?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Things that look numbered but are not sections.
_NOT_A_SECTION = re.compile(
    r"^(Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|Algorithm|Alg\.?|Theorem|Lemma|"
    r"Definition|Corollary|Proposition|Example|Step|Line|Chapter|Version|Page|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    re.IGNORECASE,
)

# Anything at or past these is back matter. Kept as chunks but never used to
# anchor the "did detection work" sanity check.
_BACK_MATTER = frozenset({"references", "bibliography", "acknowledgment", "acknowledgement"})


@dataclass(frozen=True, slots=True)
class HeadingCandidate:
    """One detection from one signal, before merging."""

    title: str
    char_start: int
    source: HeaderSource
    number: str | None = None
    # Bookmark-tree depth. Only the outline signal knows it; for an unnumbered
    # title it is the only depth information that exists at all.
    level: int | None = None


@dataclass(frozen=True, slots=True)
class DetectionReport:
    """Per-document diagnostics. Written to the ingest report so a silent
    detection failure shows up as a number rather than as bad answers later."""

    doc_id: str
    accepted: int
    rejected: int
    by_source: dict[str, int]
    disagreements: int
    used_outline: bool

    @property
    def looks_healthy(self) -> bool:
        """A research paper with fewer than three sections almost certainly failed."""
        return self.accepted >= 3


class HeaderDetector:
    """Runs the enabled signals, merges them, and returns ordered headings."""

    def __init__(self, config: HeaderConfig) -> None:
        self._config = config

    def detect(
        self,
        result: NormalizationResult,
        *,
        outline: Sequence[OutlineEntry] | None = None,
    ) -> tuple[tuple[Heading, ...], DetectionReport]:
        document = result.document
        candidates: list[HeadingCandidate] = []

        if self._config.use_outline and outline:
            candidates.extend(_from_outline(outline, document))
        if self._config.use_layout:
            candidates.extend(_from_layout(result.spans))
        if self._config.use_regex:
            candidates.extend(_from_regex(document.text))
        if self._config.use_font:
            candidates.extend(_from_font(result.spans))

        headings, rejected, disagreements = self._merge(candidates, document)

        by_source: dict[str, int] = {}
        for heading in headings:
            for source in heading.sources:
                by_source[source.value] = by_source.get(source.value, 0) + 1

        report = DetectionReport(
            doc_id=document.doc_id,
            accepted=len(headings),
            rejected=rejected,
            by_source=by_source,
            disagreements=disagreements,
            used_outline=bool(outline) and self._config.use_outline,
        )
        if not report.looks_healthy:
            log.warning(
                "header detection found too few sections; the document will chunk badly",
                fields={"doc_id": document.doc_id, "accepted": report.accepted},
            )
        return headings, report

    # ------------------------------------------------------------------ #

    def _merge(
        self, candidates: Iterable[HeadingCandidate], document: NormalizedDocument
    ) -> tuple[tuple[Heading, ...], int, int]:
        """Cluster detections of the same heading, score them, drop the failures.

        Two candidates cluster only when they are near each other *and* plausibly
        name the same heading (same section number, or same normalised title).
        Proximity alone is not enough: a genuinely tiny section puts the next
        section's heading within the window, and clustering on distance alone
        would swallow that boundary entirely.
        """
        ordered = sorted(candidates, key=lambda c: c.char_start)
        clusters: list[list[HeadingCandidate]] = []
        for candidate in ordered:
            joined = False
            if clusters:
                cluster = clusters[-1]
                near = candidate.char_start - cluster[0].char_start <= self._config.merge_window
                if near and any(_same_heading(candidate, member) for member in cluster):
                    cluster.append(candidate)
                    joined = True
            if not joined:
                clusters.append([candidate])

        headings: list[Heading] = []
        rejected = 0
        disagreements = 0

        for cluster in clusters:
            titles = {_normalise_title(c.title) for c in cluster}
            if len(titles) > 1:
                disagreements += 1

            best = _preferred(cluster)
            sources = frozenset(c.source for c in cluster)
            confidence = min(1.0, sum(s.trust for s in sources))

            if confidence < self._config.min_confidence:
                rejected += 1
                continue
            if not self._is_plausible(best):
                rejected += 1
                continue

            number = best.number
            # An unnumbered title carries no depth of its own, so fall back to the
            # bookmark-tree level when the winning candidate has one. Otherwise a
            # nested bookmark would surface as level 1 and become a chunk boundary
            # inside its parent section at max_depth=1.
            level = _level_of(number) if number else (best.level or 1)
            headings.append(
                Heading(
                    title=best.title.strip(),
                    level=level,
                    char_start=best.char_start,
                    page=page_of_offset(document.page_spans, best.char_start),
                    number=number,
                    sources=sources,
                    confidence=round(confidence, 3),
                )
            )

        return tuple(headings), rejected, disagreements

    def _is_plausible(self, candidate: HeadingCandidate) -> bool:
        title = candidate.title.strip()
        if not title or len(title) > self._config.max_title_chars:
            return False
        if _NOT_A_SECTION.match(title):
            return False
        if candidate.number:
            try:
                top = int(candidate.number.split(".")[0])
            except ValueError:
                return False
            if not 1 <= top <= self._config.max_section_number:
                return False
        return True


# --------------------------------------------------------------------------- #
# Signal 1: PDF outline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """One PDF bookmark."""

    title: str
    level: int
    page: int  # 1-indexed


def read_outline(pdf_path: Path) -> tuple[OutlineEntry, ...]:
    """Read the PDF bookmark tree. Returns empty when the PDF has none.

    Cheap, high precision, and worth trying first on every document.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        log.debug("pypdfium2 not installed, skipping outline signal")
        return ()

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        log.warning(
            "could not open pdf for outline", fields={"path": str(pdf_path), "error": str(exc)}
        )
        return ()

    entries: list[OutlineEntry] = []
    try:
        # item.title / item.page_index is the pypdfium2 v4 bookmark API. v5 replaced
        # get_toc's items with PdfBookmark objects (get_title()/get_dest() methods),
        # under which this loop raises AttributeError; the except below then degrades
        # to no outline signal rather than breaking ingest. The pyproject pin should
        # stay below 5 until this loop speaks the v5 API.
        for item in doc.get_toc():
            title = (item.title or "").strip()
            if not title:
                continue
            page_index = item.page_index if item.page_index is not None else 0
            entries.append(
                OutlineEntry(
                    title=title, level=int(getattr(item, "level", 0)) + 1, page=page_index + 1
                )
            )
    except Exception as exc:
        log.warning("outline read failed", fields={"path": str(pdf_path), "error": str(exc)})
        return ()
    finally:
        doc.close()

    log.debug("outline read", fields={"path": str(pdf_path), "entries": len(entries)})
    return tuple(entries)


def _from_outline(
    outline: Sequence[OutlineEntry], document: NormalizedDocument
) -> list[HeadingCandidate]:
    """Anchor each bookmark to a character offset by searching near its page.

    A bookmark knows its title and page but not its offset. We look for the title
    as a full line within the page's span; a substring hit inside body prose ("in
    the conclusion we argue...") would plant a trust-1.0 boundary mid-paragraph.
    The line may carry a section number the bookmark title had stripped, so the
    pattern allows an optional numeric prefix and the anchor lands on the number,
    keeping it out of the previous chunk's tail. If nothing matches near the page
    (recognition differs slightly from the bookmark string) the bookmark is
    dropped rather than anchored to a guess.
    """
    candidates: list[HeadingCandidate] = []

    for entry in outline:
        number, title = _split_number(entry.title)
        span = next((s for s in document.page_spans if s.page == entry.page), None)
        # Search from one page earlier: a section can start at the foot of the
        # previous page while the bookmark points at where it becomes visible.
        start = max(0, (span.start if span else 0) - 2000)
        end = span.end + 2000 if span else len(document.text)

        needle = _normalise_title(title)
        if len(needle) < 3:
            continue
        # IGNORECASE over the original text, never a search over text.lower():
        # str.lower() is not length-preserving (U+0130 lowers to two code points),
        # and drifted offsets silently corrupt every downstream slice.
        pattern = re.compile(
            r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?[\s:.\-]+)?" + re.escape(needle) + r"\s*$",
            re.MULTILINE | re.IGNORECASE,
        )
        match = pattern.search(document.text, start, end)
        if match is None:
            continue

        candidates.append(
            HeadingCandidate(
                title=title,
                char_start=match.start(),
                source=HeaderSource.OUTLINE,
                number=number,
                level=entry.level,
            )
        )
    return candidates


# --------------------------------------------------------------------------- #
# Signal 2: Surya layout
# --------------------------------------------------------------------------- #


def _from_layout(spans: Sequence[BlockSpan]) -> list[HeadingCandidate]:
    """Every block Surya typed as a section header."""
    candidates: list[HeadingCandidate] = []
    for span in spans:
        if span.block.type is not BlockType.SECTION_HEADER:
            continue
        number, title = _split_number(span.text.strip())
        if title:
            candidates.append(
                HeadingCandidate(
                    title=title, char_start=span.start, source=HeaderSource.LAYOUT, number=number
                )
            )
    return candidates


# --------------------------------------------------------------------------- #
# Signal 3: regex over reading-ordered text
# --------------------------------------------------------------------------- #


def _from_regex(text: str) -> list[HeadingCandidate]:
    """Numbered headings, plus the unnumbered ones every paper has."""
    candidates = [
        HeadingCandidate(
            title=match.group("title").strip(),
            char_start=match.start(),
            source=HeaderSource.REGEX,
            number=match.group("num"),
        )
        for match in _NUMBERED.finditer(text)
    ]
    candidates.extend(
        HeadingCandidate(
            title=match.group("title").strip(),
            char_start=match.start(),
            source=HeaderSource.REGEX,
            number=None,
        )
        for match in _WELL_KNOWN.finditer(text)
    )
    return candidates


# --------------------------------------------------------------------------- #
# Signal 4: font size (off by default)
# --------------------------------------------------------------------------- #


def _from_font(spans: Sequence[BlockSpan]) -> list[HeadingCandidate]:
    """Short blocks whose glyph height is an outlier against the body text median."""
    heights = [
        s.block.bbox.height for s in spans if s.block.bbox and s.block.type is BlockType.TEXT
    ]
    if len(heights) < 5:
        return []
    median = statistics.median(heights)
    if median <= 0:
        return []

    candidates: list[HeadingCandidate] = []
    for span in spans:
        box = span.block.bbox
        text = span.text.strip()
        if box is None or not text or len(text) > 90 or "\n" in text:
            continue
        if box.height > median * 1.25:
            number, title = _split_number(text)
            if title:
                candidates.append(
                    HeadingCandidate(
                        title=title, char_start=span.start, source=HeaderSource.FONT, number=number
                    )
                )
    return candidates


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_LEADING_NUMBER = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})\.?[\s:.-]+(?P<rest>.+)$")


def _split_number(raw: str) -> tuple[str | None, str]:
    """'3.1 Selective Scan' -> ('3.1', 'Selective Scan')."""
    text = raw.strip()
    if match := _LEADING_NUMBER.match(text):
        return match.group("num"), match.group("rest").strip()
    return None, text


def _level_of(number: str | None) -> int:
    """Depth from the section number. Unnumbered headings are treated as top level."""
    return number.count(".") + 1 if number else 1


def _normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def _same_heading(a: HeadingCandidate, b: HeadingCandidate) -> bool:
    """Do two candidates plausibly name the same heading?

    Same section number is decisive. Otherwise compare normalised titles, allowing
    a prefix relationship because layout sometimes reads a truncated title while
    regex reads the full line (or vice versa).
    """
    if a.number and b.number:
        return a.number == b.number
    title_a, title_b = _normalise_title(a.title), _normalise_title(b.title)
    if not title_a or not title_b:
        return False
    return title_a == title_b or title_a.startswith(title_b) or title_b.startswith(title_a)


def _preferred(cluster: Sequence[HeadingCandidate]) -> HeadingCandidate:
    """Pick the representative of a cluster.

    Highest-trust source wins the title. A number from any source is kept, because
    layout often reads the title cleanly while regex is what recovers the numbering.
    """
    best = max(cluster, key=lambda c: (c.source.trust, bool(c.number), -c.char_start))
    if best.number is None:
        numbered = next((c for c in cluster if c.number), None)
        if numbered is not None:
            return HeadingCandidate(
                title=best.title,
                char_start=min(c.char_start for c in cluster),
                source=best.source,
                number=numbered.number,
            )
    return best


def is_back_matter(heading: Heading) -> bool:
    return _normalise_title(heading.title).rstrip("s") in _BACK_MATTER
