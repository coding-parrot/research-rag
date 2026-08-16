"""Step 2 - Chunking: split a paper at its section headings (Week 3: "Chunking Strategies").

The chunk is the atomic unit of retrieval, so chunk boundaries decide what an
answer can see. Papers already carry the right boundaries: their numbered section
headings. "1 Introduction", "2 Method", "3.1 Setup" each start a new chunk.

Heading detection is ported from the course notebook's custom section chunker,
with its two defences against false positives:
  - a blacklist: "Figure 3", "Table 2", "Equation 5" look like headings but are not,
  - monotonicity: real section numbers only ever increase through a paper
    (1, 2, 3, 3.1, 3.2, 4 ...), so "2 GPUs were used" mid-paper is rejected
    when we are already past section 5.

Two size policies keep chunks retrieval-friendly:
  - a section longer than MAX_CHARS splits at paragraph breaks,
  - a section shorter than MIN_CHARS merges into the next one.
"""

import re
from dataclasses import dataclass

# The notebook's pattern: a section number, then a Title-Case (or ALL-CAPS) phrase.
# The title charset includes ?!()/. because real headings use them
# ("3 AREN'T EXISTING SOLUTIONS GOOD ENOUGH?").
HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][a-zA-Z][\w\-,'&:?!()/. ]{1,60})\s*$")
# Some layouts (ICLR style) emit the number and the title on separate lines.
NUMBER_ALONE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s*$")
TITLE_ALONE = re.compile(r"^\s*([A-Z][a-zA-Z][\w\-,'&:?!()/. ]{1,60})\s*$")
# Things that look numbered but are not sections (also from the notebook).
BLACKLIST = re.compile(
    r"^(Figure|Fig|Table|Tab|Equation|Eq|Section|Algorithm|Appendix|Theorem|Lemma|"
    r"Input|Output|Step|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Proceedings)\b",
    re.I,
)
# Unnumbered headings every paper has.
KNOWN = re.compile(r"^\s*(Abstract|Introduction|Related Work|Conclusion|References|Acknowledgements|Appendix)\s*$", re.I)

MAX_CHARS = 2500  # ~600 tokens: fits the embedding model comfortably
MIN_CHARS = 200   # anything shorter merges forward


@dataclass
class Chunk:
    id: str        # "lora:3" = paper id + running number
    paper: str     # paper id, e.g. "lora"
    title: str     # paper title, for citations
    section: str   # the heading this chunk lives under
    page: int      # 1-based page where the section starts
    text: str


def chunk_paper(paper: str, title: str, pages: list[str]) -> list[Chunk]:
    """Pages -> section chunks. The whole strategy in three passes."""
    sections = _split_at_headings(pages)
    sections = _merge_small(sections)
    chunks = []
    for section, page, text in sections:
        for part in _split_large(text):
            chunks.append(
                Chunk(id=f"{paper}:{len(chunks)}", paper=paper, title=title,
                      section=section, page=page, text=part)
            )
    return chunks


def _parse_num(num: str) -> tuple[int, ...]:
    """'3.1' -> (3, 1), so section numbers compare in document order."""
    return tuple(int(x) for x in num.strip(".").split("."))


def _split_at_headings(pages: list[str]) -> list[tuple[str, int, str]]:
    """Walk every line; an accepted heading starts a new section."""
    sections = []
    section, page_of_section, lines = "Front matter", 1, []
    prev_num: tuple[int, ...] = (0,)

    def accept(num: tuple[int, ...]) -> bool:
        # Monotonicity (the notebook's rule): real section numbers only move
        # forward, so "2 GPUs were used" inside section 5 is rejected. Jumps are
        # capped at two whole sections - enough to survive one missed heading
        # without letting table values like "70.0" register as section seventy.
        nonlocal prev_num
        if num > prev_num and num[0] <= prev_num[0] + 2 and num[0] <= 30:
            prev_num = num
            return True
        return False

    for page_number, page in enumerate(pages, start=1):
        page_lines = page.splitlines()
        i = 0
        while i < len(page_lines):
            line = page_lines[i]
            heading = None

            if (m := HEADING.match(line)) and not BLACKLIST.match(m.group(2)):
                if accept(_parse_num(m.group(1))):
                    heading = line.strip()
            elif (m := NUMBER_ALONE.match(line)) and i + 1 < len(page_lines):
                # ICLR-style layout: "2" on one line, "PROBLEM STATEMENT" on the next.
                t = TITLE_ALONE.match(page_lines[i + 1])
                if t and not BLACKLIST.match(t.group(1)) and accept(_parse_num(m.group(1))):
                    heading = f"{m.group(1)} {t.group(1).strip()}"
                    i += 1  # the title line is consumed by the heading
            elif KNOWN.match(line):
                heading = line.strip()

            if heading:
                sections.append((section, page_of_section, "\n".join(lines)))
                section, page_of_section, lines = heading, page_number, []
            else:
                lines.append(line)
            i += 1
    sections.append((section, page_of_section, "\n".join(lines)))
    return [(s, p, t.strip()) for s, p, t in sections if t.strip()]


def _merge_small(sections: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    """A tiny section joins the one after it, keeping its own heading."""
    merged = []
    carry = None
    for section, page, text in sections:
        if carry:
            section, page, text = carry[0], carry[1], carry[2] + "\n\n" + text
            carry = None
        if len(text) < MIN_CHARS:
            carry = (section, page, text)
        else:
            merged.append((section, page, text))
    if carry:  # a tiny final section has nothing to join, keep it as-is
        merged.append(carry)
    return merged


def _split_large(text: str) -> list[str]:
    """Cut an oversized section at the paragraph break nearest the size limit."""
    parts = []
    while len(text) > MAX_CHARS:
        cut = text.rfind("\n\n", 0, MAX_CHARS)
        if cut < MAX_CHARS // 2:   # no good break point: cut at the limit
            cut = MAX_CHARS
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    parts.append(text)
    return [p for p in parts if p]
