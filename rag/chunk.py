"""Step 2 - Chunking: split a paper at its section headings (Week 3: "Chunking Strategies").

The chunk is the atomic unit of retrieval, so chunk boundaries decide what an
answer can see. Too large and the meaning dilutes; too small and ideas get cut in
half. Papers already carry the right boundaries: their numbered section headings.
"1 Introduction", "2 Method", "3 Experiments" each become one chunk.

Two policies keep that honest on real papers:
  - a section longer than MAX_CHARS is split at paragraph breaks (an 8-page
    Experiments section would dilute retrieval and overflow the embedder),
  - a section shorter than MIN_CHARS is merged into the next one (a heading with
    one sentence under it is noise, not a chunk).
"""

import re
from dataclasses import dataclass

# "3 Experiments", "3.1 Setup", "7.2. Results" - a number, then a capitalised title.
HEADING = re.compile(r"^\s*\d{1,2}(\.\d{1,2})*\.?\s+[A-Z][^\n]{2,80}\s*$")
# Unnumbered headings every paper has.
KNOWN = re.compile(r"^\s*(Abstract|Introduction|Related Work|Conclusion|References|Appendix)\s*$", re.I)

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


def _split_at_headings(pages: list[str]) -> list[tuple[str, int, str]]:
    """Walk every line; a heading line starts a new section."""
    sections = []
    section, page_of_section, lines = "Front matter", 1, []
    for page_number, page in enumerate(pages, start=1):
        for line in page.splitlines():
            if HEADING.match(line) or KNOWN.match(line):
                sections.append((section, page_of_section, "\n".join(lines)))
                section, page_of_section, lines = line.strip(), page_number, []
            else:
                lines.append(line)
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
