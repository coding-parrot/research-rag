"""HTML blog ingestion: local snapshots in, OCR-shaped blocks out.

Two decisions live here and both exist because blogs are not arXiv.

SNAPSHOT SEMANTICS. The blog publishers in the corpus block automated clients
(Medium answers 403, Uber 406), so the source of truth is a local snapshot at
`<dest_dir>/<id>.html` that a human (or a real browser) places once. The fetcher
uses the snapshot when it is present and matches its manifest pin, attempts one
polite HTTP download (browser User-Agent) only when the snapshot is missing, and
otherwise fails with the exact path to place the file at. A snapshot that drifts
from its pin is an error, not a re-download: unlike a PDF, re-fetching cannot
restore the pinned bytes, because the pin was taken from a hand-placed file the
network never served us. The corpus must not drift silently either way.

HEADING MAPPING. The first h1 is the TITLE block and each h2 becomes a
SECTION_HEADER, i.e. a top-level section boundary. h3/h4 (and deeper) do NOT
become SECTION_HEADER: they are emitted as plain TEXT paragraphs so they stay
inside their h2 section. Blog headings are unnumbered, and the layout signal
treats every unnumbered heading as level 1, so an h3 SECTION_HEADER would clear
the chunker's `max_depth=1` bar and wrongly split its parent h2 section in two.
Losing the h3's heading-ness costs nothing: its text still lands inside the
right chunk, which is all retrieval needs.

Blogs have no pages, so every block reports page 1 and reading order is document
order.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from importlib.metadata import version as _package_version
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from rag.domain import Block, BlockType, OcrDocument
from rag.errors import FetchError
from rag.ingest.fetch import MAX_BYTES, Downloader, FetchResult, sha256_bytes, sha256_file
from rag.ingest.manifest import Paper
from rag.ingest.ocr.base import blocks_in_reading_order
from rag.observability import get_logger

log = get_logger("html")

ENGINE_NAME = "html"

# A mainstream browser UA. The honest research-rag UA from ingest/fetch.py is
# exactly what these publishers 403; a browser UA is the one variation worth
# trying before asking a human for a snapshot.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Tags that never carry article content.
_STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]
# Class/id vocabulary of blog chrome: navigation, subscription prompts, cookie
# banners, "read next" rails. Matched as substrings of class tokens and ids.
_CHROME_RE = re.compile(r"nav|menu|footer|sidebar|related|recommended|signup|banner|cookie", re.I)

# Tags that become blocks, in document order. A tag nested inside another emitted
# tag is skipped, because the ancestor's text already contains it (a p inside an
# li, a code inside a pre); emitting both would duplicate content in the corpus.
_EMIT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "figcaption", "pre", "code"]
_EMIT_SET = frozenset(_EMIT_TAGS)


class HtmlDownloader:
    """Best-effort HTTP download of a blog page with a browser User-Agent.

    Usually fails against the corpus publishers (bot detection sits in front of
    the origin), which is why the snapshot path exists at all. Kept because it
    makes self-hosted engineering blogs ingest with zero manual steps.
    """

    def __init__(self, timeout: float = 60.0, max_bytes: int = MAX_BYTES) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes

    def get(self, url: str) -> bytes:
        import httpx

        try:
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": BROWSER_USER_AGENT},
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.content
        except httpx.HTTPError as exc:
            raise FetchError(f"download failed for {url}: {exc}") from exc

        if len(data) > self._max_bytes:
            raise FetchError(f"{url} exceeds the {self._max_bytes} byte cap")
        return data


class HtmlSnapshotFetcher:
    """Materialises `kind: html` manifest entries from local snapshots.

    Same digest contract as `PdfFetcher`: a pinned sha256 is a promise about the
    bytes, and a mismatch fails loudly. The difference is the recovery path;
    see the module docstring.
    """

    def __init__(self, dest_dir: Path, downloader: Downloader | None = None) -> None:
        self._dest = Path(dest_dir)
        self._downloader = downloader if downloader is not None else HtmlDownloader()

    def fetch(self, paper: Paper) -> FetchResult:
        self._dest.mkdir(parents=True, exist_ok=True)
        path = self._dest / paper.filename

        if path.exists():
            digest = sha256_file(path)
            if paper.sha256 is None or digest == paper.sha256:
                log.debug("snapshot hit", fields={"paper": paper.id, "sha256": digest[:12]})
                return FetchResult(paper=paper, path=path, sha256=digest, downloaded=False)
            raise FetchError(
                f"{paper.id}: snapshot {path} has digest {digest[:12]}… but the manifest "
                f"pins {paper.sha256[:12]}…. The snapshot was edited or re-saved after "
                f"pinning; restore the original file or update the pin deliberately "
                f"rather than letting the corpus drift."
            )

        log.info("no snapshot, attempting download", fields={"paper": paper.id, "url": paper.url})
        try:
            data = self._downloader.get(paper.url)
        except FetchError as exc:
            raise FetchError(
                f"{paper.id}: no snapshot at {path} and the download failed ({exc}). "
                f"This publisher blocks automated fetching; open {paper.url} in a real "
                f"browser, save the page as HTML to exactly {path}, and re-run ingest."
            ) from exc

        digest = sha256_bytes(data)
        if paper.sha256 is not None and digest != paper.sha256:
            raise FetchError(
                f"{paper.id}: digest mismatch. Manifest pins {paper.sha256[:12]}… but "
                f"{paper.url} now serves {digest[:12]}…. The source changed under a "
                f"pinned id; update the manifest deliberately rather than letting the "
                f"corpus drift."
            )

        path.write_bytes(data)
        return FetchResult(paper=paper, path=path, sha256=digest, downloaded=True)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_html(html: str, doc_id: str, source_path: str) -> OcrDocument:
    """Turn a blog snapshot into the same `OcrDocument` shape OCR produces.

    Downstream (normalise, header detection, chunking) then runs unchanged; the
    layout signal picks the SECTION_HEADER blocks up exactly as it would Surya's.
    """
    soup = BeautifulSoup(html, "html.parser")
    scope = _main_content(soup)
    _strip_chrome(scope)

    blocks: list[Block] = []
    saw_title = False
    for element in _emittable(scope):
        name = element.name or ""
        if name == "h1":
            # Only the first h1 is the document title; a later h1 is at least as
            # top-level as an h2, so it becomes a section boundary.
            block_type = BlockType.SECTION_HEADER if saw_title else BlockType.TITLE
            saw_title = True
            text = _collapse(element)
        elif name == "h2":
            block_type, text = BlockType.SECTION_HEADER, _collapse(element)
        elif name in ("h3", "h4", "h5", "h6"):
            # Deliberately TEXT, not SECTION_HEADER; see the module docstring.
            block_type, text = BlockType.TEXT, _collapse(element)
        elif name == "li":
            block_type, text = BlockType.LIST_ITEM, f"- {_collapse(element)}"
        elif name == "table":
            block_type, text = BlockType.TABLE, _table_text(element)
        elif name == "figcaption":
            block_type, text = BlockType.CAPTION, _collapse(element)
        elif name in ("pre", "code"):
            # Verbatim: collapsing whitespace would destroy code indentation.
            block_type, text = BlockType.TEXT, element.get_text().strip("\n")
        else:  # p
            block_type, text = BlockType.TEXT, _collapse(element)

        if not text.strip():
            continue
        blocks.append(
            Block(type=block_type, text=text, page=1, order=len(blocks), bbox=None, confidence=1.0)
        )

    return OcrDocument(
        doc_id=doc_id,
        source_path=source_path,
        page_count=1,
        blocks=blocks_in_reading_order(blocks),
        engine=ENGINE_NAME,
        engine_version=_package_version("beautifulsoup4"),
    )


def _main_content(soup: BeautifulSoup) -> Tag:
    """The subtree that holds the article body.

    The first `article` tag wins outright: on the corpus blogs it wraps exactly
    the post. Without one, take the densest of main / div[role=main] / body by
    paragraph count, because chrome-heavy pages hang whole navigation trees off
    body while the real content sits under main.
    """
    article = soup.find("article")
    if isinstance(article, Tag):
        return article

    candidates = [soup.find("main"), soup.find("div", attrs={"role": "main"}), soup.body]
    best: Tag | None = None
    best_count = -1
    for candidate in candidates:
        if not isinstance(candidate, Tag):
            continue
        count = len(candidate.find_all("p"))
        if count > best_count:  # strict: earlier candidates win ties, they are more specific
            best, best_count = candidate, count
    return best if best is not None else soup


def _strip_chrome(scope: Tag) -> None:
    """Remove non-content subtrees in place, tag-name matches first, then class/id."""
    for tag in scope.find_all(_STRIP_TAGS):
        if isinstance(tag, Tag) and not tag.decomposed:
            tag.decompose()
    for tag in scope.find_all(True):
        if isinstance(tag, Tag) and not tag.decomposed and _looks_like_chrome(tag):
            tag.decompose()


def _looks_like_chrome(tag: Tag) -> bool:
    tokens: list[str] = []
    tag_id = tag.get("id")
    if isinstance(tag_id, str):
        tokens.append(tag_id)
    classes = tag.get("class")
    if isinstance(classes, str):
        tokens.append(classes)
    elif isinstance(classes, list):
        tokens.extend(c for c in classes if isinstance(c, str))
    return any(_CHROME_RE.search(token) for token in tokens)


def _emittable(scope: Tag) -> Iterator[Tag]:
    """Emit-worthy tags in document order, skipping tags inside emitted ancestors."""
    for element in scope.find_all(_EMIT_TAGS):
        if not isinstance(element, Tag) or element.decomposed:
            continue
        if any(isinstance(parent, Tag) and parent.name in _EMIT_SET for parent in element.parents):
            continue
        yield element


def _collapse(element: Tag) -> str:
    """One-line text of an element, HTML whitespace collapsed."""
    return " ".join(element.get_text().split())


def _table_text(table: Tag) -> str:
    """Pipe-delimited rows, matching the shape OCR table recognition produces."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [
            " ".join(cell.get_text().split())
            for cell in tr.find_all(["th", "td"])
            if isinstance(cell, Tag)
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)
