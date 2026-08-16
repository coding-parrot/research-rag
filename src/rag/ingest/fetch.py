"""Content-addressed PDF fetching.

Two rules:

1. A pinned digest is a contract. If the bytes on the wire no longer match, we fail
   loudly rather than silently re-ingesting a different paper under the same id.
2. Nothing is re-downloaded that is already on disk and passes its digest check.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.errors import FetchError
from rag.ingest.manifest import Manifest, Paper
from rag.observability import get_logger

log = get_logger("fetch")

USER_AGENT = "research-rag/0.1 (+https://github.com/InterviewReady/ai-engineering-resources)"
PDF_MAGIC = b"%PDF-"
MAX_BYTES = 64 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Downloader(Protocol):
    """Seam that lets tests fetch without a network."""

    def get(self, url: str) -> bytes: ...


class HttpDownloader:
    """Real HTTP downloader. Follows redirects, caps size, verifies it got a PDF."""

    def __init__(self, timeout: float = 60.0, max_bytes: int = MAX_BYTES) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes

    def get(self, url: str) -> bytes:
        import httpx

        try:
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.content
        except httpx.HTTPError as exc:
            raise FetchError(f"download failed for {url}: {exc}") from exc

        if len(data) > self._max_bytes:
            raise FetchError(f"{url} exceeds the {self._max_bytes} byte cap")
        if not data.startswith(PDF_MAGIC):
            raise FetchError(f"{url} did not return a PDF (got {data[:16]!r})")
        return data


class StubDownloader:
    """In-memory downloader for tests. Raises on any URL it was not primed with."""

    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = dict(responses)
        self.calls: list[str] = []

    def get(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self._responses:
            raise FetchError(f"StubDownloader has no response for {url}")
        return self._responses[url]


@dataclass(frozen=True, slots=True)
class FetchResult:
    paper: Paper
    path: Path
    sha256: str
    downloaded: bool  # False when served from disk

    @property
    def newly_pinned(self) -> bool:
        """True when this fetch is what established the digest."""
        return self.paper.sha256 is None


class PdfFetcher:
    """Fetches manifest entries into a local directory, verifying digests."""

    def __init__(self, dest_dir: Path, downloader: Downloader | None = None) -> None:
        self._dest = Path(dest_dir)
        self._downloader = downloader if downloader is not None else HttpDownloader()

    def fetch(self, paper: Paper) -> FetchResult:
        self._dest.mkdir(parents=True, exist_ok=True)
        path = self._dest / paper.filename

        if path.exists():
            digest = sha256_file(path)
            if paper.sha256 is None or digest == paper.sha256:
                log.debug("cache hit", fields={"paper": paper.id, "sha256": digest[:12]})
                return FetchResult(paper=paper, path=path, sha256=digest, downloaded=False)
            log.warning(
                "cached file does not match pinned digest, re-downloading",
                fields={"paper": paper.id, "expected": paper.sha256[:12], "found": digest[:12]},
            )

        log.info("downloading", fields={"paper": paper.id, "url": paper.url})
        data = self._downloader.get(paper.url)
        digest = sha256_bytes(data)

        if paper.sha256 is not None and digest != paper.sha256:
            raise FetchError(
                f"{paper.id}: digest mismatch. Manifest pins {paper.sha256[:12]}… but "
                f"{paper.url} now serves {digest[:12]}…. The source changed under a pinned "
                f"id; update the manifest deliberately rather than letting the corpus drift."
            )

        path.write_bytes(data)
        return FetchResult(paper=paper, path=path, sha256=digest, downloaded=True)

    def fetch_all(self, papers: Sequence[Paper]) -> tuple[list[FetchResult], list[tuple[str, str]]]:
        """Fetch a set of papers, collecting failures instead of aborting on the first.

        Failures are returned as (paper_id, error) pairs, not just logged: the
        ingest service must record a paper that failed to fetch as a failed
        document, or it vanishes from the report under a green exit code.
        """
        results: list[FetchResult] = []
        failures: list[tuple[str, str]] = []
        for paper in papers:
            try:
                results.append(self.fetch(paper))
            except FetchError as exc:
                failures.append((paper.id, str(exc)))
                log.error("fetch failed", fields={"paper": paper.id, "error": str(exc)})
        if failures and not results:
            raise FetchError(
                "every fetch failed:\n  " + "\n  ".join(f"{pid}: {err}" for pid, err in failures)
            )
        if failures:
            log.warning("some fetches failed", fields={"failed": len(failures), "ok": len(results)})
        return results, failures


def pin_digests(manifest: Manifest, results: Sequence[FetchResult]) -> Manifest:
    """Return a manifest with newly discovered digests written in.

    Existing pins are never overwritten: a mismatch would already have raised.
    """
    by_id = {r.paper.id: r.sha256 for r in results}
    updated = [
        p.with_digest(by_id[p.id]) if p.sha256 is None and p.id in by_id else p for p in manifest
    ]
    return manifest.with_papers(updated)
