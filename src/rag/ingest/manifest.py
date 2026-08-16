"""The corpus manifest.

Documents are never committed. `corpus.yaml` records what the corpus *is* (id,
title, source URL, licence, expected digest) and the fetcher materialises it on
demand. That keeps the repo small, keeps licensing honest, and makes the corpus a
reviewable diff rather than a directory of binaries.

Two kinds of document share the manifest: arXiv PDFs, fetched over HTTP, and HTML
blog snapshots, placed on disk by a human because the publishers block automated
fetching. Both are pinned by sha256, so either kind drifting fails loudly.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast, get_args

import yaml

from rag.errors import ManifestError

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PaperKind = Literal["pdf", "html"]
_KINDS: frozenset[str] = frozenset(get_args(PaperKind))


@dataclass(frozen=True, slots=True)
class Paper:
    """One document in the corpus."""

    id: str
    title: str
    url: str
    topic: str
    license: str
    kind: PaperKind = "pdf"
    arxiv_id: str | None = None
    sha256: str | None = None  # None until first fetch pins it
    notes: str = ""

    @property
    def filename(self) -> str:
        """On-disk name of the fetched document; the extension follows the kind."""
        return f"{self.id}.html" if self.kind == "html" else f"{self.id}.pdf"

    def with_digest(self, digest: str) -> Paper:
        return replace(self, sha256=digest)


@dataclass(frozen=True, slots=True)
class Manifest:
    """The full corpus definition, loaded from and written back to YAML."""

    version: int
    source_repo: str
    papers: tuple[Paper, ...]

    def __iter__(self) -> Iterator[Paper]:
        return iter(self.papers)

    def __len__(self) -> int:
        return len(self.papers)

    def get(self, paper_id: str) -> Paper:
        for p in self.papers:
            if p.id == paper_id:
                return p
        raise ManifestError(f"no paper with id {paper_id!r} in the manifest")

    def select(self, ids: Sequence[str] | None) -> tuple[Paper, ...]:
        """Subset the corpus by id, preserving manifest order."""
        if not ids:
            return self.papers
        wanted = set(ids)
        unknown = wanted - {p.id for p in self.papers}
        if unknown:
            raise ManifestError(f"unknown paper ids: {sorted(unknown)}")
        return tuple(p for p in self.papers if p.id in wanted)

    @property
    def topics(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for p in self.papers:
            seen.setdefault(p.topic, None)
        return tuple(seen)

    def with_papers(self, papers: Sequence[Paper]) -> Manifest:
        return Manifest(version=self.version, source_repo=self.source_repo, papers=tuple(papers))


def load_manifest(path: Path | str) -> Manifest:
    """Parse and validate corpus.yaml.

    Validation is strict on purpose: a duplicate id or a malformed digest silently
    produces a corpus that does not match the one an eval run claims to use.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(f"{path} must contain a mapping at the top level")

    entries = raw.get("papers")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{path} must define a non-empty 'papers' list")

    papers: list[Paper] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        papers.append(_parse_paper(entry, index=i, seen_ids=seen_ids))

    return Manifest(
        version=int(raw.get("version", 1)),
        source_repo=str(raw.get("source_repo", "")),
        papers=tuple(papers),
    )


def _parse_paper(entry: object, *, index: int, seen_ids: set[str]) -> Paper:
    where = f"papers[{index}]"
    if not isinstance(entry, dict):
        raise ManifestError(f"{where} must be a mapping")

    missing = [k for k in ("id", "title", "url", "topic", "license") if not entry.get(k)]
    if missing:
        raise ManifestError(f"{where} is missing required fields: {missing}")

    paper_id = str(entry["id"])
    if not _ID_RE.match(paper_id):
        raise ManifestError(
            f"{where}: id {paper_id!r} must be lowercase alphanumeric with - or _, 2-64 chars"
        )
    if paper_id in seen_ids:
        raise ManifestError(f"{where}: duplicate id {paper_id!r}")
    seen_ids.add(paper_id)

    url = str(entry["url"])
    if not url.startswith("https://"):
        raise ManifestError(f"{where}: url must be https, got {url!r}")

    raw_kind = str(entry.get("kind") or "pdf")
    if raw_kind not in _KINDS:
        raise ManifestError(f"{where}: kind must be one of {sorted(_KINDS)}, got {raw_kind!r}")

    digest = entry.get("sha256")
    if digest is not None:
        digest = str(digest).lower()
        if not _SHA256_RE.match(digest):
            raise ManifestError(f"{where}: sha256 must be 64 lowercase hex chars")

    return Paper(
        id=paper_id,
        title=str(entry["title"]),
        url=url,
        topic=str(entry["topic"]),
        license=str(entry["license"]),
        kind=cast(PaperKind, raw_kind),  # narrowed by the membership check above
        arxiv_id=str(entry["arxiv_id"]) if entry.get("arxiv_id") else None,
        sha256=digest,
        notes=str(entry.get("notes", "")),
    )


def save_manifest(manifest: Manifest, path: Path | str) -> None:
    """Write the manifest back, preserving field order for a readable diff.

    Used after a first fetch to pin digests. Round-tripping through this function
    is what turns 'whatever arxiv served today' into a reproducible corpus.
    """
    payload = {
        "version": manifest.version,
        "source_repo": manifest.source_repo,
        "papers": [
            {
                k: v
                for k, v in (
                    ("id", p.id),
                    ("title", p.title),
                    # "pdf" is the default and is omitted, so the existing arXiv
                    # entries round-trip without a corpus-wide diff.
                    ("kind", p.kind if p.kind != "pdf" else None),
                    ("arxiv_id", p.arxiv_id),
                    ("topic", p.topic),
                    ("url", p.url),
                    ("license", p.license),
                    ("sha256", p.sha256),
                    ("notes", p.notes or None),
                )
                if v is not None
            }
            for p in manifest.papers
        ],
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))
    tmp.replace(path)  # atomic, so a crash mid-pin never truncates the corpus manifest
