"""Disk cache for OCR output.

Surya takes minutes per paper. Without this, every re-chunk, re-index or eval run
would pay for it again, and the whole ablation loop would be unusable.

Cache key is (pdf sha256, engine, engine version, ocr config hash). All four matter:
a Surya upgrade or a DPI change must invalidate, but a retrieval config change must not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag.config import OcrConfig
from rag.domain import BBox, Block, BlockType, OcrDocument
from rag.ingest.ocr.base import OcrEngine
from rag.observability import get_logger

log = get_logger("ocr.cache")

CACHE_FORMAT_VERSION = 1


class CachedOcrEngine:
    """Wraps any `OcrEngine` with a JSON cache. Also an `OcrEngine`."""

    def __init__(self, inner: OcrEngine, cache_dir: Path, config: OcrConfig) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self._config = config
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    def read(self, pdf_path: Path, doc_id: str) -> OcrDocument:
        if not self._config.cache:
            return self._inner.read(pdf_path, doc_id)

        path = self._cache_path(pdf_path, doc_id)
        if path.exists():
            try:
                document = load_ocr_document(path)
                self.hits += 1
                log.debug("ocr cache hit", fields={"doc_id": doc_id, "path": str(path)})
                return document
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning(
                    "ocr cache entry unreadable, recomputing",
                    fields={"doc_id": doc_id, "error": str(exc)},
                )

        self.misses += 1
        document = self._inner.read(pdf_path, doc_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        save_ocr_document(document, path)
        return document

    def _cache_path(self, pdf_path: Path, doc_id: str) -> Path:
        key = _cache_key(pdf_path, self._inner.name, self._inner.version, self._config)
        return self._dir / f"{doc_id}.{key}.json"


def _cache_key(pdf_path: Path, engine: str, engine_version: str, config: OcrConfig) -> str:
    from rag.ingest.fetch import sha256_file

    payload = {
        "format": CACHE_FORMAT_VERSION,
        "pdf": sha256_file(pdf_path),
        "engine": engine,
        "engine_version": engine_version,
        # batch_size is excluded: it only changes how pages are grouped per predictor
        # call, never the recognised output, so bumping it must not re-OCR the corpus
        # (config.ingest_hash excludes it for the same reason). engine is excluded as
        # redundant: the engine/engine_version payload fields above carry identity.
        "config": config.model_dump(mode="json", exclude={"cache", "batch_size", "engine"}),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Serialisation. Kept explicit rather than pickled so cache files are readable,
# diffable, and safe to load from a shared directory.
# --------------------------------------------------------------------------- #


def save_ocr_document(document: OcrDocument, path: Path) -> None:
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "doc_id": document.doc_id,
        "source_path": document.source_path,
        "page_count": document.page_count,
        "engine": document.engine,
        "engine_version": document.engine_version,
        "blocks": [
            {
                "type": block.type.value,
                "text": block.text,
                "page": block.page,
                "order": block.order,
                "bbox": asdict(block.bbox) if block.bbox else None,
                "confidence": round(block.confidence, 4),
            }
            for block in document.blocks
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(path)  # atomic, so a crash mid-write never leaves a half cache entry


def load_ocr_document(path: Path) -> OcrDocument:
    raw: dict[str, Any] = json.loads(path.read_text())
    if raw.get("format") != CACHE_FORMAT_VERSION:
        raise ValueError(f"cache format {raw.get('format')} != {CACHE_FORMAT_VERSION}")

    blocks = tuple(
        Block(
            type=BlockType(entry["type"]),
            text=entry["text"],
            page=int(entry["page"]),
            order=int(entry["order"]),
            bbox=BBox(**entry["bbox"]) if entry.get("bbox") else None,
            confidence=float(entry.get("confidence", 1.0)),
        )
        for entry in raw["blocks"]
    )
    return OcrDocument(
        doc_id=raw["doc_id"],
        source_path=raw["source_path"],
        page_count=int(raw["page_count"]),
        blocks=blocks,
        engine=raw["engine"],
        engine_version=raw["engine_version"],
    )
