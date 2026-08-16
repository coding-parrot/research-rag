"""The vector-store seam plus the shared chunk store.

Chunks live in one place regardless of which index is in front of them. Both the
dense and the lexical index return chunk ids and scores; the store resolves ids back
to `Chunk` objects. That keeps the two indexes independently swappable and means a
fusion step only ever manipulates ids.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.domain import Chunk
from rag.embed.base import Matrix, Vector
from rag.errors import IndexError_


@dataclass(frozen=True, slots=True)
class Hit:
    """One index result. Deliberately id-only so fusion never copies chunk text."""

    chunk_id: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Dense similarity search over chunk embeddings."""

    @property
    def size(self) -> int: ...

    def add(self, chunk_ids: Sequence[str], vectors: Matrix) -> None: ...

    def search(self, query: Vector, k: int) -> list[Hit]: ...

    def save(self, directory: Path) -> None: ...

    @classmethod
    def load(cls, directory: Path) -> VectorStore: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Keyword search. Catches exact terms that embeddings blur together.

    On a research corpus this matters more than usual: 'FlashAttention-2' and
    'FlashAttention' are near-identical to an embedding model and completely
    different papers to a reader.
    """

    @property
    def size(self) -> int: ...

    def add(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None: ...

    def search(self, query: str, k: int) -> list[Hit]: ...

    def save(self, directory: Path) -> None: ...


class ChunkStore:
    """Id-to-chunk mapping, persisted as JSON Lines.

    JSONL rather than pickle: readable, diffable, and safe to load from a directory
    you did not write yourself.
    """

    FILENAME = "chunks.jsonl"

    def __init__(self, chunks: Iterable[Chunk] = ()) -> None:
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._by_id.values())

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self._by_id[chunk.chunk_id] = chunk

    def get(self, chunk_id: str) -> Chunk:
        try:
            return self._by_id[chunk_id]
        except KeyError as exc:
            raise IndexError_(
                f"chunk {chunk_id!r} is in the index but not the chunk store; "
                "the index and store are out of sync, rebuild both"
            ) from exc

    def resolve(self, hits: Sequence[Hit]) -> list[tuple[Chunk, float]]:
        """Hits to (chunk, score), skipping ids the store does not know.

        Skipping rather than raising because a stale index entry should degrade one
        result, not fail the request. The count is logged by the retriever.
        """
        resolved: list[tuple[Chunk, float]] = []
        for hit in hits:
            chunk = self._by_id.get(hit.chunk_id)
            if chunk is not None:
                resolved.append((chunk, hit.score))
        return resolved

    def by_doc(self) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in self._by_id.values():
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        return grouped

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.FILENAME
        with path.open("w", encoding="utf-8") as fh:
            for chunk in self._by_id.values():
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> ChunkStore:
        path = Path(directory) / cls.FILENAME
        if not path.exists():
            raise IndexError_(f"no chunk store at {path}; run `rag index` first")
        chunks = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))
        return cls(chunks)
