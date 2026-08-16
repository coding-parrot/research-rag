"""Vector and lexical index implementations.

`InMemoryVectorStore` is the reference: exact brute-force cosine, no dependencies,
used by every unit test. `FaissVectorStore` is the same semantics at speed. Both
must agree on ordering, and a test asserts that they do, which is what keeps the
fast path honest.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag.embed.base import Matrix, Vector, l2_normalize
from rag.errors import IndexError_
from rag.index.base import Hit
from rag.observability import get_logger

log = get_logger("index")

_IDS_FILE = "ids.json"
_VECTORS_FILE = "vectors.npy"
_FAISS_FILE = "index.faiss"
_META_FILE = "index_meta.json"


class InMemoryVectorStore:
    """Exact cosine search over a dense matrix.

    Linear in corpus size, which is entirely fine at a few thousand chunks and is
    the correct default for a corpus of this scale. It is also the oracle the FAISS
    implementation is tested against.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._ids: list[str] = []
        self._vectors = np.zeros((0, dimension), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, chunk_ids: Sequence[str], vectors: Matrix) -> None:
        if len(chunk_ids) != len(vectors):
            raise IndexError_(f"{len(chunk_ids)} ids but {len(vectors)} vectors")
        if len(vectors) == 0:
            return
        if vectors.shape[1] != self._dimension:
            raise IndexError_(f"expected {self._dimension}-dim vectors, got {vectors.shape[1]}")
        self._ids.extend(chunk_ids)
        self._vectors = np.vstack(
            [self._vectors, l2_normalize(np.asarray(vectors, dtype=np.float32))]
        )

    def search(self, query: Vector, k: int) -> list[Hit]:
        if self.size == 0:
            return []
        normalized = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self._vectors @ normalized.astype(np.float32)
        top = np.argsort(-scores)[: min(k, self.size)]
        return [Hit(chunk_id=self._ids[i], score=float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _IDS_FILE).write_text(json.dumps(self._ids))
        np.save(directory / _VECTORS_FILE, self._vectors)
        (directory / _META_FILE).write_text(
            json.dumps({"kind": "inmemory", "dimension": self._dimension})
        )

    @classmethod
    def load(cls, directory: Path) -> InMemoryVectorStore:
        directory = Path(directory)
        meta_path = directory / _META_FILE
        if not meta_path.exists():
            raise IndexError_(f"no index metadata at {meta_path}")
        meta = json.loads(meta_path.read_text())
        store = cls(dimension=int(meta["dimension"]))
        store._ids = json.loads((directory / _IDS_FILE).read_text())
        store._vectors = np.load(directory / _VECTORS_FILE).astype(np.float32)
        return store


class FaissVectorStore:
    """FAISS `IndexFlatIP` over L2-normalised vectors, which is exact cosine.

    Flat rather than HNSW or IVF deliberately: at a few thousand chunks an
    approximate index buys nothing measurable and introduces a recall parameter
    that would silently confound every retrieval eval number in the report.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._ids: list[str] = []
        self._index: Any = _new_faiss_index(dimension)

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, chunk_ids: Sequence[str], vectors: Matrix) -> None:
        if len(chunk_ids) != len(vectors):
            raise IndexError_(f"{len(chunk_ids)} ids but {len(vectors)} vectors")
        if len(vectors) == 0:
            return
        self._index.add(l2_normalize(np.asarray(vectors, dtype=np.float32)))
        self._ids.extend(chunk_ids)

    def search(self, query: Vector, k: int) -> list[Hit]:
        if self.size == 0:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        vector = l2_normalize(vector)
        scores, indices = self._index.search(vector, min(k, self.size))
        return [
            Hit(chunk_id=self._ids[int(i)], score=float(s))
            for s, i in zip(scores[0], indices[0], strict=True)
            if i >= 0
        ]

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / _FAISS_FILE))
        (directory / _IDS_FILE).write_text(json.dumps(self._ids))
        (directory / _META_FILE).write_text(
            json.dumps({"kind": "faiss", "dimension": self._dimension})
        )

    @classmethod
    def load(cls, directory: Path) -> FaissVectorStore:
        import faiss

        directory = Path(directory)
        meta_path = directory / _META_FILE
        if not meta_path.exists():
            raise IndexError_(f"no index metadata at {meta_path}")
        meta = json.loads(meta_path.read_text())
        store = cls(dimension=int(meta["dimension"]))
        store._index = faiss.read_index(str(directory / _FAISS_FILE))
        store._ids = json.loads((directory / _IDS_FILE).read_text())
        return store


def _new_faiss_index(dimension: int) -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - install-dependent
        raise ImportError("pip install -e '.[index]' to use FaissVectorStore") from exc
    return faiss.IndexFlatIP(dimension)


class Bm25Index:
    """BM25 over chunk text.

    The lexical half of hybrid retrieval. Its job is exact-term recall: model names,
    dataset names, hyperparameters, and acronyms, all of which embeddings smooth over.

    Uses BM25+ rather than Okapi: Okapi's IDF is zero for a term that appears in
    half the documents, which makes it useless on small corpora (and this corpus is
    small by BM25 standards). BM25+ keeps IDF positive. Because BM25+ also gives a
    small floor score to documents that lack the term entirely, hits are explicitly
    masked to documents sharing at least one query token.
    """

    FILENAME = "bm25.json"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._token_sets: list[frozenset[str]] = []
        self._model: Any | None = None

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise IndexError_(f"{len(chunk_ids)} ids but {len(texts)} texts")
        self._ids.extend(chunk_ids)
        self._texts.extend(texts)
        self._token_sets.extend(frozenset(tokenize(t)) for t in texts)
        self._model = None  # rebuilt lazily on the next search

    def _build(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from rank_bm25 import BM25Plus
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise ImportError("rank-bm25 is required for Bm25Index") from exc
        self._model = BM25Plus([tokenize(t) for t in self._texts])
        return self._model

    def search(self, query: str, k: int) -> list[Hit]:
        if self.size == 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        query_set = set(tokens)
        scores = np.asarray(self._build().get_scores(tokens), dtype=np.float32)
        # Mask out documents that contain none of the query tokens: BM25+'s floor
        # would otherwise rank pure non-matches.
        for i, token_set in enumerate(self._token_sets):
            if not (token_set & query_set):
                scores[i] = float("-inf")
        top = np.argsort(-scores)[: min(k, self.size)]
        return [
            Hit(chunk_id=self._ids[i], score=float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.FILENAME).write_text(
            json.dumps({"ids": self._ids, "texts": self._texts}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, directory: Path) -> Bm25Index:
        path = Path(directory) / cls.FILENAME
        if not path.exists():
            raise IndexError_(f"no BM25 index at {path}")
        payload = json.loads(path.read_text())
        index = cls()
        index.add(list(payload["ids"]), list(payload["texts"]))
        return index


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping internal hyphens and dots.

    Keeping them matters here: 'gpt-4', 'bge-m3' and 'v100' are exactly the tokens
    BM25 is in the pipeline to catch, and a naive `\\w+` split shreds them.
    """
    import re

    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())
