"""Embedder implementations: real, cached, and fake."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag.config import EmbedConfig
from rag.embed.base import Embedder, Matrix, Vector, l2_normalize, make_fingerprint
from rag.observability import get_logger, timed

log = get_logger("embed")


class SentenceTransformerEmbedder:
    """Local sentence-transformers model.

    Local rather than hosted on purpose: the eval matrix re-embeds the corpus on
    every configuration sweep, and a per-token embedding bill would make that
    sweep something you avoid running.
    """

    def __init__(self, config: EmbedConfig) -> None:
        self._config = config
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def fingerprint(self) -> str:
        return make_fingerprint(self._config.model, self._config.dimension, self._config.normalize)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise ImportError(
                "pip install -e '.[embed]' to use SentenceTransformerEmbedder"
            ) from exc

        with timed(log, "embed.load", model=self._config.model):
            model = SentenceTransformer(self._config.model)

        actual = int(model.get_sentence_embedding_dimension())
        if actual != self._config.dimension:
            raise ValueError(
                f"{self._config.model} produces {actual}-dim vectors but config says "
                f"{self._config.dimension}. Fix embed.dimension and rebuild the index."
            )
        self._model = model
        return model

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
            normalize_embeddings=self._config.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector


class CachedEmbedder:
    """Wraps an `Embedder` with a content-addressed on-disk cache.

    Re-chunking usually changes only a handful of chunks. Without this, one changed
    heading threshold re-embeds the whole corpus.
    """

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self._inner = inner
        self._dir = Path(cache_dir) / inner.fingerprint
        self._dir.mkdir(parents=True, exist_ok=True)
        self._store = self._dir / "vectors.npz"
        self._cache: dict[str, np.ndarray] = self._load_cache()
        self.hits = 0
        self.misses = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self._store.exists():
            return {}
        try:
            with np.load(self._store) as data:
                return {key: data[key] for key in data.files}
        except Exception as exc:
            log.warning("embedding cache unreadable, starting fresh", fields={"error": str(exc)})
            return {}

    def flush(self) -> None:
        if self._cache:
            np.savez_compressed(self._store, **self._cache)  # type: ignore[arg-type]

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        keys = [_text_key(t) for t in texts]
        missing = [
            (i, t) for i, (k, t) in enumerate(zip(keys, texts, strict=True)) if k not in self._cache
        ]

        if missing:
            fresh = self._inner.embed_documents([t for _, t in missing])
            for (index, _), vector in zip(missing, fresh, strict=True):
                self._cache[keys[index]] = vector
            self.misses += len(missing)
        self.hits += len(texts) - len(missing)

        return np.stack([self._cache[k] for k in keys]).astype(np.float32)

    def embed_query(self, text: str) -> Vector:
        # Queries are not cached: they are unique by nature and caching them would
        # just grow the file forever.
        return self._inner.embed_query(text)


class FakeEmbedder:
    """Deterministic hash-based embedder for tests.

    Not semantically meaningful, but stable and cheap, and similar strings do land
    near each other because the token hash contributions overlap. That is enough to
    test ranking mechanics without loading a model.
    """

    def __init__(self, dimension: int = 32, *, normalize: bool = True) -> None:
        self._dimension = dimension
        self._normalize = normalize
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fingerprint(self) -> str:
        return make_fingerprint("fake", self._dimension, self._normalize)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.call_count += 1
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        matrix = np.stack([self._vector(t) for t in texts])
        return l2_normalize(matrix) if self._normalize else matrix

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector

    def _vector(self, text: str) -> "Vector":
        """Bag-of-tokens projected into the vector space by stable hashing."""
        vector = np.zeros(self._dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            vector[0] = 1.0
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def build_embedder(config: EmbedConfig, cache_dir: Path | None = None) -> Embedder:
    """Construct the configured embedder, wrapped in a cache when one is available."""
    inner: Embedder
    if config.provider == "fake":
        inner = FakeEmbedder(dimension=config.dimension, normalize=config.normalize)
    else:
        inner = SentenceTransformerEmbedder(config)

    if config.cache and cache_dir is not None:
        return CachedEmbedder(inner, cache_dir)
    return inner


def embedder_manifest(embedder: Embedder) -> str:
    """Serialised identity written next to an index."""
    return json.dumps({"fingerprint": embedder.fingerprint, "dimension": embedder.dimension})
