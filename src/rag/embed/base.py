"""The embedding seam.

The embedding model is a contract: the same model must embed the chunks at index
time and the query at retrieval time. Swap it without reindexing and the index is
silently dead, returning plausible-looking nonsense. `Embedder.fingerprint` exists so
the index can refuse to load under a different model rather than fail quietly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Text to fixed-length vectors."""

    @property
    def dimension(self) -> int: ...

    @property
    def fingerprint(self) -> str:
        """Identity of (model, dimension, normalisation). Stored with the index."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """Batch-embed chunk texts. Shape (len(texts), dimension)."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a single query. Shape (dimension,)."""
        ...


def make_fingerprint(model: str, dimension: int, normalized: bool) -> str:
    raw = f"{model}|{dimension}|{int(normalized)}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def l2_normalize(matrix: Matrix) -> Matrix:
    """Row-wise L2 normalisation, safe on zero vectors.

    With normalised vectors, cosine similarity is a plain dot product, which is what
    lets the FAISS inner-product index and the in-memory store agree exactly.
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def cosine_similarity(query: Vector, matrix: Matrix) -> NDArray[np.float32]:
    """Cosine similarity of one query against every row."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    result: NDArray[np.float32] = np.asarray(
        matrix @ query / np.maximum(denominator, 1e-12), dtype=np.float32
    )
    return result
