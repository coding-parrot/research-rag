"""Rank fusion and diversity.

BM25 scores and cosine scores are not comparable: one is unbounded and corpus
dependent, the other lives in [-1, 1]. Normalising them onto a shared scale means
inventing a weighting nobody can justify. Reciprocal rank fusion sidesteps that
entirely by using only rank position, which is why it is the default here.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain import Chunk
from rag.index.base import Hit

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """Fuse ranked lists by summing 1 / (k + rank).

    `k` damps the head of each list: a large k makes the fusion more egalitarian,
    a small k lets a single first-place finish dominate. 60 is the value from the
    original paper and behaves well without tuning.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(rankings)} rankings but {len(weights)} weights")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Hit(chunk_id=chunk_id, score=score) for chunk_id, score in ordered]


def deduplicate(
    scored: Sequence[tuple[Chunk, float]], *, threshold: float = 0.75
) -> list[tuple[Chunk, float]]:
    """Drop chunks that are near-verbatim copies of, or fully contained in, a kept chunk.

    Similarity is containment (shared shingles over the smaller set), not Jaccard:
    Jaccard punishes length differences, so a chunk quoted whole inside a longer one
    scores ~0.35 and never trips a high threshold, while containment scores it 1.0.
    Adjacent parts of a split section share only their construction overlap
    (containment ~0.2) and are deliberately retained: the rest of each part is
    unique content.
    """
    kept: list[tuple[Chunk, float]] = []
    seen_shingles: list[frozenset[str]] = []

    for chunk, score in scored:
        shingles = _shingles(chunk.text)
        if any(_containment(shingles, other) >= threshold for other in seen_shingles):
            continue
        kept.append((chunk, score))
        seen_shingles.append(shingles)
    return kept


def cap_per_document(
    scored: Sequence[tuple[Chunk, float]], *, max_per_doc: int
) -> list[tuple[Chunk, float]]:
    """Keep at most N chunks from any one paper, preserving order.

    Without this, a query that happens to match one paper's vocabulary fills the
    whole context with that paper, and a cross-paper question becomes unanswerable
    even though the right chunks were ranked 5th and 6th.
    """
    counts: dict[str, int] = {}
    kept: list[tuple[Chunk, float]] = []
    for chunk, score in scored:
        seen = counts.get(chunk.doc_id, 0)
        if seen >= max_per_doc:
            continue
        counts[chunk.doc_id] = seen + 1
        kept.append((chunk, score))
    return kept


def _shingles(text: str, size: int = 5) -> frozenset[str]:
    """Word-level shingles, the cheap way to compare texts of different lengths."""
    words = text.lower().split()
    if len(words) < size:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


def _containment(left: frozenset[str], right: frozenset[str]) -> float:
    """Fraction of the smaller shingle set that appears in the larger one."""
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
