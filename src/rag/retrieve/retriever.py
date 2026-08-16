"""The retriever.

Retrieval is the ceiling on RAG quality. If the right section does not come back in
the top-k, no amount of prompt engineering recovers it, because the model has no way
to know what it is missing. Everything here exists to raise that ceiling.

Order of operations:

    query -> transform -> dense search (+ BM25) -> fuse -> dedup -> rerank -> cap -> top-k

Dedup runs before rerank so the reranker never spends its budget scoring two copies
of the same text. The per-document cap runs *after* rerank so the cap applies to the
final ordering rather than to the candidate pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.config import GuardrailConfig, RetrieveConfig
from rag.domain import Scored
from rag.embed.base import Embedder
from rag.index.base import ChunkStore, Hit, LexicalIndex, VectorStore
from rag.observability import get_logger
from rag.retrieve.fusion import cap_per_document, deduplicate, reciprocal_rank_fusion
from rag.retrieve.rerank import Reranker
from rag.retrieve.rewrite import QueryTransform, RewriteResult

log = get_logger("retrieve")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What retrieval returns, with enough detail to debug a bad answer."""

    results: tuple[Scored, ...]
    rewrite: RewriteResult
    candidates_considered: int
    dense_hits: int
    lexical_hits: int
    deduped: int
    reranker: str
    missing_from_store: int = 0
    # Best dense cosine similarity for the ORIGINAL query only, before fusion. This
    # is the calibrated relevance signal: post-fusion scores are rank-based (RRF) or
    # unbounded logits (cross-encoder), and a threshold on either is meaningless.
    # Rewrites are excluded on purpose: a HyDE paragraph is invented text generated
    # to resemble the corpus, so its cosine clears any floor regardless of whether
    # the user's question is answerable. None means dense search produced no ranking
    # at all (empty vector store), so no calibrated signal exists.
    top_dense_score: float | None = None

    @property
    def top_score(self) -> float:
        return self.results[0].score if self.results else 0.0

    @property
    def is_empty(self) -> bool:
        return not self.results

    @property
    def doc_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for scored in self.results:
            seen.setdefault(scored.chunk.doc_id, None)
        return tuple(seen)


class Retriever:
    """Hybrid retrieval with optional query transform and reranking."""

    def __init__(
        self,
        *,
        store: ChunkStore,
        vectors: VectorStore,
        embedder: Embedder,
        config: RetrieveConfig,
        guardrails: GuardrailConfig,
        lexical: LexicalIndex | None = None,
        transform: QueryTransform,
        reranker: Reranker,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._config = config
        self._guardrails = guardrails
        self._lexical = lexical
        self._transform = transform
        self._reranker = reranker

    def retrieve(self, query: str) -> RetrievalResult:
        rewrite = self._transform.transform(query)

        dense_rankings = [self._dense(q) for q in rewrite.queries]
        lexical_rankings = self._lexical_rankings(query, rewrite)
        top_dense = self._original_dense_top(query, rewrite, dense_rankings)

        # Weight the dense block so it carries constant total weight against the
        # single lexical ranking: under multi-query, N equally weighted dense lists
        # would otherwise dilute BM25 evidence N-fold as a side effect of the
        # rewrite strategy, confounding the strategy ablation.
        nonempty_dense = [r for r in dense_rankings if r]
        nonempty_lexical = [r for r in lexical_rankings if r]
        rankings = [*nonempty_dense, *nonempty_lexical]
        # max(len, 1): with a broken dense index every dense ranking is empty and
        # retrieval proceeds on lexical alone, which must not divide by zero.
        dense_weight = 1.0 / max(len(nonempty_dense), 1)
        weights = [dense_weight] * len(nonempty_dense) + [1.0] * len(nonempty_lexical)
        if not rankings:
            return RetrievalResult(
                results=(),
                rewrite=rewrite,
                candidates_considered=0,
                dense_hits=0,
                lexical_hits=0,
                deduped=0,
                reranker=self._reranker.name,
                top_dense_score=top_dense,
            )

        fused = reciprocal_rank_fusion(rankings, k=self._config.rrf_k, weights=weights)
        resolved = self._store.resolve(fused[: self._config.fetch_k])
        missing = len(fused[: self._config.fetch_k]) - len(resolved)
        if missing:
            log.warning(
                "index references chunks the store does not have",
                fields={"missing": missing, "hint": "rebuild the index"},
            )

        before_dedup = len(resolved)
        deduped = deduplicate(resolved, threshold=self._guardrails.dedup_threshold)

        # Rerank the entire deduped pool (bounded by fetch_k) so the per-document
        # cap always has candidates to fall back on: a narrower rerank window can
        # be filled by a single dominating paper, and capping it would then return
        # fewer than top_k results even though cap-valid chunks from other papers
        # sit just past the window.
        reranked = self._reranker.rerank(query, deduped, top_k=len(deduped))
        capped = cap_per_document(reranked, max_per_doc=self._config.max_per_doc)
        final = capped[: self._config.top_k]

        results = tuple(
            Scored(
                chunk=chunk,
                score=score,
                rank=rank,
                retriever=f"{rewrite.strategy}+{self._reranker.name}",
            )
            for rank, (chunk, score) in enumerate(final, start=1)
        )

        log.info(
            "retrieved",
            fields={
                "strategy": rewrite.strategy,
                "queries": len(rewrite.queries),
                "candidates": before_dedup,
                "returned": len(results),
                "top_score": round(results[0].score, 4) if results else 0.0,
                "docs": len(set(s.chunk.doc_id for s in results)),
            },
        )
        return RetrievalResult(
            results=results,
            rewrite=rewrite,
            candidates_considered=before_dedup,
            dense_hits=sum(len(r) for r in dense_rankings),
            lexical_hits=sum(len(r) for r in lexical_rankings),
            deduped=before_dedup - len(deduped),
            reranker=self._reranker.name,
            missing_from_store=missing,
            top_dense_score=top_dense,
        )

    # ------------------------------------------------------------------ #

    def _dense(self, query: str) -> list[Hit]:
        vector = self._embedder.embed_query(query)
        return self._vectors.search(vector, k=self._config.fetch_k)

    def _original_dense_top(
        self, query: str, rewrite: RewriteResult, dense_rankings: list[list[Hit]]
    ) -> float | None:
        """Best dense cosine for the user's own words, the relevance-floor signal.

        Located by matching the query string, not by position: HyDE deliberately
        puts the hypothetical first, so `rewrite.queries[0]` can be invented text.
        Every current transform keeps the original among its queries; the extra
        search is a fail-closed path for a future transform that drops it.
        """
        for rewritten, ranking in zip(rewrite.queries, dense_rankings, strict=True):
            if rewritten == query:
                return max((hit.score for hit in ranking), default=None)
        ranking = self._dense(query)
        return max((hit.score for hit in ranking), default=None)

    def _lexical_rankings(self, original: str, rewrite: RewriteResult) -> list[list[Hit]]:
        """BM25 over the user's actual words.

        Deliberately not over the rewrites: a HyDE paragraph is invented text, and
        keyword-matching against invented terms is how a hallucinated model name
        ends up steering retrieval.
        """
        if self._lexical is None or self._lexical.size == 0:
            return []
        del rewrite
        hits = self._lexical.search(original, k=self._config.fetch_k)
        return [hits] if hits else []


def to_scored(
    pairs: Sequence[tuple[object, float]], retriever: str = "manual"
) -> tuple[Scored, ...]:
    """Helper for tests and eval fixtures that build results by hand."""
    from rag.domain import Chunk

    return tuple(
        Scored(chunk=chunk, score=score, rank=i, retriever=retriever)
        for i, (chunk, score) in enumerate(pairs, start=1)
        if isinstance(chunk, Chunk)
    )
