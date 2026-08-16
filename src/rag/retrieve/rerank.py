"""Reranking: stage two of two-stage retrieval.

The vector store returns a wide, cheap candidate pool. The reranker reads the query
and each candidate *together* with a cross-encoder, which is strictly more
information than comparing two pre-computed embeddings can use.

Default is a local cross-encoder rather than a hosted API. The eval matrix reranks
thousands of (query, chunk) pairs per sweep; a per-call bill turns the sweep into
something you run once and then stop running.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from rag.config import RetrieveConfig
from rag.domain import Chunk
from rag.observability import get_logger, timed

log = get_logger("rerank")


@runtime_checkable
class Reranker(Protocol):
    """Re-score (query, chunk) pairs and return the best `top_k`."""

    @property
    def name(self) -> str: ...

    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]], top_k: int
    ) -> list[tuple[Chunk, float]]: ...


class NoopReranker:
    """Passthrough. Keeps the retriever free of `if reranker is not None` branches."""

    @property
    def name(self) -> str:
        return "none"

    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]], top_k: int
    ) -> list[tuple[Chunk, float]]:
        return list(candidates[:top_k])


class CrossEncoderReranker:
    """Local sentence-transformers cross-encoder."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._model_name}"

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise ImportError("pip install -e '.[embed]' to use CrossEncoderReranker") from exc
        with timed(log, "rerank.load", model=self._model_name):
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]], top_k: int
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            ((chunk, float(score)) for (chunk, _), score in zip(candidates, scores, strict=True)),
            key=lambda pair: -pair[1],
        )
        return ranked[:top_k]


class CohereReranker:
    """Hosted reranker.

    The API key comes from the environment only. It is never a constructor default,
    never read from a config file, and never logged.
    """

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY is not set; export it or use the cross-encoder reranker"
            )
        self._model = model
        self._client: Any | None = None
        self._api_key = api_key

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise ImportError("pip install -e '.[rerank]' to use CohereReranker") from exc
        self._client = cohere.Client(self._api_key)
        return self._client

    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]], top_k: int
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        client = self._load()
        response = client.rerank(
            model=self._model,
            query=query,
            documents=[chunk.text for chunk, _ in candidates],
            top_n=min(top_k, len(candidates)),
        )
        return [(candidates[r.index][0], float(r.relevance_score)) for r in response.results]


class StubReranker:
    """Test double that reverses the candidate order.

    Reversing rather than shuffling makes it obvious in an assertion whether the
    reranker actually ran, without needing a model.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "stub"

    def rerank(
        self, query: str, candidates: Sequence[tuple[Chunk, float]], top_k: int
    ) -> list[tuple[Chunk, float]]:
        self.calls.append(query)
        reversed_pairs = list(reversed(candidates))
        return [(chunk, 1.0 - i * 0.01) for i, (chunk, _) in enumerate(reversed_pairs)][:top_k]


def build_reranker(config: RetrieveConfig, *, cohere_api_key: str | None = None) -> Reranker:
    if not config.rerank or config.reranker == "none":
        return NoopReranker()
    if config.reranker == "cohere":
        return CohereReranker(config.cohere_rerank_model, cohere_api_key or "")
    return CrossEncoderReranker(config.reranker_model)
