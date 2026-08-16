"""Query transforms.

Both exist for the same reason: a query and the passage that answers it are often
not in the same vector neighbourhood. A user asks "can I be banned?" and the paper
says "termination of access". Same meaning, different vectors.

  MULTI-QUERY  Paraphrase the question N ways, retrieve each, fuse the results.
               Attacks surface-form sensitivity. Costs one LLM call plus N searches.
  HYDE         Ask the model to invent an answer, then embed *that*. An answer-shaped
               string sits closer to real answer-shaped chunks than a question does.
               Costs one LLM call plus one search.

Both are measured against vanilla in the eval matrix rather than assumed to help.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag.generate.client import LlmClient, LlmRequest
from rag.observability import get_logger

log = get_logger("rewrite")

MULTI_QUERY_PROMPT = """\
Rewrite the question below as {count} alternative search queries for a corpus of \
machine learning research papers.

Rules:
- Output exactly {count} lines, one query per line.
- No numbering, no bullets, no preamble, no explanation.
- Vary the vocabulary: use the technical terms a paper would use, not only the \
words in the question.
- Each line must stand alone as a search query.

Question: {question}"""

HYDE_PROMPT = """\
Write a single short paragraph, in the style of a machine learning research paper, \
that would plausibly answer the question below.

Write it as though it were an excerpt from a paper. Do not hedge, do not say you \
are unsure, and do not mention that this is hypothetical. Accuracy does not matter; \
what matters is that the paragraph reads like a passage from a paper.

Question: {question}"""


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Queries to search with, plus what produced them."""

    queries: tuple[str, ...]
    strategy: str
    llm_calls: int = 0

    @property
    def primary(self) -> str:
        return self.queries[0]


@runtime_checkable
class QueryTransform(Protocol):
    @property
    def name(self) -> str: ...

    def transform(self, query: str) -> RewriteResult: ...


class IdentityTransform:
    """Vanilla retrieval: search the question as written."""

    @property
    def name(self) -> str:
        return "vanilla"

    def transform(self, query: str) -> RewriteResult:
        return RewriteResult(queries=(query,), strategy="vanilla")


class MultiQueryTransform:
    """Paraphrase the query N ways and search all of them."""

    def __init__(
        self, client: LlmClient, count: int = 3, model: str = "", effort: str = "low"
    ) -> None:
        self._client = client
        self._count = count
        self._model = model
        self._effort = effort

    @property
    def name(self) -> str:
        return "multi_query"

    def transform(self, query: str) -> RewriteResult:
        prompt = MULTI_QUERY_PROMPT.format(count=self._count, question=query)
        try:
            response = self._client.complete(
                LlmRequest(prompt=prompt, model=self._model, effort=self._effort, max_tokens=512)
            )
        except Exception as exc:
            # A rewrite failure must degrade to vanilla retrieval, never fail the
            # request. The user asked a question, not for a paraphrase.
            log.warning(
                "multi-query rewrite failed, falling back to vanilla", fields={"error": str(exc)}
            )
            return RewriteResult(queries=(query,), strategy="multi_query:fallback")

        paraphrases = parse_query_list(response.text, limit=self._count)
        # The original always leads: a bad paraphrase must not be able to displace
        # the question the user actually asked.
        queries = (query, *(p for p in paraphrases if p.lower() != query.lower()))
        return RewriteResult(queries=queries, strategy="multi_query", llm_calls=1)


class HydeTransform:
    """Embed a hypothetical answer instead of the question."""

    def __init__(self, client: LlmClient, model: str = "", effort: str = "low") -> None:
        self._client = client
        self._model = model
        self._effort = effort

    @property
    def name(self) -> str:
        return "hyde"

    def transform(self, query: str) -> RewriteResult:
        prompt = HYDE_PROMPT.format(question=query)
        try:
            response = self._client.complete(
                LlmRequest(prompt=prompt, model=self._model, effort=self._effort, max_tokens=512)
            )
        except Exception as exc:
            log.warning(
                "hyde generation failed, falling back to vanilla", fields={"error": str(exc)}
            )
            return RewriteResult(queries=(query,), strategy="hyde:fallback")

        hypothetical = response.text.strip()
        if len(hypothetical) < 40:
            # Too short to be answer-shaped, so it carries no more signal than the
            # question and costs a search.
            return RewriteResult(queries=(query,), strategy="hyde:too-short", llm_calls=1)

        # Keep the original alongside the hypothetical: HyDE helps on some queries
        # and hurts on others, and fusing both is more robust than replacing.
        return RewriteResult(queries=(hypothetical, query), strategy="hyde", llm_calls=1)


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_query_list(text: str, *, limit: int) -> tuple[str, ...]:
    """Pull clean queries out of a model's line-per-query response.

    Small models add preamble ("Here are three alternatives:"), numbering and
    markdown emphasis no matter how firmly the prompt forbids it. Parse defensively
    rather than trusting the format.
    """
    queries: list[str] = []
    for raw in text.splitlines():
        line = _BULLET.sub("", raw.strip()).strip("*_` ").strip()
        if len(line) < 8 or line.endswith(":"):
            continue
        if line.lower().startswith(("here are", "sure", "certainly", "alternative")):
            continue
        queries.append(line)
        if len(queries) >= limit:
            break
    return tuple(queries)


def build_transform(
    strategy: str, client: LlmClient | None, *, count: int, model: str = "", effort: str = "low"
) -> QueryTransform:
    """Construct the configured transform.

    `hybrid` uses the plain question: its gain comes from BM25 plus dense fusion,
    not from rewriting, and stacking a rewrite on top would confound the ablation.
    """
    if strategy in {"vanilla", "hybrid"} or client is None:
        return IdentityTransform()
    if strategy == "multi_query":
        return MultiQueryTransform(client, count=count, model=model, effort=effort)
    if strategy == "hyde":
        return HydeTransform(client, model=model, effort=effort)
    raise ValueError(f"unknown retrieval strategy {strategy!r}")
