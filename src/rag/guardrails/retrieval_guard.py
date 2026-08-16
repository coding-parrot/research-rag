"""Retrieval guardrails.

Two jobs, both about refusing early:

  Relevance floor. A vector store always returns k results. When nothing in the
  corpus is relevant it returns the k least-unrelated chunks and the model is left
  to notice, which it frequently does not. Checking the top score before calling the
  model turns a likely hallucination into a cheap, honest refusal.

  Injection scan. Retrieved text is untrusted data. A paper can contain text that
  reads as an instruction, deliberately or by accident (this corpus contains papers
  *about* prompt injection, which quote attack strings verbatim). We neutralise
  rather than drop: the chunk may still be the correct answer to the question.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import GuardrailConfig
from rag.domain import Action, Decision, Scored
from rag.guardrails.input_guard import scan_for_injection
from rag.observability import get_logger

log = get_logger("guard.retrieval")


@dataclass(frozen=True, slots=True)
class RetrievalVerdict:
    results: tuple[Scored, ...]
    decisions: tuple[Decision, ...]

    @property
    def allowed(self) -> bool:
        return all(d.allowed for d in self.decisions)

    @property
    def denial(self) -> Decision | None:
        return next((d for d in self.decisions if d.action is Action.DENY), None)

    @property
    def flagged_chunk_ids(self) -> tuple[str, ...]:
        """Chunks that tripped the injection scan and were marked as quarantined."""
        return tuple(
            d.evidence.split("|", 1)[0]
            for d in self.decisions
            if d.action is Action.MODIFY and d.rule_id.startswith("retrieval.injection")
        )


class RetrievalGuard:
    def __init__(self, config: GuardrailConfig) -> None:
        self._config = config

    def check(
        self, results: tuple[Scored, ...], *, top_dense_score: float | None = None
    ) -> RetrievalVerdict:
        """Validate a retrieval before it reaches the model.

        `top_dense_score` is the best raw cosine similarity, the calibrated signal
        for the relevance floor. Post-fusion `Scored.score` values are rank-based
        (RRF) or unbounded logits (cross-encoder); thresholds on those mean nothing.
        When it is not supplied (tests, vanilla single-query mode), the top result's
        own score is used.
        """
        decisions: list[Decision] = []

        if not results:
            return RetrievalVerdict(
                results=(),
                decisions=(
                    Decision.deny(
                        "retrieval.empty",
                        "I could not find anything in the indexed papers that speaks to that question.",
                    ),
                ),
            )

        top = top_dense_score if top_dense_score is not None else results[0].score
        if top < self._config.relevance_floor:
            decisions.append(
                Decision.deny(
                    "retrieval.relevance_floor",
                    "I could not find anything in the indexed papers that speaks to that question.",
                    evidence=f"top dense score {top:.3f} < floor {self._config.relevance_floor:.3f}",
                )
            )
            log.info(
                "refused below relevance floor",
                fields={"top_score": round(top, 4), "floor": self._config.relevance_floor},
            )
            return RetrievalVerdict(results=results, decisions=tuple(decisions))

        decisions.append(Decision.allow("retrieval.relevance_floor", f"top dense score {top:.3f}"))

        if self._config.scan_retrieved_for_injection:
            decisions.extend(self._scan(results))

        return RetrievalVerdict(results=results, decisions=tuple(decisions))

    def _scan(self, results: tuple[Scored, ...]) -> list[Decision]:
        findings = scan_for_injection([s.chunk.text for s in results])
        if not findings:
            return [Decision.allow("retrieval.injection")]

        decisions: list[Decision] = []
        for index, rule, snippet in findings:
            chunk = results[index].chunk
            log.warning(
                "instruction-like text in a retrieved chunk",
                fields={"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "rule": rule},
            )
            decisions.append(
                Decision.modify(
                    f"retrieval.injection.{rule}",
                    "A retrieved passage contains instruction-like text; it was quarantined "
                    "as data before being shown to the model.",
                    evidence=f"{chunk.chunk_id}|{snippet}",
                )
            )
        return decisions
