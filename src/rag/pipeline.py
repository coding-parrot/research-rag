"""The pipeline: one entry point, one typed result.

    ask(question)
      -> input guard          (deny -> BLOCKED_INPUT, no retrieval, no model call)
      -> retrieve             (transform, hybrid search, fuse, dedup, rerank, cap)
      -> retrieval guard      (deny -> NO_RESULTS, no model call)
      -> answer               (generate, validate citations, retry once, or refuse)

Refusals are answers. Every path returns an `Answer` with its status, the decisions
that produced it, and the usage it cost. Nothing here raises on a policy outcome.
"""

from __future__ import annotations

from rag.domain import Answer, AnswerStatus, Usage
from rag.generate.answerer import Answerer
from rag.guardrails.input_guard import InputGuard
from rag.guardrails.retrieval_guard import RetrievalGuard
from rag.observability import get_logger, timed, trace
from rag.retrieve.retriever import Retriever

log = get_logger("pipeline")


class Pipeline:
    def __init__(
        self,
        *,
        input_guard: InputGuard,
        retriever: Retriever,
        retrieval_guard: RetrievalGuard,
        answerer: Answerer,
    ) -> None:
        self._input_guard = input_guard
        self._retriever = retriever
        self._retrieval_guard = retrieval_guard
        self._answerer = answerer

    def ask(self, question: str) -> Answer:
        with trace() as trace_id, timed(log, "ask"):
            verdict = self._input_guard.check(question)
            if not verdict.allowed:
                denial = verdict.denial
                log.info(
                    "blocked at input",
                    fields={"rule": denial.rule_id if denial else "unknown"},
                )
                return Answer(
                    status=AnswerStatus.BLOCKED_INPUT,
                    text=verdict.refusal_message,
                    decisions=verdict.decisions,
                    usage=Usage(),
                    trace_id=trace_id,
                )

            retrieval = self._retriever.retrieve(verdict.query)
            retrieval_verdict = self._retrieval_guard.check(
                retrieval.results, top_dense_score=retrieval.top_dense_score
            )
            decisions = (*verdict.decisions, *retrieval_verdict.decisions)

            if not retrieval_verdict.allowed:
                denial = retrieval_verdict.denial
                log.info(
                    "refused at retrieval",
                    fields={"rule": denial.rule_id if denial else "unknown"},
                )
                return Answer(
                    status=AnswerStatus.NO_RESULTS,
                    text=denial.reason if denial else "Nothing relevant was found.",
                    retrieved=retrieval.results,
                    decisions=decisions,
                    usage=Usage(),
                    trace_id=trace_id,
                )

            return self._answerer.answer(
                verdict.query,
                retrieval.results,
                quarantined=retrieval_verdict.flagged_chunk_ids,
                prior_decisions=decisions,
            )
