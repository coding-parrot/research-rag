"""The answerer: generation with enforced citations.

Flow per request:

    retrieved chunks -> prompt -> model -> citation validation -> answer
                                     ^                |
                                     +--- one retry --+  (then typed refusal)

The retry is deliberately bounded and specific: it fires only when citation
validation failed, it tells the model exactly which rule failed, and after
`max_regenerations` attempts the answer becomes `INSUFFICIENT_EVIDENCE` rather than
shipping unverified claims. Fabrication is contained by construction, not by hope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag.config import GenerateConfig
from rag.domain import Answer, AnswerStatus, Decision, Scored, Usage
from rag.errors import LlmError
from rag.generate.client import LlmClient, LlmRequest
from rag.generate.prompts import (
    ANSWER_SCHEMA,
    REGENERATION_SUFFIX,
    SYSTEM_PROMPT,
    build_answer_prompt,
)
from rag.guardrails.output_guard import OutputGuard, OutputVerdict
from rag.observability import current_trace_id, get_logger

log = get_logger("answerer")

# The model saying "the papers do not cover this" with no citations is a refusal,
# not a citation failure. Detecting it keeps the retry loop from pointlessly asking
# a correct refusal to add citations it correctly does not have.
_REFUSAL_MARKERS = (
    "do not cover",
    "does not cover",
    "not covered",
    "do not contain",
    "does not contain",
    "no information",
    "not addressed",
    "cannot answer",
    "can't answer",
    "don't discuss",
    "do not discuss",
)


class Answerer:
    def __init__(self, client: LlmClient, guard: OutputGuard, config: GenerateConfig) -> None:
        self._client = client
        self._guard = guard
        self._config = config

    def answer(
        self,
        question: str,
        retrieved: Sequence[Scored],
        *,
        quarantined: Sequence[str] = (),
        prior_decisions: Sequence[Decision] = (),
    ) -> Answer:
        usage = Usage()
        decisions: list[Decision] = list(prior_decisions)
        prompt = build_answer_prompt(question, retrieved, quarantined=quarantined)

        attempts = 1 + max(0, self._config.max_regenerations)
        last_verdict: OutputVerdict | None = None

        for attempt in range(attempts):
            try:
                response = self._client.complete(
                    LlmRequest(
                        prompt=prompt,
                        system=SYSTEM_PROMPT,
                        model=self._config.model,
                        effort=self._config.effort,
                        max_tokens=self._config.max_tokens,
                        schema=ANSWER_SCHEMA,
                        cache_system=True,
                    )
                )
            except LlmError as exc:
                log.error("generation failed", fields={"attempt": attempt, "error": str(exc)})
                decisions.append(
                    Decision.deny("generate.error", "The model call failed.", evidence=str(exc))
                )
                return self._refusal(
                    AnswerStatus.INSUFFICIENT_EVIDENCE, retrieved, decisions, usage
                )

            usage = usage + response.usage

            if response.refused:
                decisions.append(
                    Decision.deny(
                        "generate.model_refusal", "The model declined to answer this question."
                    )
                )
                return self._refusal(AnswerStatus.BLOCKED_OUTPUT, retrieved, decisions, usage)

            text, raw_citations = _parse(response.parsed, response.text)

            if _is_model_refusal(text, raw_citations):
                decisions.append(Decision.allow("generate.no_answer", "model reported no coverage"))
                return Answer(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    text=text,
                    retrieved=tuple(retrieved),
                    decisions=tuple(decisions),
                    usage=usage,
                    trace_id=current_trace_id(),
                )

            verdict = self._guard.check(text=text, raw_citations=raw_citations, retrieved=retrieved)
            decisions.extend(verdict.decisions)
            last_verdict = verdict

            if verdict.allowed:
                if attempt > 0:
                    log.info("regeneration recovered a valid answer", fields={"attempt": attempt})
                return Answer(
                    status=AnswerStatus.OK,
                    text=verdict.text,
                    citations=verdict.citations,
                    retrieved=tuple(retrieved),
                    decisions=tuple(decisions),
                    usage=usage,
                    trace_id=current_trace_id(),
                )

            if not verdict.should_retry or attempt + 1 >= attempts:
                break

            denial = verdict.denial
            reason = denial.reason if denial else "citation validation failed"
            log.warning(
                "citation validation failed, regenerating",
                fields={"attempt": attempt, "reason": reason},
            )
            prompt = build_answer_prompt(question, retrieved, quarantined=quarantined)
            prompt += REGENERATION_SUFFIX.format(reason=reason)

        denial = last_verdict.denial if last_verdict else None
        log.warning(
            "no valid answer after retries",
            fields={"rule": denial.rule_id if denial else "unknown"},
        )
        return self._refusal(AnswerStatus.INSUFFICIENT_EVIDENCE, retrieved, decisions, usage)

    # ------------------------------------------------------------------ #

    def _refusal(
        self,
        status: AnswerStatus,
        retrieved: Sequence[Scored],
        decisions: Sequence[Decision],
        usage: Usage,
    ) -> Answer:
        return Answer(
            status=status,
            text=(
                "I can't give a properly cited answer to that from the indexed papers, "
                "so I'd rather not answer than answer without evidence."
            ),
            retrieved=tuple(retrieved),
            decisions=tuple(decisions),
            usage=usage,
            trace_id=current_trace_id(),
        )


def _parse(parsed: Mapping[str, Any] | None, raw_text: str) -> tuple[str, list[Mapping[str, str]]]:
    """Extract (answer, citations) from the structured response.

    Falls back to treating the whole response as uncited prose when parsing fails,
    which then fails citation validation and produces an honest refusal rather than
    a crash.
    """
    if parsed is None:
        return raw_text.strip(), []

    text = str(parsed.get("answer", "")).strip()
    raw = parsed.get("citations", [])
    citations: list[Mapping[str, str]] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, Mapping):
                citations.append(
                    {
                        "chunk_id": str(entry.get("chunk_id", "")),
                        "quote": str(entry.get("quote", "")),
                    }
                )
    return text, citations


def _is_model_refusal(text: str, citations: Sequence[Mapping[str, str]]) -> bool:
    if citations:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)
