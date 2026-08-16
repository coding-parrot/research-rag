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

# A genuine refusal per ANSWER_TEMPLATE is "a brief statement that the indexed
# papers do not cover it", so it leads with the marker and stays short. A
# substantive answer that merely contains a marker phrase somewhere must fall
# through to citation validation and its retry instead of being misfiled here.
_REFUSAL_MARKER_WINDOW = 160  # the marker must appear within this many chars
_REFUSAL_MAX_CHARS = 200  # and the whole answer must be this short


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
        # An allowed verdict whose citations were partially dropped. Kept so a retry
        # that fails outright still ships the earlier partial answer instead of a
        # refusal; the MODIFY decisions record what was dropped.
        partial: OutputVerdict | None = None

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

            if response.truncated:
                # A body cut at max_tokens is not a parseable final answer, and a
                # retry at the same limit would truncate again. Treated like an
                # LlmError: record the real cause, refuse without burning the
                # citation retry on a doomed regeneration.
                log.error(
                    "generation truncated at max_tokens",
                    fields={"attempt": attempt, "max_tokens": self._config.max_tokens},
                )
                decisions.append(
                    Decision.deny(
                        "generate.truncated",
                        "The model ran out of output tokens before finishing its answer.",
                        evidence=f"max_tokens={self._config.max_tokens}",
                    )
                )
                return self._refusal(
                    AnswerStatus.INSUFFICIENT_EVIDENCE, retrieved, decisions, usage
                )

            text, raw_citations = _parse(response.parsed, response.text)

            if _is_model_refusal(text, raw_citations):
                # Refusal prose is still model-authored text built over retrieved
                # chunks, so it gets the same redaction the OK path gets.
                text, redaction = self._guard.redact(text)
                if redaction is not None:
                    decisions.append(redaction)
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

            final = attempt + 1 >= attempts
            if verdict.allowed:
                if not verdict.should_retry or final:
                    # Clean, or the final attempt: dropped citations (if any) are
                    # on record as MODIFY decisions and the survivors ship.
                    if attempt > 0:
                        log.info(
                            "regeneration recovered a valid answer", fields={"attempt": attempt}
                        )
                    return self._ok(verdict, retrieved, decisions, usage)
                # Some citations were dropped; the claims they supported would ship
                # unverified. Worth one regeneration, like any citation failure.
                partial = verdict

            if not verdict.should_retry or final:
                break

            denial = verdict.denial
            if denial is not None:
                reason = denial.reason
            elif verdict.dropped_citations:
                reason = "Some citations failed validation and were dropped: " + " ".join(
                    d.reason for d in verdict.dropped_citations
                )
            else:
                reason = "citation validation failed"
            log.warning(
                "citation validation failed, regenerating",
                fields={"attempt": attempt, "reason": reason},
            )
            prompt = build_answer_prompt(question, retrieved, quarantined=quarantined)
            prompt += REGENERATION_SUFFIX.format(reason=reason)

        if partial is not None:
            # The retry did worse than the partial answer it was meant to improve
            # on; ship the partial rather than refuse.
            log.info("shipping partial-citation answer after failed regeneration")
            return self._ok(partial, retrieved, decisions, usage)

        denial = last_verdict.denial if last_verdict else None
        log.warning(
            "no valid answer after retries",
            fields={"rule": denial.rule_id if denial else "unknown"},
        )
        return self._refusal(AnswerStatus.INSUFFICIENT_EVIDENCE, retrieved, decisions, usage)

    # ------------------------------------------------------------------ #

    def _ok(
        self,
        verdict: OutputVerdict,
        retrieved: Sequence[Scored],
        decisions: Sequence[Decision],
        usage: Usage,
    ) -> Answer:
        return Answer(
            status=AnswerStatus.OK,
            text=verdict.text,
            citations=verdict.citations,
            retrieved=tuple(retrieved),
            decisions=tuple(decisions),
            usage=usage,
            trace_id=current_trace_id(),
        )

    def _refusal(
        self,
        status: AnswerStatus,
        retrieved: Sequence[Scored],
        decisions: Sequence[Decision],
        usage: Usage,
    ) -> Answer:
        # The template is static so redact() is a no-op today; it is routed through
        # anyway so every text the answerer ships passes the same seam.
        text, redaction = self._guard.redact(
            "I can't give a properly cited answer to that from the indexed papers, "
            "so I'd rather not answer than answer without evidence."
        )
        all_decisions = list(decisions)
        if redaction is not None:
            all_decisions.append(redaction)
        return Answer(
            status=status,
            text=text,
            retrieved=tuple(retrieved),
            decisions=tuple(all_decisions),
            usage=usage,
            trace_id=current_trace_id(),
        )


def _parse(parsed: Mapping[str, Any] | None, raw_text: str) -> tuple[str, list[Mapping[str, str]]]:
    """Extract (answer, citations) from the structured response.

    Falls back to treating the whole response as uncited prose when parsing fails,
    which then fails citation validation and produces an honest refusal rather than
    a crash. A structurally wrong `answer` (dict, list, null, missing) gets the same
    raw-text fallback rather than a str() coercion: its repr must never ship as
    prose, and the fallback fails validation with the retryable
    `output.citation.none_valid` rule rather than the non-retryable `output.empty`.
    """
    if parsed is None:
        return raw_text.strip(), []

    answer = parsed.get("answer")
    if not isinstance(answer, str):
        return raw_text.strip(), []

    raw = parsed.get("citations", [])
    citations: list[Mapping[str, str]] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            chunk_id = entry.get("chunk_id", "")
            quote = entry.get("quote", "")
            # Non-string citation fields are the model misreading the schema, not
            # evidence; dropping the entry keeps str() reprs out of validation.
            if not isinstance(chunk_id, str) or not isinstance(quote, str):
                continue
            citations.append({"chunk_id": chunk_id, "quote": quote})
    return answer.strip(), citations


def _is_model_refusal(text: str, citations: Sequence[Mapping[str, str]]) -> bool:
    if citations:
        return False
    stripped = text.strip()
    if len(stripped) > _REFUSAL_MAX_CHARS:
        return False
    head = stripped[:_REFUSAL_MARKER_WINDOW].lower()
    return any(marker in head for marker in _REFUSAL_MARKERS)
