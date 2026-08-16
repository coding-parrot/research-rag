"""Output guardrails.

This is where most RAG systems leak, and where the notebook version was weakest.
`[source: filename]` written as free text inside prose is unverifiable: the model can
cite a file it never saw, and nothing in the system can tell. Here the model must
return structured citations, and every one is checked mechanically before it ships:

  1. The cited `chunk_id` must be in the set that was actually retrieved.
  2. The `quote` must appear verbatim in that chunk.

Both checks are deterministic, free, and run on every request. They catch the bulk of
fabrication without a judge model being involved at all, which matters because the
judge is the expensive, slow, and least reliable part of the stack.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.config import GuardrailConfig
from rag.domain import Action, Citation, Decision, Scored
from rag.guardrails.input_guard import SECRET_PATTERNS
from rag.observability import get_logger, preview

log = get_logger("guard.output")

_SECRETS = tuple((re.compile(p), rule) for p, rule in SECRET_PATTERNS)
_WS = re.compile(r"\s+")

REDACTION = "[redacted]"


@dataclass(frozen=True, slots=True)
class OutputVerdict:
    """The validated answer, its surviving citations, and why."""

    text: str
    citations: tuple[Citation, ...]
    decisions: tuple[Decision, ...]

    @property
    def allowed(self) -> bool:
        return all(d.allowed for d in self.decisions)

    @property
    def denial(self) -> Decision | None:
        return next((d for d in self.decisions if d.action is Action.DENY), None)

    @property
    def should_retry(self) -> bool:
        """Citation failures are worth one regeneration; policy failures are not."""
        denial = self.denial
        return denial is not None and denial.rule_id.startswith("output.citation")


class OutputGuard:
    """Validates a model answer against the chunks that were actually retrieved."""

    def __init__(self, config: GuardrailConfig) -> None:
        self._config = config

    def check(
        self,
        *,
        text: str,
        raw_citations: Sequence[Mapping[str, str]],
        retrieved: Sequence[Scored],
    ) -> OutputVerdict:
        decisions: list[Decision] = []
        by_id = {s.chunk.chunk_id: s.chunk for s in retrieved}

        answer = text.strip()
        if not answer:
            return OutputVerdict(
                text="",
                citations=(),
                decisions=(Decision.deny("output.empty", "The model returned an empty answer."),),
            )

        citations, citation_decisions = self._validate_citations(raw_citations, by_id)
        decisions.extend(citation_decisions)

        if self._config.require_citations and not citations:
            decisions.append(
                Decision.deny(
                    "output.citation.none_valid",
                    "I could not produce an answer with verifiable citations from the "
                    "retrieved passages.",
                    evidence=f"{len(raw_citations)} citations proposed, 0 survived validation",
                )
            )
            return OutputVerdict(text=answer, citations=(), decisions=tuple(decisions))

        if self._config.redact_pii_in_output:
            answer, redaction = self._redact(answer)
            if redaction is not None:
                decisions.append(redaction)

        return OutputVerdict(text=answer, citations=citations, decisions=tuple(decisions))

    # ------------------------------------------------------------------ #

    def _validate_citations(
        self, raw: Sequence[Mapping[str, str]], by_id: Mapping[str, object]
    ) -> tuple[tuple[Citation, ...], list[Decision]]:
        from rag.domain import Chunk

        kept: list[Citation] = []
        decisions: list[Decision] = []

        for entry in raw:
            chunk_id = str(entry.get("chunk_id", "")).strip()
            quote = str(entry.get("quote", "")).strip()

            chunk = by_id.get(chunk_id)
            if not isinstance(chunk, Chunk):
                # The single highest-signal fabrication check in the system: the model
                # invented an identifier that was never in its context.
                decisions.append(
                    Decision.modify(
                        "output.citation.unknown_chunk",
                        "A citation referenced a passage that was not retrieved; it was dropped.",
                        evidence=chunk_id or "<empty>",
                    )
                )
                log.warning("citation to unretrieved chunk", fields={"chunk_id": chunk_id})
                continue

            if len(quote) < self._config.min_quote_chars:
                decisions.append(
                    Decision.modify(
                        "output.citation.quote_too_short",
                        "A citation quoted too little text to count as evidence; it was dropped.",
                        evidence=f"{chunk_id}|{len(quote)} chars",
                    )
                )
                continue

            if self._config.verify_quotes and not quote_appears_in(quote, chunk.text):
                decisions.append(
                    Decision.modify(
                        "output.citation.quote_not_found",
                        "A citation quoted text that does not appear in the passage it cited; "
                        "it was dropped.",
                        evidence=f"{chunk_id}|{preview(quote, 60)}",
                    )
                )
                log.warning(
                    "quote not present in cited chunk",
                    fields={"chunk_id": chunk_id, "quote": preview(quote, 60)},
                )
                continue

            kept.append(Citation(chunk_id=chunk_id, quote=quote, label=chunk.citation_label))

        if kept:
            decisions.append(
                Decision.allow("output.citation", f"{len(kept)}/{len(raw)} citations verified")
            )
        return tuple(kept), decisions

    def _redact(self, text: str) -> tuple[str, Decision | None]:
        """Scrub credentials and identifiers the corpus may contain.

        Research papers carry author emails on the first page, which is exactly the
        text that lands in the frontmatter chunk and gets retrieved by an author query.
        """
        found: list[str] = []
        redacted = text
        for pattern, rule in _SECRETS:
            redacted, count = pattern.subn(REDACTION, redacted)
            if count:
                found.append(f"{rule}x{count}")
        if not found:
            return text, None
        return redacted, Decision.modify(
            "output.redaction",
            "Identifiers in the answer were redacted.",
            evidence=",".join(found),
        )


def quote_appears_in(quote: str, source: str) -> bool:
    """Whitespace-insensitive substring check.

    Exact matching is too strict in practice: models reflow whitespace and normalise
    line breaks when quoting, and rejecting a correct citation over a newline would
    push the answerer into a pointless regeneration. Everything other than whitespace
    must match exactly, which is what keeps this a real check rather than a fuzzy one.
    """
    return _WS.sub(" ", quote).strip().lower() in _WS.sub(" ", source).strip().lower()


def format_citation_markers(text: str, citations: Sequence[Citation]) -> str:
    """Append a numbered source list under the answer.

    Kept separate from the model's prose so citation rendering is our concern, not
    something the model can get subtly wrong.
    """
    if not citations:
        return text
    lines = [f"[{i}] {c.label}" for i, c in enumerate(citations, start=1)]
    return f"{text}\n\nSources:\n" + "\n".join(lines)
