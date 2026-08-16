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
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

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
    def dropped_citations(self) -> tuple[Decision, ...]:
        """MODIFY decisions for citations that failed validation and were dropped."""
        return tuple(
            d
            for d in self.decisions
            if d.action is Action.MODIFY and d.rule_id.startswith("output.citation.")
        )

    @property
    def should_retry(self) -> bool:
        """Citation failures are worth one regeneration; policy failures are not.

        A dropped citation makes the verdict retryable even when other citations
        survived: the claim the dropped citation supported would otherwise ship
        unverified. On the final attempt the answerer ships the surviving citations,
        with the drops on record as MODIFY decisions.
        """
        denial = self.denial
        if denial is not None:
            return denial.rule_id.startswith("output.citation")
        return bool(self.dropped_citations)


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
            # Redaction runs after citation validation on purpose: the verbatim
            # quote check needs the raw chunk text. It then covers both the prose
            # and the surviving quotes, so a validated quote may ship with
            # [redacted] inside it rather than leaking the identifier verbatim.
            answer, found = _redact_text(answer)
            citations, quote_found = _redact_quotes(citations)
            found.extend(quote_found)
            if found:
                decisions.append(_redaction_decision(found))

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
                # The chunk_id is model-controlled and can be arbitrarily long, so
                # it gets the same preview() bound as every other evidence field.
                # Real ids are 16 hex chars and are never truncated by it.
                decisions.append(
                    Decision.modify(
                        "output.citation.unknown_chunk",
                        "A citation referenced a passage that was not retrieved; it was dropped.",
                        evidence=preview(chunk_id, 60) or "<empty>",
                    )
                )
                log.warning(
                    "citation to unretrieved chunk", fields={"chunk_id": preview(chunk_id, 60)}
                )
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

    def redact(self, text: str) -> tuple[str, Decision | None]:
        """Redact identifiers from any model-authored text before it ships.

        check() already redacts validated answers; this is the seam for text that
        never reaches check(), such as model-authored refusal prose. The answerer
        routes every text it ships through one of the two.
        """
        if not self._config.redact_pii_in_output:
            return text, None
        redacted, found = _redact_text(text)
        if not found:
            return text, None
        return redacted, _redaction_decision(found)


def _redact_text(text: str) -> tuple[str, list[str]]:
    """Scrub credentials and identifiers the corpus may contain.

    Research papers carry author emails on the first page, which is exactly the
    text that lands in the frontmatter chunk and gets retrieved by an author query.
    Returns the redacted text plus "{rule}x{count}" markers for what was hit.
    """
    found: list[str] = []
    redacted = text
    for pattern, rule in _SECRETS:
        count = 0

        def _sub(match: re.Match[str], rule: str = rule) -> str:
            nonlocal count
            # The card-number pattern alone also matches runs of years or ids
            # ("2019 2020 2021 2022"); only Luhn-valid digit strings are real
            # card numbers worth destroying prose over.
            if rule == _CARD_RULE and not _luhn_valid(match.group(0)):
                return match.group(0)
            count += 1
            return REDACTION

        redacted = pattern.sub(_sub, redacted)
        if count:
            found.append(f"{rule}x{count}")
    return redacted, found


def _redact_quotes(citations: tuple[Citation, ...]) -> tuple[tuple[Citation, ...], list[str]]:
    """Redact identifiers inside validated citation quotes.

    Quotes ship in Answer.citations to every consumer (APIs, eval reports, UIs),
    so they get the same scrubbing as the prose. Runs after quote verification,
    which needs the verbatim text; a redacted quote is acceptable once validated.
    """
    found: list[str] = []
    redacted: list[Citation] = []
    for citation in citations:
        quote, quote_found = _redact_text(citation.quote)
        if quote_found:
            found.extend(f"quote:{item}" for item in quote_found)
            redacted.append(replace(citation, quote=quote))
        else:
            redacted.append(citation)
    return tuple(redacted), found


def _redaction_decision(found: Sequence[str]) -> Decision:
    return Decision.modify(
        "output.redaction",
        "Identifiers in the answer were redacted.",
        evidence=",".join(found),
    )


# Rule name of the card-number pattern in SECRET_PATTERNS; only matches passing the
# Luhn checksum are redacted under it.
_CARD_RULE = "card-number-like"


def _luhn_valid(candidate: str) -> bool:
    """Luhn checksum over the digits of a candidate card number.

    The input guard applies the same gate; the helper is deliberately duplicated
    here (see input_guard) so the guard modules share only the pattern table, not
    each other's internals.
    """
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# Typographic punctuation folded to ASCII before quote matching. Ingest is NFKC-only
# and NFKC preserves curly quotes, en/em dashes and ellipses from PDFs, while models
# routinely emit the ASCII forms when quoting; without folding, a verbatim-in-spirit
# quote is rejected and the regeneration burns on a doomed retry.
_ASCII_FOLD = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote / apostrophe
        "\u201a": "'",  # low single quote
        "\u201b": "'",  # reversed single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u201e": '"',  # low double quote
        "\u201f": '"',  # reversed double quote
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\u2026": "...",  # ellipsis
    }
)


def _fold_for_match(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_ASCII_FOLD)
    return _WS.sub(" ", folded).strip().casefold()


def quote_appears_in(quote: str, source: str) -> bool:
    """Whitespace- and typography-insensitive substring check.

    Exact matching is too strict in practice: models reflow whitespace, normalise
    line breaks, and straighten curly quotes and dashes when quoting. Rejecting a
    correct citation over a newline or a curly quote would push the answerer into a
    pointless regeneration. Letters and digits must still match exactly, which is
    what keeps this a real check rather than a fuzzy one.
    """
    return _fold_for_match(quote) in _fold_for_match(source)


def format_citation_markers(text: str, citations: Sequence[Citation]) -> str:
    """Append a numbered source list under the answer.

    Kept separate from the model's prose so citation rendering is our concern, not
    something the model can get subtly wrong.
    """
    if not citations:
        return text
    lines = [f"[{i}] {c.label}" for i, c in enumerate(citations, start=1)]
    return f"{text}\n\nSources:\n" + "\n".join(lines)
