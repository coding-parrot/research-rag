"""Input guardrails.

Three checks, in increasing cost order: shape, injection, scope. Every rule returns a
`Decision` rather than raising or returning a bare bool, so each is unit-testable in
isolation and each shows up as a countable outcome in the eval report.

The scope check is embedding-based rather than a keyword list. A keyword list
over-refuses in a way that is invisible until users complain: "how long do you train
for?" contains no distinctive vocabulary but is a perfectly good question about this
corpus. False refusal is a tracked metric here for exactly that reason.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rag.config import GuardrailConfig
from rag.domain import Action, Decision
from rag.embed.base import Embedder, cosine_similarity
from rag.observability import get_logger, preview

log = get_logger("guard.input")

# Ordered by how strongly each phrase indicates an override attempt rather than a
# question that merely mentions prompts. The corpus is about language models, so
# "system prompt" alone is a legitimate research query and is not on this list.
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(?:the\s+|all\s+)?(?:previous|above|prior|preceding)\b", "override-previous"),
    (
        r"disregard\s+(?:the\s+|all\s+)?(?:previous|above|prior|instructions?)\b",
        "override-disregard",
    ),
    (r"forget\s+(?:everything|all|your)\s+(?:instructions?|above|previous)", "override-forget"),
    (r"you\s+are\s+now\s+(?:a|an|the)\b", "persona-switch"),
    (
        r"(?:reveal|print|repeat|output|show)\s+(?:me\s+)?your\s+(?:system\s+)?(?:prompt|instructions)",
        "prompt-exfil",
    ),
    (r"\bnew\s+(?:instructions?|rules?)\s*[:.]", "instruction-injection"),
    (r"</?(?:system|instructions?|context)>", "delimiter-injection"),
    (r"\bDAN\b|\bjailbreak\b|developer\s+mode", "jailbreak"),
)
_INJECTION = tuple((re.compile(p, re.IGNORECASE), rule) for p, rule in INJECTION_PATTERNS)

# Credentials and identifiers that should never be echoed back or sent to a model.
SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bsk-[A-Za-z0-9_-]{16,}\b", "api-key-openai-style"),
    (r"\bsk-ant-[A-Za-z0-9_-]{16,}\b", "api-key-anthropic"),
    (r"\bghp_[A-Za-z0-9]{30,}\b", "github-token"),
    (r"\bAKIA[0-9A-Z]{16}\b", "aws-access-key"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email-address"),
    (r"\b(?:\d[ -]?){13,19}\b", "card-number-like"),
)
_SECRETS = tuple((re.compile(p), rule) for p, rule in SECRET_PATTERNS)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class InputVerdict:
    """The outcome of every input rule, plus the query to actually use."""

    query: str
    decisions: tuple[Decision, ...]

    @property
    def allowed(self) -> bool:
        return all(d.allowed for d in self.decisions)

    @property
    def denial(self) -> Decision | None:
        return next((d for d in self.decisions if d.action is Action.DENY), None)

    @property
    def refusal_message(self) -> str:
        denial = self.denial
        return denial.reason if denial else ""


class ScopeClassifier:
    """Is this question about the indexed corpus?

    Compares the query against the centroid of the corpus embeddings. Cheap, needs
    no LLM call, and unlike a keyword list it generalises to vocabulary nobody
    thought to enumerate.
    """

    def __init__(self, embedder: Embedder, corpus_vectors: np.ndarray | None = None) -> None:
        self._embedder = embedder
        self._centroid: np.ndarray | None = None
        if corpus_vectors is not None and len(corpus_vectors):
            self.fit(corpus_vectors)

    def fit(self, corpus_vectors: np.ndarray) -> None:
        centroid = np.asarray(corpus_vectors, dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        self._centroid = centroid / norm if norm > 1e-12 else centroid

    @property
    def ready(self) -> bool:
        return self._centroid is not None

    def score(self, query: str) -> float:
        if self._centroid is None:
            return 1.0  # unfitted classifier must not block anything
        vector = self._embedder.embed_query(query)
        return float(cosine_similarity(vector, self._centroid.reshape(1, -1))[0])


class InputGuard:
    """Runs every input rule in order, stopping at the first denial."""

    def __init__(self, config: GuardrailConfig, scope: ScopeClassifier | None = None) -> None:
        self._config = config
        self._scope = scope

    def check(self, query: str) -> InputVerdict:
        decisions: list[Decision] = []
        normalized = normalize_query(query)

        shape = self._check_shape(normalized)
        decisions.append(shape)
        if not shape.allowed:
            return InputVerdict(query=normalized, decisions=tuple(decisions))

        injection = self._check_injection(normalized)
        decisions.append(injection)
        if not injection.allowed:
            return InputVerdict(query=normalized, decisions=tuple(decisions))

        secrets = self._check_secrets(normalized)
        decisions.append(secrets)
        if not secrets.allowed:
            return InputVerdict(query=normalized, decisions=tuple(decisions))

        decisions.append(self._check_scope(normalized))
        return InputVerdict(query=normalized, decisions=tuple(decisions))

    # ------------------------------------------------------------------ #

    def _check_shape(self, query: str) -> Decision:
        if len(query) < self._config.min_query_chars:
            return Decision.deny(
                "input.length.min",
                "That question is too short for me to work with. Could you expand it?",
                evidence=f"{len(query)} chars",
            )
        if len(query) > self._config.max_query_chars:
            return Decision.deny(
                "input.length.max",
                f"That question is longer than the {self._config.max_query_chars} character limit.",
                evidence=f"{len(query)} chars",
            )
        return Decision.allow("input.length")

    def _check_injection(self, query: str) -> Decision:
        for pattern, rule in _INJECTION:
            if match := pattern.search(query):
                log.warning(
                    "injection pattern in query",
                    fields={"rule": rule, "match": preview(match.group(0), 40)},
                )
                return Decision.deny(
                    f"input.injection.{rule}",
                    "That request looks like an attempt to change my instructions, so I did not run it.",
                    evidence=preview(match.group(0), 60),
                )
        return Decision.allow("input.injection")

    def _check_secrets(self, query: str) -> Decision:
        for pattern, rule in _SECRETS:
            if pattern.search(query):
                # Deliberately no evidence field: recording the match would write the
                # secret into the log line this decision ends up in.
                return Decision.deny(
                    f"input.secret.{rule}",
                    "That question appears to contain a credential or personal identifier, "
                    "so I did not send it on. Please remove it and ask again.",
                )
        return Decision.allow("input.secret")

    def _check_scope(self, query: str) -> Decision:
        if self._scope is None or not self._scope.ready:
            return Decision.allow("input.scope", "classifier not fitted")
        score = self._scope.score(query)
        if score < self._config.scope_threshold:
            return Decision.deny(
                "input.scope",
                "That question does not look like it is about the research papers I have indexed.",
                evidence=f"similarity {score:.3f} < {self._config.scope_threshold:.3f}",
            )
        return Decision.allow("input.scope", f"similarity {score:.3f}")


def normalize_query(query: str) -> str:
    """NFKC-normalise, strip control characters, collapse whitespace.

    Unicode normalisation is a security step, not a cosmetic one: it collapses
    homoglyph and full-width variants that would otherwise slip past the injection
    patterns while still reading as the same instruction to the model.
    """
    text = unicodedata.normalize("NFKC", query or "")
    text = _CONTROL_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def scan_for_injection(texts: Sequence[str]) -> list[tuple[int, str, str]]:
    """Find injection patterns in retrieved chunks.

    This is the check the notebook version was missing. Guarding the query defends
    against a hostile user; the more realistic threat in a document-grounded system
    is a hostile *document*, because retrieved text enters the same prompt with none
    of the user's text having been involved.
    """
    findings: list[tuple[int, str, str]] = []
    for index, text in enumerate(texts):
        for pattern, rule in _INJECTION:
            if match := pattern.search(text):
                findings.append((index, rule, preview(match.group(0), 60)))
                break
    return findings
