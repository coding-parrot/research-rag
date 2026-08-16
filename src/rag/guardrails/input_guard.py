"""Input guardrails.

Four checks, in increasing cost order: shape, injection, secrets, scope. Every rule
returns a `Decision` rather than raising or returning a bare bool, so each is
unit-testable in isolation and each shows up as a countable outcome in the eval
report. Secrets is the one rule allowed to rewrite the query: identifiers that are
merely *in* a legitimate question (an author's published email, a card number pasted
by accident) are redacted and the question continues, while credential-shaped input
is denied outright.

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
#
# This tuple is shared with scan_for_injection, so every loosening or tightening
# here applies to hostile retrieved documents as well as hostile queries. That is
# deliberate: the two paths must not drift apart.
#
# The override rules tolerate a little punctuation between verb and object (\W{1,3})
# but never arbitrary words: a free word gap falsely matches benign research
# phrasings on this corpus ("ignore adversarial instructions in retrieved text").
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"(?:ignore|disregard)\W{1,3}(?:(?:all|any)\s+(?:of\s+)?)?"
        r"(?:the\s+|your\s+|my\s+|these\s+|those\s+)?(?:previous|above|prior|preceding|earlier)\b",
        "override-previous",
    ),
    (
        r"(?:ignore|disregard)\W{1,3}(?:your|my|these|those|all)\s+"
        r"(?:instructions?|rules?|prompts?)\b",
        "override-object",
    ),
    (r"(?:ignore|disregard)\W{1,3}everything\b", "override-everything"),
    (
        r"disregard\s+(?:the\s+|all\s+)?(?:previous|above|prior|instructions?)\b",
        "override-disregard",
    ),
    (r"forget\s+(?:everything|all|your)\s+(?:instructions?|above|previous)", "override-forget"),
    (r"you\s+are\s+now\s+(?:a|an|the)\b", "persona-switch"),
    (
        r"(?:reveal|print|repeat|output|show)\s+(?:me\s+)?(?:your|the|its)\s+"
        r"(?:system\s+)?(?:prompt|instructions)",
        "prompt-exfil",
    ),
    (r"\bnew\s+(?:instructions?|rules?)\s*[:.]", "instruction-injection"),
    # passages? covers the <passage>/<passages> tags prompts.py wraps retrieved text
    # in: a document containing a literal </passage> can forge passage boundaries.
    (r"</?(?:system|instructions?|context|passages?)>", "delimiter-injection"),
    (
        # (?-i:...) keeps DAN case-sensitive inside this otherwise IGNORECASE tuple:
        # the jailbreak persona is written in caps, researchers named Dan are not.
        # "jailbreak" requires imperative context (an object like "yourself") so that
        # topic questions about jailbreak papers survive.
        r"(?:you\s+are\s+now|act\s+as|become)\s+(?-i:DAN)\b"
        r"|(?:enable|activate|enter|engage)\s+(?-i:DAN)\b"
        r"|(?-i:\bDAN\b)\s+mode"
        r"|jailbreak\s+(?:yourself|(?:the|this)\s+(?:model|assistant|system))"
        r"|developer\s+mode",
        "jailbreak",
    ),
)
_INJECTION = tuple((re.compile(p, re.IGNORECASE), rule) for p, rule in INJECTION_PATTERNS)

# Credentials and identifiers that should never be echoed back or sent to a model.
# Constraints the tuple itself cannot show:
#   * First match wins in the input guard, so the specific sk-ant- prefix must come
#     before the generic sk- rule or every Anthropic key gets mislabelled
#     api-key-openai-style in rule ids and eval counts.
#   * Deny classes (keys, tokens) must precede redact classes (email, card): the
#     input guard hard-denies on the former and rewrites the query on the latter,
#     and a deny must win over any partial redaction. See _check_secrets.
#   * output_guard reuses this tuple for blanket redaction of answers.
SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bsk-ant-[A-Za-z0-9_-]{16,}\b", "api-key-anthropic"),
    (r"\bsk-[A-Za-z0-9_-]{16,}\b", "api-key-openai-style"),
    (r"\bghp_[A-Za-z0-9]{30,}\b", "github-token"),
    (r"\bAKIA[0-9A-Z]{16}\b", "aws-access-key"),
    # Redacted from the query, not denied: author-lookup questions legitimately
    # quote published contact addresses from paper frontmatter.
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email-address"),
    # A candidate finder only. This regex alone also matches ISBNs, lists of years,
    # and epoch-millisecond timestamps, so the input guard redacts a match only when
    # the separator-stripped digits pass the Luhn checksum. See _redact_card_numbers.
    (r"\b(?:\d[ -]?){13,19}\b", "card-number-like"),
)
_SECRETS = tuple((re.compile(p), rule) for p, rule in SECRET_PATTERNS)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CARD_SEPARATORS = re.compile(r"[ -]")

# ISBN-13s live in the 978/979 "bookland" prefix and are ordinary research query
# content, so a 13-digit run starting with either is never treated as a card, even
# in the roughly one-in-ten case where such a run also passes Luhn.
_ISBN_PREFIXES = ("978", "979")

# Same literal as output_guard.REDACTION. Defined here as well because output_guard
# imports from this module, so importing it back would be circular.
REDACTION = "[redacted]"

# Common Cyrillic and Greek letters that render identically to Latin ones. NFKC does
# not fold these (they are distinct letters, not compatibility forms), so an attacker
# can spell "ignore" with U+0435 and slip past every injection pattern. The fold is
# applied to a scan-only copy: the query itself is never rewritten with it, so Greek
# symbols in legitimate maths questions reach retrieval untouched. Case-preserving on
# purpose, because the DAN rule is case-sensitive. Curated, not exhaustive: this is
# defence in depth, not a full confusables database.
_HOMOGLYPHS = str.maketrans(
    {
        # Cyrillic lowercase
        0x0430: "a",
        0x0435: "e",
        0x0450: "e",
        0x0451: "e",
        0x043E: "o",
        0x0440: "p",
        0x0441: "c",
        0x0445: "x",
        0x0443: "y",
        0x0456: "i",
        0x0457: "i",
        0x0455: "s",
        0x0458: "j",
        0x04BB: "h",
        0x051B: "q",
        0x051D: "w",
        # Cyrillic uppercase
        0x0410: "A",
        0x0412: "B",
        0x0415: "E",
        0x0400: "E",
        0x0401: "E",
        0x041A: "K",
        0x041C: "M",
        0x041D: "H",
        0x041E: "O",
        0x0420: "P",
        0x0421: "C",
        0x0422: "T",
        0x0425: "X",
        0x0423: "Y",
        0x0405: "S",
        0x0406: "I",
        0x0407: "I",
        0x0408: "J",
        0x04BA: "H",
        0x051A: "Q",
        0x051C: "W",
        # Greek lowercase
        0x03B1: "a",
        0x03B5: "e",
        0x03B9: "i",
        0x03BA: "k",
        0x03BD: "v",
        0x03BF: "o",
        0x03C1: "p",
        0x03C4: "t",
        0x03C5: "u",
        0x03C7: "x",
        # Greek uppercase
        0x0391: "A",
        0x0392: "B",
        0x0395: "E",
        0x0396: "Z",
        0x0397: "H",
        0x0399: "I",
        0x039A: "K",
        0x039C: "M",
        0x039D: "N",
        0x039F: "O",
        0x03A1: "P",
        0x03A4: "T",
        0x03A5: "Y",
        0x03A7: "X",
    }
)


@dataclass(frozen=True, slots=True)
class InputVerdict:
    """The outcome of every input rule, plus the query to actually use.

    `query` is the normalised text after any redaction, and it is what retrieval and
    generation must use: reaching back to the raw user text would resurrect exactly
    the identifiers the secrets rule stripped.
    """

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

        # The secrets rule may rewrite the query (redaction), so everything past
        # this point, including the scope check and the verdict itself, must use
        # the rewritten text or the redacted value would still reach the embedder
        # and the model.
        checked_query, secret_decisions = self._check_secrets(normalized)
        decisions.extend(secret_decisions)
        if any(not d.allowed for d in secret_decisions):
            return InputVerdict(query=normalized, decisions=tuple(decisions))

        decisions.append(self._check_scope(checked_query))
        return InputVerdict(query=checked_query, decisions=tuple(decisions))

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
        # Patterns run against a homoglyph-folded scan copy; the query itself is
        # never rewritten with the fold. Evidence therefore quotes the folded view,
        # which is what the pattern actually matched.
        scanned = query.translate(_HOMOGLYPHS)
        for pattern, rule in _INJECTION:
            if match := pattern.search(scanned):
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

    def _check_secrets(self, query: str) -> tuple[str, tuple[Decision, ...]]:
        """Deny credentials, redact personal identifiers, allow the rest.

        Returns the query to keep using, which differs from the input when a
        redact-class match was scrubbed out. Two classes are redacted rather than
        denied: email addresses (author-lookup questions legitimately quote
        published contact addresses) and Luhn-valid card numbers (the digits are
        removable without destroying the question). Credential-shaped input stays
        a hard deny.
        """
        decisions: list[Decision] = []
        result = query
        for pattern, rule in _SECRETS:
            if rule == "card-number-like":
                result, count = _redact_card_numbers(pattern, result)
            elif rule == "email-address":
                result, count = pattern.subn(REDACTION, result)
            elif pattern.search(result):
                # Deliberately no evidence field: recording the match would write the
                # secret into the log line this decision ends up in.
                return query, (
                    Decision.deny(
                        f"input.secret.{rule}",
                        "That question appears to contain a credential or personal identifier, "
                        "so I did not send it on. Please remove it and ask again.",
                    ),
                )
            else:
                continue
            if count:
                # Count-only evidence, for the same reason the deny path records
                # none: the matched value must never land in a log line.
                decisions.append(
                    Decision.modify(
                        f"input.secret.{rule}",
                        "A personal identifier was redacted from the question before "
                        "it was processed.",
                        evidence=f"{rule}x{count}",
                    )
                )
        if not decisions:
            decisions.append(Decision.allow("input.secret"))
        return result, tuple(decisions)

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
    r"""NFKC-normalise, drop format characters, strip controls, collapse whitespace.

    Unicode normalisation is a security step, not a cosmetic one. NFKC folds
    full-width and other compatibility variants; deleting Cf (format) characters
    removes zero-width and bidi controls that would otherwise split a word like
    "ignore" invisibly. Cf characters are deleted rather than replaced with a space
    precisely so the split word rejoins. Cc control characters must NOT be deleted
    the same way: \n and \t are Cc, and deleting them would glue adjacent words
    together, so they become spaces instead. Cyrillic and Greek lookalike letters
    survive every step here; those are folded separately, and only on a scan-only
    copy, via _HOMOGLYPHS.
    """
    text = unicodedata.normalize("NFKC", query or "")
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = _CONTROL_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum. The card regex is only a candidate finder; this confirms it."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _redact_card_numbers(pattern: re.Pattern[str], text: str) -> tuple[str, int]:
    """Redact digit runs that actually look like payment cards.

    The regex alone also matches ISBNs, lists of years, and epoch-millisecond
    timestamps, all ordinary research query content, so a match is only a candidate.
    A run is redacted when the separator-stripped digits pass the Luhn checksum;
    978/979 bookland (ISBN-13) runs are exempt outright. Non-card digit runs are
    left in place because they usually ARE the question.
    """
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        run = match.group(0)
        digits = _CARD_SEPARATORS.sub("", run)
        if len(digits) == 13 and digits.startswith(_ISBN_PREFIXES):
            return run
        if not _luhn_valid(digits):
            return run
        count += 1
        # The regex may swallow one trailing separator ("4111 ... 1111 was" matches
        # up to and including the space); keep it so redaction never glues words.
        return REDACTION + run[len(run.rstrip(" -")) :]

    return pattern.sub(_replace, text), count


def scan_for_injection(texts: Sequence[str]) -> list[tuple[int, str, str]]:
    """Find injection patterns in retrieved chunks.

    This is the check the notebook version was missing. Guarding the query defends
    against a hostile user; the more realistic threat in a document-grounded system
    is a hostile *document*, because retrieved text enters the same prompt with none
    of the user's text having been involved.

    Chunks get the same scan view as queries (normalisation plus homoglyph fold): a
    hostile document can hide zero-width characters or Cyrillic lookalikes just as
    easily as a hostile user can. Reported evidence quotes the folded view.
    """
    findings: list[tuple[int, str, str]] = []
    for index, text in enumerate(texts):
        scanned = normalize_query(text).translate(_HOMOGLYPHS)
        for pattern, rule in _INJECTION:
            if match := pattern.search(scanned):
                findings.append((index, rule, preview(match.group(0), 60)))
                break
    return findings
