"""Deterministic eval metrics.

Everything in this module is pure arithmetic over ids and strings: no model, no
network, no API cost. These run on every PR and gate CI with hard thresholds. The
LLM-judge metrics live in judge.py and run nightly, because a metric that costs
money per run is a metric that stops being run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag.domain import Answer, AnswerStatus, Chunk, Scored
from rag.eval.datasets import GoldenItem, HeaderLabelItem, MustCite

# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #


def chunk_matches(chunk: Chunk, target: MustCite) -> bool:
    """Does a retrieved chunk satisfy a must-cite requirement?

    Paper must match exactly. Section, when specified, matches as a case-insensitive
    substring of the section label, so "3.2" matches "3.2 Selective Scan" and
    "experiments" matches "4 Experiments".
    """
    if chunk.doc_id != target.paper:
        return False
    if target.section is None:
        return True
    return target.section.lower() in chunk.section_label.lower()


def recall_at_k(retrieved: Sequence[Scored], targets: Sequence[MustCite], k: int) -> float:
    """Fraction of required citations that appear in the top-k retrieved chunks."""
    if not targets:
        return 1.0
    top = [s.chunk for s in retrieved[:k]]
    hit = sum(1 for t in targets if any(chunk_matches(c, t) for c in top))
    return hit / len(targets)


def mrr(retrieved: Sequence[Scored], targets: Sequence[MustCite]) -> float:
    """Reciprocal rank of the first chunk satisfying any requirement."""
    for position, scored in enumerate(retrieved, start=1):
        if any(chunk_matches(scored.chunk, t) for t in targets):
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[Scored], targets: Sequence[MustCite], k: int) -> float:
    """Binary-relevance nDCG. Order within the top-k matters, recall alone does not."""
    if not targets:
        return 1.0
    gains = [1.0 if any(chunk_matches(s.chunk, t) for t in targets) else 0.0 for s in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(len(targets), k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / ideal if ideal > 0 else 0.0


def context_precision(retrieved: Sequence[Scored], targets: Sequence[MustCite]) -> float:
    """Fraction of retrieved chunks that were actually needed.

    Low precision with high recall means the context window is mostly filler, which
    costs tokens and dilutes the model's attention.
    """
    if not retrieved:
        return 0.0
    useful = sum(1 for s in retrieved if any(chunk_matches(s.chunk, t) for t in targets))
    return useful / len(retrieved)


# --------------------------------------------------------------------------- #
# Citation metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CitationStats:
    proposed: int
    valid: int

    @property
    def validity_rate(self) -> float:
        return self.valid / self.proposed if self.proposed else 1.0


def citation_validity(answer: Answer) -> CitationStats:
    """Surviving citations over proposed citations.

    The output guard drops invalid citations and records each drop as a MODIFY
    decision, so proposed = surviving + dropped.
    """
    dropped = sum(
        1
        for d in answer.decisions
        if d.rule_id.startswith("output.citation.") and d.rule_id != "output.citation"
    )
    valid = len(answer.citations)
    return CitationStats(proposed=valid + dropped, valid=valid)


def must_cite_satisfied(answer: Answer, targets: Sequence[MustCite]) -> float:
    """Fraction of required citations the final answer actually cites."""
    if not targets:
        return 1.0
    by_id = {s.chunk.chunk_id: s.chunk for s in answer.retrieved}
    cited_chunks = [by_id[c.chunk_id] for c in answer.citations if c.chunk_id in by_id]
    hit = sum(1 for t in targets if any(chunk_matches(c, t) for c in cited_chunks))
    return hit / len(targets)


# --------------------------------------------------------------------------- #
# Guardrail metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RefusalOutcome:
    item_id: str
    expected_refusal: bool
    actually_refused: bool

    @property
    def correct(self) -> bool:
        return self.expected_refusal == self.actually_refused


def refusal_outcome(item: GoldenItem, answer: Answer) -> RefusalOutcome:
    return RefusalOutcome(
        item_id=item.id,
        expected_refusal=item.category.expects_refusal,
        actually_refused=answer.status is not AnswerStatus.OK,
    )


@dataclass(frozen=True, slots=True)
class RefusalStats:
    """Precision and recall of refusing.

    False refusal rate is the one that degrades silently: every guardrail tightening
    improves the adversarial numbers and quietly worsens this one.
    """

    true_refusals: int
    false_refusals: int
    missed_refusals: int
    correct_answers: int

    @property
    def refusal_precision(self) -> float:
        denominator = self.true_refusals + self.false_refusals
        return self.true_refusals / denominator if denominator else 1.0

    @property
    def refusal_recall(self) -> float:
        denominator = self.true_refusals + self.missed_refusals
        return self.true_refusals / denominator if denominator else 1.0

    @property
    def false_refusal_rate(self) -> float:
        denominator = self.false_refusals + self.correct_answers
        return self.false_refusals / denominator if denominator else 0.0


def refusal_stats(outcomes: Sequence[RefusalOutcome]) -> RefusalStats:
    return RefusalStats(
        true_refusals=sum(1 for o in outcomes if o.expected_refusal and o.actually_refused),
        false_refusals=sum(1 for o in outcomes if not o.expected_refusal and o.actually_refused),
        missed_refusals=sum(1 for o in outcomes if o.expected_refusal and not o.actually_refused),
        correct_answers=sum(
            1 for o in outcomes if not o.expected_refusal and not o.actually_refused
        ),
    )


# --------------------------------------------------------------------------- #
# Header-detection metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HeaderScore:
    doc_id: str
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def _normalise_label(label: str) -> str:
    """'3. Experiments' and '3 Experiments' and '3  experiments' are the same label."""
    text = label.lower().strip()
    text = re.sub(r"^(\d+(?:\.\d+)*)\.?\s+", r"\1 ", text)
    return re.sub(r"\s+", " ", text)


def score_headers(detected: Sequence[str], label: HeaderLabelItem) -> HeaderScore:
    """Compare detected section labels against the hand-labelled truth.

    This is the metric that catches a Surya version bump, a changed threshold, or a
    new PDF style silently degrading everything downstream. It runs in CI against
    cached OCR fixtures, so it is deterministic and free.
    """
    truth = {_normalise_label(s) for s in label.sections}
    found = {_normalise_label(s) for s in detected}
    if not found:
        return HeaderScore(doc_id=label.doc_id, precision=0.0, recall=0.0)
    true_positives = len(truth & found)
    return HeaderScore(
        doc_id=label.doc_id,
        precision=true_positives / len(found),
        recall=true_positives / len(truth),
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
