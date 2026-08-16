"""LLM-as-judge metrics.

Two graded dimensions: faithfulness (is every claim supported by the retrieved
passages) and correctness (does the answer convey the reference answer).

The judge is itself a measurement instrument, so it gets calibrated: `calibrate()`
runs it against human-labelled examples and reports the agreement rate. A judge
below its agreement threshold means the judge's numbers are noise, and the report
says so rather than printing them as truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.domain import Answer, Usage
from rag.errors import LlmError
from rag.eval.metrics import mean
from rag.generate.client import LlmClient, LlmRequest
from rag.generate.prompts import (
    CORRECTNESS_PROMPT,
    CORRECTNESS_SCHEMA,
    FAITHFULNESS_PROMPT,
    FAITHFULNESS_SCHEMA,
    format_passages,
)
from rag.observability import get_logger

log = get_logger("judge")


@dataclass(frozen=True, slots=True)
class FaithfulnessVerdict:
    faithful: bool
    supported_claims: int
    unsupported_claims: int
    explanation: str
    usage: Usage

    @property
    def score(self) -> float:
        total = self.supported_claims + self.unsupported_claims
        return self.supported_claims / total if total else 1.0


@dataclass(frozen=True, slots=True)
class CorrectnessVerdict:
    score: float
    explanation: str
    usage: Usage


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Agreement between the judge and human labels on a labelled sample."""

    total: int
    agreements: int

    @property
    def agreement_rate(self) -> float:
        return self.agreements / self.total if self.total else 0.0


class Judge:
    def __init__(self, client: LlmClient, *, model: str = "", effort: str = "high") -> None:
        self._client = client
        self._model = model
        self._effort = effort

    def faithfulness(self, answer: Answer) -> FaithfulnessVerdict:
        """Grade whether the answer's claims are supported by what was retrieved."""
        passages = format_passages(answer.retrieved)
        prompt = FAITHFULNESS_PROMPT.format(passages=passages, answer=answer.text)
        response = self._client.complete(
            LlmRequest(
                prompt=prompt,
                model=self._model,
                effort=self._effort,
                max_tokens=2048,
                schema=FAITHFULNESS_SCHEMA,
            )
        )
        parsed = response.parsed
        if parsed is None:
            raise LlmError(
                f"faithfulness judge returned unparseable output: {response.text[:200]!r}"
            )
        return FaithfulnessVerdict(
            faithful=str(parsed.get("verdict", "")) == "faithful",
            supported_claims=int(parsed.get("supported_claims", 0)),
            unsupported_claims=int(parsed.get("unsupported_claims", 0)),
            explanation=str(parsed.get("explanation", "")),
            usage=response.usage,
        )

    def correctness(self, question: str, reference: str, answer: Answer) -> CorrectnessVerdict:
        """Grade the answer against the golden reference answer."""
        prompt = CORRECTNESS_PROMPT.format(
            question=question, reference=reference, answer=answer.text
        )
        response = self._client.complete(
            LlmRequest(
                prompt=prompt,
                model=self._model,
                effort=self._effort,
                max_tokens=1024,
                schema=CORRECTNESS_SCHEMA,
            )
        )
        parsed = response.parsed
        if parsed is None:
            raise LlmError(
                f"correctness judge returned unparseable output: {response.text[:200]!r}"
            )
        score = float(parsed.get("score", 0.0))
        return CorrectnessVerdict(
            score=max(0.0, min(1.0, score)),
            explanation=str(parsed.get("explanation", "")),
            usage=response.usage,
        )

    # ------------------------------------------------------------------ #

    def calibrate(self, labelled: Sequence[Mapping[str, object]]) -> CalibrationResult:
        """Run the faithfulness judge against human-labelled examples.

        Each example: {"answer": Answer-like content, "passages": str, "human_verdict":
        "faithful" | "unfaithful"}. Reported as an agreement rate alongside every
        judge metric, so a reader knows how much to trust the judge's numbers.
        """
        agreements = 0
        for example in labelled:
            prompt = FAITHFULNESS_PROMPT.format(
                passages=str(example["passages"]), answer=str(example["answer"])
            )
            response = self._client.complete(
                LlmRequest(
                    prompt=prompt,
                    model=self._model,
                    effort=self._effort,
                    max_tokens=2048,
                    schema=FAITHFULNESS_SCHEMA,
                )
            )
            parsed = response.parsed or {}
            if str(parsed.get("verdict", "")) == str(example["human_verdict"]):
                agreements += 1
        return CalibrationResult(total=len(labelled), agreements=agreements)


def summarize_faithfulness(verdicts: Sequence[FaithfulnessVerdict]) -> dict[str, float]:
    return {
        "faithful_rate": mean([1.0 if v.faithful else 0.0 for v in verdicts]),
        "claim_support_rate": mean([v.score for v in verdicts]),
    }


def summarize_correctness(verdicts: Sequence[CorrectnessVerdict]) -> dict[str, float]:
    return {"mean_correctness": mean([v.score for v in verdicts])}
