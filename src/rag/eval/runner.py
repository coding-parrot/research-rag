"""The eval runner.

Runs the golden set (whose adversarial coverage lives in its `adversarial`
category items) through any object with an `ask(question)` method, computes the
deterministic metrics, optionally the judge metrics, checks thresholds, and
writes a run report.

Built against the pipeline *interface*, not the pipeline: the harness's own tests
run it against a fake pipeline, so the harness is trusted before there is anything
real to measure. Results are JSON in `evals/results/`, committed, so a regression
shows up as a diff in review rather than as a memory.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from rag.config import Config, EvalConfig
from rag.domain import Answer, AnswerStatus
from rag.eval.datasets import GoldenItem, GoldenSet, coverage_report, load_judge_calibration
from rag.eval.judge import Judge
from rag.eval.metrics import (
    RefusalOutcome,
    citation_validity,
    context_precision,
    mean,
    mrr,
    must_cite_satisfied,
    ndcg_at_k,
    recall_at_k,
    refusal_outcome,
    refusal_stats,
)
from rag.observability import get_logger, timed

log = get_logger("eval")


class AskFn(Protocol):
    def ask(self, question: str) -> Answer: ...


@dataclass(frozen=True, slots=True)
class ItemResult:
    item_id: str
    category: str
    status: str
    recall_at_k: float
    mrr: float
    ndcg: float
    context_precision: float
    citation_validity: float
    must_cite_satisfied: float
    expected_refusal: bool
    actually_refused: bool
    # True when the non-OK status came from an infrastructure failure
    # (generate.error), not a policy refusal; refusal aggregates skip these.
    infra_error: bool = False
    faithfulness: float | None = None
    correctness: float | None = None
    elapsed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ThresholdCheck:
    name: str
    value: float
    threshold: float
    higher_is_better: bool = True

    @property
    def passed(self) -> bool:
        return (
            self.value >= self.threshold if self.higher_is_better else self.value <= self.threshold
        )


@dataclass(slots=True)
class EvalReport:
    run_id: str
    config_hash: str
    git_sha: str
    started_at: str
    coverage: dict[str, int]
    environment: dict[str, str] = field(default_factory=dict)
    items: list[ItemResult] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)
    checks: list[ThresholdCheck] = field(default_factory=list)
    judge_agreement: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # An empty check list must not pass: a run that enforced nothing (no
        # reviewed items, or none answerable) would otherwise keep CI green
        # while measuring nothing.
        return bool(self.checks) and all(c.passed for c in self.checks)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["passed"] = self.passed
        return json.dumps(payload, indent=2, sort_keys=True)


class EvalRunner:
    def __init__(self, config: Config, *, judge: Judge | None = None) -> None:
        self._config = config
        self._judge = judge

    def run(
        self,
        pipeline: AskFn,
        golden: GoldenSet,
        *,
        gate_on_unreviewed: bool = False,
        with_judge: bool = False,
    ) -> EvalReport:
        eval_config = self._config.eval
        report = EvalReport(
            # gmtime keeps the run_id consistent with started_at; the random
            # suffix keeps two same-second runs (parallel CI shards, fast
            # fake-pipeline runs) from sharing a report file.
            run_id=f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}",
            config_hash=self._config.hash(),
            git_sha=_git_sha(),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            coverage=coverage_report(golden),
            environment=environment_fingerprint(),
        )

        gateable = (
            golden.items if gate_on_unreviewed else tuple(i for i in golden.items if i.gateable)
        )
        if not gateable:
            report.notes.append(
                "No reviewed items: thresholds were not enforced. Review the golden set."
            )
        skipped = len(golden.items) - len(gateable)
        if skipped:
            report.notes.append(f"{skipped} unreviewed items were run but did not gate.")

        with timed(log, "eval.run", items=len(golden.items)):
            for item in golden.items:
                report.items.append(self._run_item(pipeline, item, with_judge=with_judge))

        self._aggregate(report, golden, gateable, eval_config, with_judge=with_judge)

        if with_judge and self._judge is not None:
            self._calibrate_judge(report, self._judge)

        if not report.checks:
            report.notes.append(
                "No threshold checks were produced, so this report cannot pass: "
                "nothing was enforced."
            )
        return report

    def _calibrate_judge(self, report: EvalReport, judge: Judge) -> None:
        """Measure judge agreement against human labels.

        The judge is a measurement instrument (see judge.py); its gated numbers
        carry a trust signal only when the agreement rate ships in the report.
        """
        path = self._config.paths.evals / "judge_calibration.yaml"
        if not path.exists():
            report.notes.append(
                f"Judge calibration file not found ({path}): judge_agreement not measured."
            )
            return
        # A malformed committed file raises ManifestError loudly, like every other
        # dataset. Only the model calls themselves degrade to a note.
        examples = load_judge_calibration(path)
        try:
            result = judge.calibrate(examples)
        except Exception as exc:
            log.warning("judge calibration failed", fields={"error": str(exc)})
            report.notes.append(f"Judge calibration failed: {exc}")
            return
        report.judge_agreement = result.agreement_rate

    # ------------------------------------------------------------------ #

    def _run_item(self, pipeline: AskFn, item: GoldenItem, *, with_judge: bool) -> ItemResult:
        start = time.perf_counter()
        answer = pipeline.ask(item.question)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # None marks an infrastructure failure (generate.error): recorded on the
        # item so refusal aggregates can exclude it.
        outcome = refusal_outcome(item, answer)
        stats = citation_validity(answer)
        top_k = self._config.retrieve.top_k

        faithfulness: float | None = None
        correctness: float | None = None
        if with_judge and self._judge is not None and answer.status is AnswerStatus.OK:
            try:
                faithfulness = self._judge.faithfulness(answer).score
                if item.reference_answer:
                    correctness = self._judge.correctness(
                        item.question, item.reference_answer, answer
                    ).score
            except Exception as exc:
                log.warning("judge failed on item", fields={"item": item.id, "error": str(exc)})

        return ItemResult(
            item_id=item.id,
            category=item.category.value,
            status=answer.status.value,
            recall_at_k=recall_at_k(answer.retrieved, item.must_cite, top_k),
            mrr=mrr(answer.retrieved, item.must_cite),
            ndcg=ndcg_at_k(answer.retrieved, item.must_cite, top_k),
            context_precision=context_precision(answer.retrieved, item.must_cite),
            citation_validity=stats.validity_rate,
            must_cite_satisfied=must_cite_satisfied(answer, item.must_cite),
            expected_refusal=item.category.expects_refusal,
            actually_refused=answer.status is not AnswerStatus.OK,
            infra_error=outcome is None,
            faithfulness=faithfulness,
            correctness=correctness,
            elapsed_ms=round(elapsed_ms, 1),
        )

    def _aggregate(
        self,
        report: EvalReport,
        golden: GoldenSet,
        gateable: Sequence[GoldenItem],
        eval_config: EvalConfig,
        *,
        with_judge: bool,
    ) -> None:
        by_id = {r.item_id: r for r in report.items}
        gate_ids = {i.id for i in gateable}
        answerable = [by_id[i.id] for i in golden.answerable if i.id in by_id and i.id in gate_ids]
        all_gateable = [by_id[i.id] for i in gateable if i.id in by_id]

        # Items that failed on infrastructure (generate.error) are outages, not
        # refusals: they neither credit refusal_recall nor charge false refusals.
        outcomes = [
            RefusalOutcome(
                item_id=r.item_id,
                expected_refusal=r.expected_refusal,
                actually_refused=r.actually_refused,
            )
            for r in all_gateable
            if not r.infra_error
        ]
        stats = refusal_stats(outcomes)
        infra_errors = sum(1 for r in report.items if r.infra_error)
        if infra_errors:
            report.notes.append(
                f"{infra_errors} items failed on infrastructure errors (generate.error) "
                "and were excluded from refusal stats."
            )

        # Citation validity is only defined for items that actually answered:
        # refused or blocked items carry no citations, and their vacuous 1.0
        # would let a pipeline that refuses most questions prop up the gate.
        # An empty denominator scores 0.0, which fails safe.
        answered = [r for r in answerable if r.status == AnswerStatus.OK.value]

        aggregates = {
            "recall_at_k": mean([r.recall_at_k for r in answerable]),
            "mrr": mean([r.mrr for r in answerable]),
            "ndcg": mean([r.ndcg for r in answerable]),
            "context_precision": mean([r.context_precision for r in answerable]),
            "citation_validity": mean([r.citation_validity for r in answered]),
            "must_cite_satisfied": mean([r.must_cite_satisfied for r in answerable]),
            "refusal_precision": stats.refusal_precision,
            "refusal_recall": stats.refusal_recall,
            "false_refusal_rate": stats.false_refusal_rate,
        }
        if with_judge:
            faith = [r.faithfulness for r in answerable if r.faithfulness is not None]
            correct = [r.correctness for r in answerable if r.correctness is not None]
            if faith:
                aggregates["faithfulness"] = mean(faith)
            if correct:
                aggregates["correctness"] = mean(correct)

        report.aggregates = {k: round(v, 4) for k, v in aggregates.items()}

        if answerable:
            report.checks = [
                ThresholdCheck(
                    "recall_at_k", aggregates["recall_at_k"], eval_config.min_recall_at_k
                ),
                ThresholdCheck("mrr", aggregates["mrr"], eval_config.min_mrr),
                ThresholdCheck(
                    "citation_validity",
                    aggregates["citation_validity"],
                    eval_config.min_citation_validity,
                ),
                ThresholdCheck(
                    "false_refusal_rate",
                    aggregates["false_refusal_rate"],
                    eval_config.max_false_refusal_rate,
                    higher_is_better=False,
                ),
            ]
            if with_judge and "faithfulness" in aggregates:
                report.checks.append(
                    ThresholdCheck(
                        "faithfulness", aggregates["faithfulness"], eval_config.min_faithfulness
                    )
                )
            if with_judge and "correctness" in aggregates:
                report.checks.append(
                    ThresholdCheck(
                        "correctness", aggregates["correctness"], eval_config.min_answer_correctness
                    )
                )


def save_report(report: EvalReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"eval-{report.run_id}.json"
    path.write_text(report.to_json())
    latest = directory / "latest.json"
    latest.write_text(report.to_json())
    log.info("report written", fields={"path": str(path), "passed": report.passed})
    return path


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        # TimeoutExpired is a SubprocessError, not an OSError: a hung git must
        # degrade to "unknown", not crash the run before any item executes.
        return "unknown"


def environment_fingerprint() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform()}
