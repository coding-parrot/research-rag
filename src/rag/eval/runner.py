"""The eval runner.

Runs the golden and adversarial sets through any object with an `ask(question)`
method, computes the deterministic metrics, optionally the judge metrics, checks
thresholds, and writes a run report.

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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from rag.config import Config, EvalConfig
from rag.domain import Answer, AnswerStatus
from rag.eval.datasets import GoldenItem, GoldenSet, coverage_report
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
    items: list[ItemResult] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)
    checks: list[ThresholdCheck] = field(default_factory=list)
    judge_agreement: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

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
            run_id=time.strftime("%Y%m%d-%H%M%S"),
            config_hash=self._config.hash(),
            git_sha=_git_sha(),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            coverage=coverage_report(golden),
        )

        gateable = golden.items if gate_on_unreviewed else golden.reviewed
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
        return report

    # ------------------------------------------------------------------ #

    def _run_item(self, pipeline: AskFn, item: GoldenItem, *, with_judge: bool) -> ItemResult:
        start = time.perf_counter()
        answer = pipeline.ask(item.question)
        elapsed_ms = (time.perf_counter() - start) * 1000

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
            expected_refusal=outcome.expected_refusal,
            actually_refused=outcome.actually_refused,
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

        outcomes = [
            RefusalOutcome(
                item_id=r.item_id,
                expected_refusal=r.expected_refusal,
                actually_refused=r.actually_refused,
            )
            for r in all_gateable
        ]
        stats = refusal_stats(outcomes)

        aggregates = {
            "recall_at_k": mean([r.recall_at_k for r in answerable]),
            "mrr": mean([r.mrr for r in answerable]),
            "ndcg": mean([r.ndcg for r in answerable]),
            "context_precision": mean([r.context_precision for r in answerable]),
            "citation_validity": mean([r.citation_validity for r in answerable]),
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
    except OSError:
        return "unknown"


def environment_fingerprint() -> dict[str, str]:
    return {"python": platform.python_version(), "platform": platform.platform()}
