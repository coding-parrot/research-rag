"""Eval harness tests: the harness is itself tested against a scripted pipeline
before it is ever pointed at the real one."""

from __future__ import annotations

import json
import logging
import subprocess

import pytest

from rag.config import Config
from rag.domain import Answer, AnswerStatus, Chunk, Citation, Decision, Scored, make_chunk_id
from rag.errors import ManifestError
from rag.eval.datasets import (
    GoldenItem,
    GoldenSet,
    HeaderLabelItem,
    MustCite,
    QuestionCategory,
    load_golden,
    load_judge_calibration,
)
from rag.eval.judge import Judge
from rag.eval.metrics import (
    chunk_matches,
    citation_validity,
    context_precision,
    mrr,
    must_cite_satisfied,
    ndcg_at_k,
    recall_at_k,
    refusal_outcome,
    refusal_stats,
    score_headers,
)
from rag.eval.runner import EvalRunner, _git_sha, save_report
from rag.generate.client import LlmResponse
from rag.observability import JsonFormatter, get_logger, timed


def _chunk(doc_id: str, section: str, number: str | None = "3") -> Chunk:
    text = f"content of {section} in {doc_id}"
    return Chunk(
        chunk_id=make_chunk_id(doc_id, 0, len(text), 0),
        doc_id=doc_id,
        doc_title=doc_id.title(),
        text=text,
        char_start=0,
        char_end=len(text),
        section_title=section,
        section_number=number,
        page_start=1,
        page_end=1,
    )


def _scored(*chunks: Chunk) -> tuple[Scored, ...]:
    return tuple(
        Scored(chunk=c, score=1.0 - i * 0.1, rank=i + 1, retriever="t")
        for i, c in enumerate(chunks)
    )


class TestRetrievalMetrics:
    def test_recall(self):
        retrieved = _scored(_chunk("mamba", "Selective Scan"), _chunk("bert", "Experiments"))
        targets = [MustCite(paper="mamba"), MustCite(paper="lora")]
        assert recall_at_k(retrieved, targets, k=2) == 0.5

    def test_section_substring_match(self):
        retrieved = _scored(_chunk("mamba", "Selective Scan Mechanism", "3.2"))
        assert recall_at_k(retrieved, [MustCite(paper="mamba", section="3.2")], k=1) == 1.0
        assert (
            recall_at_k(retrieved, [MustCite(paper="mamba", section="selective scan")], k=1) == 1.0
        )
        assert recall_at_k(retrieved, [MustCite(paper="mamba", section="4.1")], k=1) == 0.0

    def test_numeric_section_targets_use_number_prefix_semantics(self):
        # Regression: substring matching let target "1" match "10 Related Work"
        # and "3.2" match "13.2", inflating every retrieval metric.
        assert not chunk_matches(_chunk("mamba", "Related Work", "10"), MustCite("mamba", "1"))
        assert not chunk_matches(_chunk("mamba", "Ablations", "13.2"), MustCite("mamba", "3.2"))
        assert not chunk_matches(_chunk("mamba", "Extras", "30"), MustCite("mamba", "3"))
        assert chunk_matches(_chunk("mamba", "Setup", "3.1"), MustCite("mamba", "3"))
        assert chunk_matches(_chunk("mamba", "Setup", "3.2"), MustCite("mamba", "3.2"))
        # Numeric targets need a section number to match at all.
        assert not chunk_matches(_chunk("mamba", "Abstract", None), MustCite("mamba", "1"))

    def test_split_part_suffix_does_not_match_numeric_target(self):
        # Regression: the "(part 1/2)" suffix on split chunks used to satisfy a
        # section "1" target via the full-label substring match.
        text = "appendix content"
        part = Chunk(
            chunk_id=make_chunk_id("mamba", 0, len(text), 0),
            doc_id="mamba",
            doc_title="Mamba",
            text=text,
            char_start=0,
            char_end=len(text),
            section_title="Appendix",
            section_number="7",
            page_start=9,
            page_end=9,
            part_index=0,
            part_count=2,
        )
        assert "(part 1/2)" in part.section_label
        assert not chunk_matches(part, MustCite(paper="mamba", section="1"))
        assert chunk_matches(part, MustCite(paper="mamba", section="7"))
        # Textual targets still match on the title.
        assert chunk_matches(part, MustCite(paper="mamba", section="appendix"))

    def test_mrr_position(self):
        retrieved = _scored(_chunk("bert", "Intro"), _chunk("mamba", "Scan"))
        assert mrr(retrieved, [MustCite(paper="mamba")]) == 0.5
        assert mrr(retrieved, [MustCite(paper="absent")]) == 0.0

    def test_ndcg_rewards_early_hits(self):
        hit_first = _scored(_chunk("mamba", "Scan"), _chunk("bert", "Intro"))
        hit_second = _scored(_chunk("bert", "Intro"), _chunk("mamba", "Scan"))
        target = [MustCite(paper="mamba")]
        assert ndcg_at_k(hit_first, target, 2) > ndcg_at_k(hit_second, target, 2)

    def test_ndcg_bounded_when_multiple_chunks_match_one_target(self):
        # Regression: every matching chunk used to gain 1.0, so four chunks of
        # one relevant paper scored ndcg 2.56 and top-k stuffing beat perfection.
        duplicates = _scored(*[_chunk("mamba", f"Scan part {i}") for i in range(4)])
        target = [MustCite(paper="mamba")]
        assert ndcg_at_k(duplicates, target, 4) == 1.0
        two_targets = [MustCite(paper="mamba"), MustCite(paper="bert")]
        score = ndcg_at_k(duplicates, two_targets, 4)
        assert 0.0 < score < 1.0  # second target never satisfied, duplicates gain nothing

    def test_context_precision(self):
        retrieved = _scored(_chunk("mamba", "Scan"), _chunk("bert", "Intro"))
        assert context_precision(retrieved, [MustCite(paper="mamba")]) == 0.5


class TestAnswerMetrics:
    def test_citation_validity_counts_drops(self):
        answer = Answer(
            status=AnswerStatus.OK,
            text="x",
            citations=(Citation(chunk_id="a", quote="q", label="A"),),
            decisions=(
                Decision.modify("output.citation.unknown_chunk", "dropped"),
                Decision.allow("output.citation", "1/2"),
            ),
        )
        stats = citation_validity(answer)
        assert stats.proposed == 2
        assert stats.valid == 1
        assert stats.validity_rate == 0.5

    def test_none_valid_verdict_row_is_not_a_dropped_citation(self):
        # Regression: the DENY 'output.citation.none_valid' verdict row used to
        # count as a proposed citation, so an answer that proposed nothing
        # reported validity 0.0 instead of the documented 1.0.
        answer = Answer(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            text="",
            citations=(),
            decisions=(Decision.deny("output.citation.none_valid", "no citations"),),
        )
        stats = citation_validity(answer)
        assert stats.proposed == 0
        assert stats.validity_rate == 1.0

    def test_citation_validity_scores_final_attempt_only(self):
        # Regression: the answerer accumulates decisions across regenerations, so
        # a failed first attempt's drops depressed a fully valid retry to 0.4.
        retry_all_valid = Answer(
            status=AnswerStatus.OK,
            text="x",
            citations=(
                Citation(chunk_id="a", quote="q", label="A"),
                Citation(chunk_id="b", quote="q", label="B"),
            ),
            decisions=(
                Decision.modify("output.citation.unknown_chunk", "dropped"),
                Decision.modify("output.citation.quote_not_found", "dropped"),
                Decision.deny("output.citation.none_valid", "attempt 1 failed"),
                Decision.allow("output.citation", "2/2"),
            ),
        )
        stats = citation_validity(retry_all_valid)
        assert stats.proposed == 2
        assert stats.valid == 2
        assert stats.validity_rate == 1.0

        retry_with_drop = Answer(
            status=AnswerStatus.OK,
            text="x",
            citations=(
                Citation(chunk_id="a", quote="q", label="A"),
                Citation(chunk_id="b", quote="q", label="B"),
            ),
            decisions=(
                Decision.modify("output.citation.unknown_chunk", "dropped"),
                Decision.deny("output.citation.none_valid", "attempt 1 failed"),
                Decision.modify("output.citation.quote_too_short", "dropped"),
                Decision.allow("output.citation", "2/3"),
            ),
        )
        stats = citation_validity(retry_with_drop)
        assert stats.proposed == 3
        assert stats.valid == 2
        assert stats.validity_rate == pytest.approx(2 / 3)

    def test_must_cite_satisfied(self):
        chunk = _chunk("mamba", "Selective Scan", "3.2")
        answer = Answer(
            status=AnswerStatus.OK,
            text="x",
            citations=(Citation(chunk_id=chunk.chunk_id, quote="q", label=""),),
            retrieved=_scored(chunk),
        )
        assert must_cite_satisfied(answer, [MustCite(paper="mamba", section="3.2")]) == 1.0
        assert must_cite_satisfied(answer, [MustCite(paper="lora")]) == 0.0


class TestRefusalMetrics:
    def _item(self, category: QuestionCategory) -> GoldenItem:
        return GoldenItem(id="i", question="q", category=category)

    def _answer(self, status: AnswerStatus) -> Answer:
        return Answer(status=status, text="")

    def test_outcomes(self):
        adversarial = self._item(QuestionCategory.ADVERSARIAL)
        factual = self._item(QuestionCategory.FACTUAL)
        assert refusal_outcome(adversarial, self._answer(AnswerStatus.BLOCKED_INPUT)).correct
        assert not refusal_outcome(factual, self._answer(AnswerStatus.NO_RESULTS)).correct

    def test_stats(self):
        outcomes = [
            refusal_outcome(
                self._item(QuestionCategory.ADVERSARIAL), self._answer(AnswerStatus.BLOCKED_INPUT)
            ),
            refusal_outcome(
                self._item(QuestionCategory.ADVERSARIAL), self._answer(AnswerStatus.OK)
            ),
            refusal_outcome(self._item(QuestionCategory.FACTUAL), self._answer(AnswerStatus.OK)),
            refusal_outcome(
                self._item(QuestionCategory.FACTUAL), self._answer(AnswerStatus.NO_RESULTS)
            ),
        ]
        stats = refusal_stats(outcomes)
        assert stats.true_refusals == 1
        assert stats.missed_refusals == 1
        assert stats.false_refusals == 1
        assert stats.false_refusal_rate == 0.5

    def test_generate_error_is_an_outage_not_a_refusal(self):
        # Regression: an LLM outage (deny generate.error -> non-OK status) used
        # to be credited as a true refusal on adversarial items and charged as a
        # false refusal on answerable ones.
        outage = Answer(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            text="",
            decisions=(Decision.deny("generate.error", "api down"),),
        )
        assert refusal_outcome(self._item(QuestionCategory.ADVERSARIAL), outage) is None
        assert refusal_outcome(self._item(QuestionCategory.FACTUAL), outage) is None


class TestHeaderScoring:
    def test_normalisation_tolerates_punctuation(self):
        label = HeaderLabelItem(doc_id="bert", sections=("1 Introduction", "2 Related Work"))
        score = score_headers(["1. Introduction", "2  related work"], label)
        assert score.precision == 1.0 and score.recall == 1.0 and score.f1 == 1.0

    def test_partial_detection(self):
        label = HeaderLabelItem(doc_id="bert", sections=("1 Intro", "2 Method", "3 Results"))
        score = score_headers(["1 Intro", "9 Bogus"], label)
        assert score.precision == 0.5
        assert score.recall == pytest.approx(1 / 3)


class TestGoldenLoading:
    def test_valid_file(self, tmp_path):
        path = tmp_path / "golden.yaml"
        path.write_text(
            """
questions:
  - id: q1
    category: factual
    question: How does LoRA work?
    reference_answer: Low-rank adapters.
    must_cite: [{paper: lora}]
    reviewed: true
  - id: q2
    category: out_of_scope
    question: Best pizza?
"""
        )
        golden = load_golden(path)
        assert len(golden) == 2
        assert len(golden.reviewed) == 1
        assert golden.answerable[0].id == "q1"

    def test_answerable_requires_reference_and_citations(self, tmp_path):
        path = tmp_path / "golden.yaml"
        path.write_text("questions:\n  - id: q1\n    category: factual\n    question: How?\n")
        with pytest.raises(ManifestError, match="reference_answer"):
            load_golden(path)

    def test_duplicate_ids_rejected(self, tmp_path):
        path = tmp_path / "golden.yaml"
        path.write_text(
            """
questions:
  - {id: q1, category: out_of_scope, question: a?}
  - {id: q1, category: out_of_scope, question: b?}
"""
        )
        with pytest.raises(ManifestError, match="duplicate"):
            load_golden(path)

    def test_repo_golden_file_is_valid(self):
        from rag.config import RepoRoot

        golden = load_golden(RepoRoot / "evals" / "golden" / "golden.yaml")
        assert len(golden) >= 15
        # Every category is represented, including the Surya-justifying one.
        assert golden.by_category(QuestionCategory.TABLE_LOOKUP)
        assert golden.by_category(QuestionCategory.ADVERSARIAL)

    def test_float_section_rejected(self, tmp_path):
        # Regression: unquoted `section: 3.10` parses as YAML float 3.1, silently
        # retargeting subsection ten at 3.1. The loader must refuse, not coerce.
        path = tmp_path / "golden.yaml"
        path.write_text(
            """
questions:
  - id: q1
    category: factual
    question: How?
    reference_answer: r
    must_cite: [{paper: mamba, section: 3.10}]
"""
        )
        with pytest.raises(ManifestError, match="quote"):
            load_golden(path)

    def test_int_section_coerced_to_string(self, tmp_path):
        path = tmp_path / "golden.yaml"
        path.write_text(
            """
questions:
  - id: q1
    category: factual
    question: How?
    reference_answer: r
    must_cite: [{paper: mamba, section: 3}]
"""
        )
        golden = load_golden(path)
        assert golden.items[0].must_cite[0].section == "3"

    def test_malformed_must_cite_entry_rejected(self, tmp_path):
        # Regression: non-mapping entries were silently dropped by an isinstance
        # filter, weakening the requirement without any error.
        path = tmp_path / "golden.yaml"
        path.write_text(
            """
questions:
  - id: q1
    category: out_of_scope
    question: Best pizza?
    must_cite: ["mamba section 3"]
"""
        )
        with pytest.raises(ManifestError, match="must_cite"):
            load_golden(path)


class TestJudgeCalibrationLoading:
    def test_valid_file(self, tmp_path):
        path = tmp_path / "judge_calibration.yaml"
        path.write_text(
            """
examples:
  - passages: "The sky is blue."
    answer: "The sky is blue."
    human_verdict: faithful
  - passages: "The sky is blue."
    answer: "The sky is green."
    human_verdict: unfaithful
"""
        )
        examples = load_judge_calibration(path)
        assert len(examples) == 2
        assert examples[0]["human_verdict"] == "faithful"

    def test_bad_verdict_rejected(self, tmp_path):
        path = tmp_path / "judge_calibration.yaml"
        path.write_text("examples:\n  - {passages: p, answer: a, human_verdict: maybe}\n")
        with pytest.raises(ManifestError, match="human_verdict"):
            load_judge_calibration(path)


class ScriptedPipeline:
    """Fake pipeline for harness tests: perfect on some ids, wrong on others."""

    def __init__(self, script: dict[str, Answer]) -> None:
        self._script = script
        self.asked: list[str] = []

    def ask(self, question: str) -> Answer:
        self.asked.append(question)
        return self._script.get(question, Answer(status=AnswerStatus.NO_RESULTS, text="nothing"))


class TestRunner:
    def _golden(self) -> GoldenSet:
        return GoldenSet(
            items=(
                GoldenItem(
                    id="good",
                    question="good question",
                    category=QuestionCategory.FACTUAL,
                    reference_answer="ref",
                    must_cite=(MustCite(paper="mamba"),),
                    reviewed=True,
                ),
                GoldenItem(
                    id="oos",
                    question="pizza?",
                    category=QuestionCategory.OUT_OF_SCOPE,
                    reviewed=True,
                ),
            )
        )

    def _perfect_answer(self) -> Answer:
        chunk = _chunk("mamba", "Scan")
        return Answer(
            status=AnswerStatus.OK,
            text="answer",
            citations=(Citation(chunk_id=chunk.chunk_id, quote="content of", label="Mamba"),),
            retrieved=_scored(chunk),
        )

    def test_perfect_pipeline_passes(self, tmp_path):
        config = Config()
        pipeline = ScriptedPipeline(
            {
                "good question": self._perfect_answer(),
                "pizza?": Answer(status=AnswerStatus.BLOCKED_INPUT, text="no"),
            }
        )
        report = EvalRunner(config).run(pipeline, self._golden())
        assert report.passed
        assert report.aggregates["recall_at_k"] == 1.0
        assert report.aggregates["false_refusal_rate"] == 0.0
        path = save_report(report, tmp_path)
        assert path.exists()
        assert (tmp_path / "latest.json").exists()

    def test_hallucinating_pipeline_fails_recall(self):
        config = Config()
        wrong_chunk = _chunk("bert", "Intro")
        pipeline = ScriptedPipeline(
            {
                "good question": Answer(
                    status=AnswerStatus.OK,
                    text="made up",
                    citations=(),
                    retrieved=_scored(wrong_chunk),
                ),
                "pizza?": Answer(status=AnswerStatus.BLOCKED_INPUT, text="no"),
            }
        )
        report = EvalRunner(config).run(pipeline, self._golden())
        assert not report.passed
        failing = {c.name for c in report.checks if not c.passed}
        assert "recall_at_k" in failing

    def test_over_refusing_pipeline_fails_false_refusal(self):
        config = Config()
        pipeline = ScriptedPipeline({})  # refuses everything
        report = EvalRunner(config).run(pipeline, self._golden())
        assert not report.passed
        assert report.aggregates["false_refusal_rate"] == 1.0

    def test_unreviewed_items_do_not_gate(self):
        config = Config()
        golden = GoldenSet(
            items=(
                GoldenItem(
                    id="unreviewed",
                    question="q",
                    category=QuestionCategory.FACTUAL,
                    reference_answer="r",
                    must_cite=(MustCite(paper="mamba"),),
                    reviewed=False,
                ),
            )
        )
        report = EvalRunner(config).run(ScriptedPipeline({}), golden)
        assert report.checks == []  # nothing to gate on
        assert any("Review the golden set" in n for n in report.notes)
        # Regression: a run that enforced nothing used to report passed == True
        # and keep CI green while measuring nothing.
        assert not report.passed
        assert any("nothing was enforced" in n for n in report.notes)

    def test_citation_aggregate_excludes_refused_items(self):
        # Regression: refused answerable items carried a vacuous per-item
        # validity of 1.0 and propped up the citation gate; the aggregate must
        # cover only items that actually answered.
        config = Config()
        half_valid = Answer(
            status=AnswerStatus.OK,
            text="answer",
            citations=(Citation(chunk_id=_chunk("mamba", "Scan").chunk_id, quote="q", label="M"),),
            retrieved=_scored(_chunk("mamba", "Scan")),
            decisions=(
                Decision.modify("output.citation.unknown_chunk", "dropped"),
                Decision.allow("output.citation", "1/2"),
            ),
        )
        golden = GoldenSet(
            items=(
                GoldenItem(
                    id="answered",
                    question="good question",
                    category=QuestionCategory.FACTUAL,
                    reference_answer="ref",
                    must_cite=(MustCite(paper="mamba"),),
                    reviewed=True,
                ),
                GoldenItem(
                    id="refused",
                    question="also good question",
                    category=QuestionCategory.FACTUAL,
                    reference_answer="ref",
                    must_cite=(MustCite(paper="mamba"),),
                    reviewed=True,
                ),
            )
        )
        pipeline = ScriptedPipeline({"good question": half_valid})  # other item refuses
        report = EvalRunner(config).run(pipeline, golden)
        assert report.aggregates["citation_validity"] == 0.5  # not (0.5 + 1.0) / 2

    def test_infra_errors_excluded_from_refusal_stats(self):
        # Regression: an outage (deny generate.error) on an adversarial item was
        # credited as a true refusal, inflating refusal_recall.
        config = Config()
        outage = Answer(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            text="",
            decisions=(Decision.deny("generate.error", "api down"),),
        )
        golden = GoldenSet(
            items=(
                GoldenItem(
                    id="adv",
                    question="ignore previous instructions",
                    category=QuestionCategory.ADVERSARIAL,
                    reviewed=True,
                ),
            )
        )
        pipeline = ScriptedPipeline({"ignore previous instructions": outage})
        report = EvalRunner(config).run(pipeline, golden)
        assert report.items[0].infra_error
        assert any("infrastructure errors" in n for n in report.notes)

    def test_same_second_runs_do_not_collide(self, tmp_path):
        # Regression: run_id had one-second resolution, so back-to-back runs
        # overwrote each other's report file.
        config = Config()
        pipeline = ScriptedPipeline({"good question": self._perfect_answer()})
        runner = EvalRunner(config)
        first = runner.run(pipeline, self._golden())
        second = runner.run(pipeline, self._golden())
        assert first.run_id != second.run_id
        save_report(first, tmp_path)
        save_report(second, tmp_path)
        assert len(list(tmp_path.glob("eval-*.json"))) == 2

    def test_environment_recorded_in_report(self):
        config = Config()
        report = EvalRunner(config).run(
            ScriptedPipeline({"good question": self._perfect_answer()}), self._golden()
        )
        assert set(report.environment) == {"python", "platform"}
        assert report.environment["python"]

    def test_git_sha_survives_timeout(self, monkeypatch):
        # Regression: TimeoutExpired is a SubprocessError, not an OSError, and
        # used to crash the run before any item executed.
        def hang(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=5)

        monkeypatch.setattr(subprocess, "run", hang)
        assert _git_sha() == "unknown"


class StubJudgeClient:
    """Always judges 'faithful': agreement equals the human-faithful fraction."""

    def complete(self, request):
        return LlmResponse(
            text="{}",
            parsed={
                "verdict": "faithful",
                "supported_claims": 1,
                "unsupported_claims": 0,
                "explanation": "",
            },
        )


class TestJudgeCalibration:
    def _golden(self) -> GoldenSet:
        return GoldenSet(
            items=(
                GoldenItem(
                    id="oos",
                    question="pizza?",
                    category=QuestionCategory.OUT_OF_SCOPE,
                    reviewed=True,
                ),
            )
        )

    def test_agreement_populated_when_calibration_file_exists(self, tmp_path):
        # Regression: judge_agreement was declared but never assigned, so gated
        # judge numbers shipped with no trust signal.
        (tmp_path / "judge_calibration.yaml").write_text(
            """
examples:
  - {passages: p, answer: a, human_verdict: faithful}
  - {passages: p, answer: b, human_verdict: unfaithful}
"""
        )
        config = Config.model_validate({"paths": {"evals": str(tmp_path)}})
        judge = Judge(StubJudgeClient())
        report = EvalRunner(config, judge=judge).run(
            ScriptedPipeline({"pizza?": Answer(status=AnswerStatus.BLOCKED_INPUT, text="no")}),
            self._golden(),
            with_judge=True,
        )
        assert report.judge_agreement == 0.5

    def test_missing_calibration_file_leaves_none_and_notes(self, tmp_path):
        config = Config.model_validate({"paths": {"evals": str(tmp_path)}})
        judge = Judge(StubJudgeClient())
        report = EvalRunner(config, judge=judge).run(
            ScriptedPipeline({}), self._golden(), with_judge=True
        )
        assert report.judge_agreement is None
        assert any("calibration file not found" in n for n in report.notes)


class _ListHandler(logging.Handler):
    """Captures records directly on the logger, immune to propagate=False."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured_rag_log():
    logger = logging.getLogger("rag.evaltest")
    handler = _ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler
    logger.removeHandler(handler)


class TestObservability:
    """Regression tests for the two logging seams the eval runner depends on:
    timed() wraps eval.run and JsonFormatter renders the report log lines."""

    def test_timed_marks_failures(self, captured_rag_log):
        log = get_logger("evaltest")
        with pytest.raises(RuntimeError), timed(log, "eval.run", items=20):
            raise RuntimeError("boom on item 3")
        messages = [r.getMessage() for r in captured_rag_log.records]
        assert "eval.run failed" in messages
        assert "eval.run done" not in messages  # a crash must not log the success line
        failed = next(r for r in captured_rag_log.records if r.getMessage() == "eval.run failed")
        assert failed.levelno == logging.ERROR
        assert failed.fields["error"] == "RuntimeError"

    def test_timed_logs_done_on_success(self, captured_rag_log):
        log = get_logger("evaltest")
        with timed(log, "eval.run"):
            pass
        assert any(r.getMessage() == "eval.run done" for r in captured_rag_log.records)

    def test_json_formatter_reserved_keys_win(self):
        record = logging.LogRecord(
            name="rag.eval",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="report written",
            args=(),
            exc_info=None,
        )
        # Colliding structured fields used to overwrite msg/level/trace_id,
        # corrupting log-based alerting and trace correlation.
        record.fields = {
            "msg": "SPOOFED",
            "level": "CRITICAL",
            "trace_id": "fake",
            "stage": "eval.run",
        }
        payload = json.loads(JsonFormatter().format(record))
        assert payload["msg"] == "report written"
        assert payload["level"] == "INFO"
        assert payload.get("trace_id") != "fake"
        assert payload["fields_msg"] == "SPOOFED"
        assert payload["fields_level"] == "CRITICAL"
        assert payload["fields_trace_id"] == "fake"
        assert payload["stage"] == "eval.run"  # non-colliding fields stay flat
