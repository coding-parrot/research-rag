"""Eval harness tests: the harness is itself tested against a scripted pipeline
before it is ever pointed at the real one."""

from __future__ import annotations

import pytest

from rag.config import Config
from rag.domain import Answer, AnswerStatus, Chunk, Citation, Scored, make_chunk_id
from rag.errors import ManifestError
from rag.eval.datasets import (
    GoldenItem,
    GoldenSet,
    HeaderLabelItem,
    MustCite,
    QuestionCategory,
    load_golden,
)
from rag.eval.metrics import (
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
from rag.eval.runner import EvalRunner, save_report


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

    def test_mrr_position(self):
        retrieved = _scored(_chunk("bert", "Intro"), _chunk("mamba", "Scan"))
        assert mrr(retrieved, [MustCite(paper="mamba")]) == 0.5
        assert mrr(retrieved, [MustCite(paper="absent")]) == 0.0

    def test_ndcg_rewards_early_hits(self):
        hit_first = _scored(_chunk("mamba", "Scan"), _chunk("bert", "Intro"))
        hit_second = _scored(_chunk("bert", "Intro"), _chunk("mamba", "Scan"))
        target = [MustCite(paper="mamba")]
        assert ndcg_at_k(hit_first, target, 2) > ndcg_at_k(hit_second, target, 2)

    def test_context_precision(self):
        retrieved = _scored(_chunk("mamba", "Scan"), _chunk("bert", "Intro"))
        assert context_precision(retrieved, [MustCite(paper="mamba")]) == 0.5


class TestAnswerMetrics:
    def test_citation_validity_counts_drops(self):
        from rag.domain import Decision

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
