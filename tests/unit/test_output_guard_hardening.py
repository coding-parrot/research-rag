"""Regression tests for the output-guard and answerer hardening pass.

Each test here is derived from a verified failure scenario in the review findings:
text shipped without redaction on refusal paths, citation quotes leaking what the
prose redacted, year runs mangled by the card-number pattern, substantive answers
misfiled as refusals, partial citation drops shipping unverified claims, typography
mismatches burning retries, truncated bodies parsed as answers, unbounded
model-controlled evidence, repr-shaped prose, and the fake provider reaching
production config.
"""

import json

import pytest

from rag.config import GenerateConfig, GuardrailConfig
from rag.domain import AnswerStatus, Chunk, Scored, make_chunk_id
from rag.errors import ConfigError, LlmError
from rag.generate.answerer import Answerer, _is_model_refusal, _parse
from rag.generate.client import (
    FakeLlmClient,
    LlmRequest,
    LlmResponse,
    OllamaClient,
    build_client,
)
from rag.guardrails.output_guard import OutputGuard, quote_appears_in


def _chunk(text: str, doc_id: str = "mamba") -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(doc_id, 0, len(text), 0),
        doc_id=doc_id,
        doc_title=doc_id.title(),
        text=text,
        char_start=0,
        char_end=len(text),
        section_title="Method",
        section_number="3",
        page_start=2,
        page_end=2,
    )


def _scored(text: str, doc_id: str = "mamba") -> Scored:
    return Scored(chunk=_chunk(text, doc_id), score=0.9, rank=1, retriever="test")


CHUNK = _chunk("The selective scan makes SSM parameters input-dependent functions of the token.")
RETRIEVED = (Scored(chunk=CHUNK, score=0.9, rank=1, retriever="test"),)
VALID_QUOTE = "makes SSM parameters input-dependent"


def _answer_json(
    quote: str = VALID_QUOTE,
    chunk_id: str = CHUNK.chunk_id,
    answer: str = "Parameters become input-dependent.",
) -> str:
    return json.dumps({"answer": answer, "citations": [{"chunk_id": chunk_id, "quote": quote}]})


def _answerer(client: FakeLlmClient, regenerations: int = 1) -> Answerer:
    return Answerer(
        client,
        OutputGuard(GuardrailConfig()),
        GenerateConfig(provider="fake", max_regenerations=regenerations),
    )


@pytest.fixture()
def config() -> GuardrailConfig:
    return GuardrailConfig()


class TestRefusalPathRedaction:
    """Model-authored refusal prose must pass the same redaction as the OK path."""

    def test_model_refusal_text_is_redacted(self):
        client = FakeLlmClient(
            [
                json.dumps(
                    {
                        "answer": (
                            "The papers do not cover deployment contacts; the only "
                            "email shown is jane.doe@university.edu."
                        ),
                        "citations": [],
                    }
                )
            ]
        )
        answer = _answerer(client).answer("Who do I contact?", RETRIEVED)
        assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
        assert "jane.doe@university.edu" not in answer.text
        assert "[redacted]" in answer.text
        assert any(d.rule_id == "output.redaction" for d in answer.decisions)


class TestCitationQuoteRedaction:
    """Quotes ship in Answer.citations, so they get the same scrubbing as prose,
    applied after verification so validation still sees the verbatim chunk text."""

    def test_quote_validates_against_raw_text_then_ships_redacted(self, config):
        retrieved = (_scored("Correspondence to jane.doe@university.edu about the dataset."),)
        verdict = OutputGuard(config).check(
            text="Contact is jane.doe@university.edu per the paper.",
            raw_citations=[
                {
                    "chunk_id": retrieved[0].chunk.chunk_id,
                    "quote": "Correspondence to jane.doe@university.edu",
                }
            ],
            retrieved=retrieved,
        )
        # Validation ran on the verbatim text (the citation survived), and both
        # the prose and the stored quote were redacted afterwards.
        assert verdict.allowed
        assert len(verdict.citations) == 1
        assert "jane.doe@university.edu" not in verdict.text
        assert "jane.doe@university.edu" not in verdict.citations[0].quote
        assert "[redacted]" in verdict.citations[0].quote
        redaction = next(d for d in verdict.decisions if d.rule_id == "output.redaction")
        assert "quote:" in redaction.evidence


class TestCardRedactionLuhnGated:
    """Only Luhn-valid digit runs are card numbers; year lists must survive."""

    def test_year_run_is_not_redacted(self, config):
        retrieved = RETRIEVED
        verdict = OutputGuard(config).check(
            text="Accuracy improved across 2019 2020 2021 2022 per the benchmarks.",
            raw_citations=[{"chunk_id": CHUNK.chunk_id, "quote": VALID_QUOTE}],
            retrieved=retrieved,
        )
        assert verdict.allowed
        assert verdict.text == "Accuracy improved across 2019 2020 2021 2022 per the benchmarks."
        assert not any(d.rule_id == "output.redaction" for d in verdict.decisions)

    def test_luhn_valid_card_number_is_redacted(self, config):
        verdict = OutputGuard(config).check(
            text="The card 4111 1111 1111 1111 appears in the appendix.",
            raw_citations=[{"chunk_id": CHUNK.chunk_id, "quote": VALID_QUOTE}],
            retrieved=RETRIEVED,
        )
        assert "4111" not in verdict.text
        assert "[redacted]" in verdict.text


class TestRefusalHeuristic:
    """A refusal leads with the marker and stays short; substantive uncited
    answers must fall through to citation validation and its retry."""

    def test_genuine_refusal_detected(self):
        assert _is_model_refusal("The indexed papers do not cover this topic.", [])

    def test_citations_never_a_refusal(self):
        assert not _is_model_refusal(
            "The papers do not cover it.", [{"chunk_id": "x", "quote": "some quote"}]
        )

    def test_long_answer_with_incidental_marker_is_not_a_refusal(self):
        text = (
            "The selective scan makes SSM parameters input-dependent functions of "
            "the token, and the paper reports strong benchmark results across "
            "language and DNA tasks, though it offers no information-theoretic "
            "analysis of the mechanism."
        )
        assert len(text) > 200  # precondition: substantive length
        assert not _is_model_refusal(text, [])

    def test_late_marker_in_short_answer_is_not_a_refusal(self):
        lead = (
            "The selective scan makes parameters input-dependent and the paper "
            "reports strong results across language, genomics, and audio "
            "benchmarks at multiple model scales. "
        )
        text = lead + "Speed is not addressed."
        # Preconditions of the scenario: marker past the window, answer short.
        assert len(lead) >= 160
        assert len(text) <= 200
        assert not _is_model_refusal(text, [])

    def test_substantive_uncited_answer_triggers_citation_retry(self):
        uncited = json.dumps(
            {
                "answer": (
                    "The selective scan makes SSM parameters input-dependent "
                    "functions of the token, and the paper reports strong benchmark "
                    "results across language and DNA tasks, though it offers no "
                    "information-theoretic analysis of the mechanism."
                ),
                "citations": [],
            }
        )
        client = FakeLlmClient([uncited, _answer_json()])
        answer = _answerer(client).answer("How does the scan work?", RETRIEVED)
        assert answer.status is AnswerStatus.OK
        assert client.call_count == 2  # the retry fired instead of a misfiled refusal


class TestPartialCitationDrops:
    """Any dropped citation is worth one regeneration even when others survive;
    the final attempt ships the survivors with the drops on record."""

    def _retrieved(self):
        return (
            _scored("The selective scan makes SSM parameters input-dependent functions."),
            _scored("LoRA freezes pretrained weights and trains low-rank matrices.", doc_id="lora"),
        )

    def test_verdict_with_drops_is_retryable_despite_survivors(self, config):
        retrieved = self._retrieved()
        verdict = OutputGuard(config).check(
            text="An answer citing two things.",
            raw_citations=[
                {"chunk_id": retrieved[1].chunk.chunk_id, "quote": "freezes pretrained weights"},
                {"chunk_id": "fabricated", "quote": "made up quote entirely"},
            ],
            retrieved=retrieved,
        )
        assert verdict.allowed
        assert len(verdict.citations) == 1
        assert len(verdict.dropped_citations) == 1
        assert verdict.should_retry

    def test_partial_drop_regenerates_then_ships_clean_answer(self):
        partial = json.dumps(
            {
                "answer": "Two claims, one fabricated citation.",
                "citations": [
                    {"chunk_id": CHUNK.chunk_id, "quote": VALID_QUOTE},
                    {"chunk_id": "fabricated", "quote": "made up quote entirely"},
                ],
            }
        )
        client = FakeLlmClient([partial, _answer_json()])
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.OK
        assert client.call_count == 2
        # The regeneration prompt names what was dropped.
        assert "dropped" in client.requests[1].prompt

    def test_final_attempt_ships_survivors(self):
        partial = json.dumps(
            {
                "answer": "Two claims, one fabricated citation.",
                "citations": [
                    {"chunk_id": CHUNK.chunk_id, "quote": VALID_QUOTE},
                    {"chunk_id": "fabricated", "quote": "made up quote entirely"},
                ],
            }
        )
        client = FakeLlmClient([partial, partial])
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.OK
        assert len(answer.citations) == 1
        assert client.call_count == 2

    def test_failed_retry_falls_back_to_earlier_partial(self):
        partial = json.dumps(
            {
                "answer": "Two claims, one fabricated citation.",
                "citations": [
                    {"chunk_id": CHUNK.chunk_id, "quote": VALID_QUOTE},
                    {"chunk_id": "fabricated", "quote": "made up quote entirely"},
                ],
            }
        )
        worse = _answer_json(quote="this quote was never in the chunk")
        client = FakeLlmClient([partial, worse])
        answer = _answerer(client).answer("q", RETRIEVED)
        # The retry did worse than the partial it was meant to improve on; the
        # partial ships rather than a refusal.
        assert answer.status is AnswerStatus.OK
        assert len(answer.citations) == 1


class TestQuoteTypographyFolding:
    """Curly quotes, dash variants, and ellipses fold to ASCII on both sides."""

    def test_straight_quotes_match_curly_source(self):
        assert quote_appears_in('the "selective" scan', "the “selective” scan mechanism")

    def test_curly_quotes_match_straight_source(self):
        assert quote_appears_in("the “selective” scan", 'the "selective" scan mechanism')

    def test_hyphen_matches_en_dash(self):
        assert quote_appears_in("input-dependent scan", "the input–dependent scan here")  # noqa: RUF001

    def test_ascii_ellipsis_matches_unicode(self):
        assert quote_appears_in("results... improved", "the results… improved a lot")

    def test_letters_must_still_match_exactly(self):
        assert not quote_appears_in("the scan is selective", "the selective scan mechanism")


class TestTruncatedResponses:
    """A body cut at max_tokens is a failed attempt, not a parseable answer."""

    def test_truncated_response_refuses_with_typed_decision(self):
        client = FakeLlmClient(
            [LlmResponse(text='{"answer": "The selective scan', stop_reason="max_tokens")]
        )
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
        assert any(d.rule_id == "generate.truncated" for d in answer.decisions)
        # No citation retry is burned on a doomed regeneration at the same limit.
        assert client.call_count == 1


class TestOllamaBodyHandling:
    """A 200 with a bad body must become a typed LlmError, not a raw exception."""

    def test_html_body_raises_llm_error(self, monkeypatch):
        import httpx

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>bad gateway</html>")
        )
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport))
        with pytest.raises(LlmError, match="non-JSON body"):
            OllamaClient().complete(LlmRequest(prompt="q"))

    def test_non_object_json_body_raises_llm_error(self, monkeypatch):
        import httpx

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[1, 2, 3]))
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport))
        with pytest.raises(LlmError, match="non-object"):
            OllamaClient().complete(LlmRequest(prompt="q"))


class TestUnknownChunkEvidenceBounded:
    """The chunk_id is model-controlled; evidence gets the preview() bound."""

    def test_huge_chunk_id_is_truncated_in_evidence(self, config):
        huge = "x" * 200 + "jane.doe@university.edu" + "y" * 5000
        verdict = OutputGuard(config).check(
            text="An answer.",
            raw_citations=[{"chunk_id": huge, "quote": "whatever text here"}],
            retrieved=RETRIEVED,
        )
        unknown = next(d for d in verdict.decisions if d.rule_id == "output.citation.unknown_chunk")
        assert len(unknown.evidence) <= 60
        assert "@" not in unknown.evidence

    def test_empty_chunk_id_keeps_sentinel(self, config):
        verdict = OutputGuard(config).check(
            text="An answer.",
            raw_citations=[{"chunk_id": "", "quote": "whatever text here"}],
            retrieved=RETRIEVED,
        )
        unknown = next(d for d in verdict.decisions if d.rule_id == "output.citation.unknown_chunk")
        assert unknown.evidence == "<empty>"


class TestParseRejectsNonStringFields:
    """A structurally wrong answer falls back to raw text, never str() reprs."""

    def test_dict_answer_falls_back_to_raw_text(self):
        raw = '{"answer": {"nested": "dict"}, "citations": []}'
        text, citations = _parse({"answer": {"nested": "dict"}, "citations": []}, raw)
        assert text == raw
        assert citations == []

    def test_null_answer_falls_back_to_raw_text(self):
        raw = '{"answer": null, "citations": []}'
        text, citations = _parse({"answer": None, "citations": []}, raw)
        assert text == raw
        assert citations == []

    def test_non_string_citation_fields_are_skipped(self):
        _, citations = _parse(
            {
                "answer": "fine",
                "citations": [
                    {"chunk_id": 5, "quote": "long enough quote"},
                    {"chunk_id": "id", "quote": None},
                    {"chunk_id": "kept", "quote": "kept quote"},
                ],
            },
            "raw",
        )
        assert citations == [{"chunk_id": "kept", "quote": "kept quote"}]

    def test_repr_never_ships_end_to_end(self):
        bad = json.dumps(
            {
                "answer": {"nested": "dict"},
                "citations": [{"chunk_id": CHUNK.chunk_id, "quote": VALID_QUOTE}],
            }
        )
        client = FakeLlmClient([bad, _answer_json()])
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.OK
        assert "'nested'" not in answer.text
        assert client.call_count == 2  # raw-text fallback took the retryable path


class TestBuildClientRejectsFake:
    """provider='fake' is test-only; a config that reaches build_client with it
    must fail loudly instead of silently answering nothing."""

    def test_fake_provider_raises_config_error(self):
        with pytest.raises(ConfigError, match="test-only"):
            build_client(
                "fake",
                model="claude-opus-5",
                ollama_model="gemma2:2b",
                ollama_host="http://localhost:11434",
            )
