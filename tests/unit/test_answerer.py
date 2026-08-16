import json

from rag.config import GenerateConfig, GuardrailConfig
from rag.domain import AnswerStatus, Chunk, Scored, make_chunk_id
from rag.generate.answerer import Answerer
from rag.generate.client import FakeLlmClient, LlmResponse
from rag.guardrails.output_guard import OutputGuard


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


CHUNK = _chunk("The selective scan makes SSM parameters input-dependent functions of the token.")
RETRIEVED = (Scored(chunk=CHUNK, score=0.9, rank=1, retriever="test"),)


def _answer_json(
    quote: str, chunk_id: str = CHUNK.chunk_id, answer: str = "Parameters become input-dependent."
) -> str:
    return json.dumps({"answer": answer, "citations": [{"chunk_id": chunk_id, "quote": quote}]})


def _answerer(client: FakeLlmClient, regenerations: int = 1) -> Answerer:
    return Answerer(
        client,
        OutputGuard(GuardrailConfig()),
        GenerateConfig(provider="fake", max_regenerations=regenerations),
    )


class TestHappyPath:
    def test_valid_answer(self):
        client = FakeLlmClient([_answer_json("makes SSM parameters input-dependent")])
        answer = _answerer(client).answer("How does the scan work?", RETRIEVED)
        assert answer.status is AnswerStatus.OK
        assert answer.citations[0].label.startswith("Mamba")
        assert answer.usage.llm_calls == 1

    def test_system_prompt_is_cached_and_stable(self):
        client = FakeLlmClient([_answer_json("makes SSM parameters input-dependent")])
        _answerer(client).answer("q", RETRIEVED)
        request = client.requests[0]
        assert request.cache_system
        assert "untrusted" in request.system  # the injection defence is in the system prompt
        assert request.schema is not None


class TestRegeneration:
    def test_bad_citation_retries_once_then_succeeds(self):
        client = FakeLlmClient(
            [
                _answer_json("this quote was never in the chunk"),
                _answer_json("makes SSM parameters input-dependent"),
            ]
        )
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.OK
        assert client.call_count == 2
        # The retry prompt tells the model what failed.
        assert "failed citation validation" in client.requests[1].prompt

    def test_exhausted_retries_refuse(self):
        client = FakeLlmClient(
            [
                _answer_json("fabricated quote number one here"),
                _answer_json("fabricated quote number two here"),
            ]
        )
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
        assert client.call_count == 2  # 1 + max_regenerations
        assert "evidence" in answer.text

    def test_zero_regenerations_config(self):
        client = FakeLlmClient([_answer_json("fabricated quote here entirely")])
        answer = _answerer(client, regenerations=0).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
        assert client.call_count == 1


class TestModelRefusals:
    def test_no_coverage_answer_is_insufficient_evidence_not_retry(self):
        client = FakeLlmClient(
            [json.dumps({"answer": "The indexed papers do not cover this topic.", "citations": []})]
        )
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
        assert client.call_count == 1  # a correct refusal must not trigger regeneration

    def test_safety_refusal_maps_to_blocked_output(self):
        client = FakeLlmClient([LlmResponse(text="", stop_reason="refusal")])
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.BLOCKED_OUTPUT

    def test_unparseable_response_becomes_refusal(self):
        client = FakeLlmClient(["not json at all", "still not json"])
        answer = _answerer(client).answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE


class TestErrorHandling:
    def test_llm_error_becomes_typed_refusal(self):
        class ExplodingClient:
            name = "exploding"

            def complete(self, request):
                from rag.errors import LlmError

                raise LlmError("connection reset")

        answerer = Answerer(
            ExplodingClient(),
            OutputGuard(GuardrailConfig()),
            GenerateConfig(provider="fake"),
        )
        answer = answerer.answer("q", RETRIEVED)
        assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
        assert any(d.rule_id == "generate.error" for d in answer.decisions)
