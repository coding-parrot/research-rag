"""The language-model seam.

One interface, three implementations. Everything that calls a model in this system
goes through it: answer generation, query rewriting, and the eval judge. That is what
makes the pipeline testable without tokens and what lets the judge run on a different
model from the generator without touching either call site.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rag.domain import Usage
from rag.errors import LlmError
from rag.observability import get_logger

log = get_logger("llm")


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """One model call."""

    prompt: str
    system: str = ""
    model: str = ""
    effort: str = "high"
    max_tokens: int = 4096
    # JSON Schema. When set, the response must validate against it and `parsed` is
    # populated. This is how citations stop being free text the model can invent.
    schema: Mapping[str, Any] | None = None
    # Mark the system prompt as cacheable. The system prompt is stable across every
    # request; the question and retrieved chunks are not, so they go after it.
    cache_system: bool = False


@dataclass(frozen=True, slots=True)
class LlmResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    stop_reason: str = "end_turn"
    parsed: Mapping[str, Any] | None = None

    @property
    def refused(self) -> bool:
        """The model's safety classifiers declined. Not an error, a content outcome."""
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


@runtime_checkable
class LlmClient(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, request: LlmRequest) -> LlmResponse: ...


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicClient:
    """Claude via the official SDK.

    Adaptive thinking is on, effort is configurable, and structured output is used
    whenever a schema is supplied. Streaming is used for large `max_tokens` because
    a non-streaming request at that size can outlive the HTTP timeout.
    """

    def __init__(self, default_model: str = "claude-opus-5", api_key: str | None = None) -> None:
        self._default_model = default_model
        self._api_key = api_key
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return f"anthropic:{self._default_model}"

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise LlmError("pip install -e '.[generate]' to use AnthropicClient") from exc
        # No api_key argument when unset: the SDK then resolves ANTHROPIC_API_KEY,
        # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile on its own.
        self._client = (
            anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        )
        return self._client

    def complete(self, request: LlmRequest) -> LlmResponse:
        client = self._load()
        model = request.model or self._default_model

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": request.effort},
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = (
                [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
                if request.cache_system
                else request.system
            )
        if request.schema is not None:
            kwargs["output_config"]["format"] = {
                "type": "json_schema",
                "schema": dict(request.schema),
            }

        try:
            if request.max_tokens > 16_000:
                with client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
            else:
                message = client.messages.create(**kwargs)
        except Exception as exc:
            raise LlmError(f"anthropic call failed: {exc}") from exc

        return _from_anthropic(message)


def _from_anthropic(message: Any) -> LlmResponse:
    stop_reason = str(getattr(message, "stop_reason", "end_turn") or "end_turn")

    # Check the stop reason before touching content. On a refusal `content` is empty
    # or partial, and indexing into it is the classic way this breaks in production.
    text = "" if stop_reason == "refusal" else _first_text(message.content)

    raw_usage = getattr(message, "usage", None)
    usage = Usage(
        input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0),
        llm_calls=1,
    )
    return LlmResponse(
        text=text,
        usage=usage,
        model=str(getattr(message, "model", "")),
        stop_reason=stop_reason,
        parsed=_maybe_json(text),
    )


def _first_text(content: Sequence[Any]) -> str:
    return next((block.text for block in content if getattr(block, "type", "") == "text"), "")


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


class OllamaClient:
    """Local model over the Ollama HTTP API.

    Kept as a first-class option, not a fallback: it makes the whole pipeline
    runnable with no API key and no spend, which matters for teaching and for
    iterating on prompts before spending anything.
    """

    def __init__(
        self, model: str = "gemma2:2b", host: str = "http://localhost:11434", timeout: float = 120.0
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def complete(self, request: LlmRequest) -> LlmResponse:
        import httpx

        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "prompt": request.prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": request.max_tokens},
        }
        if request.system:
            payload["system"] = request.system
        if request.schema is not None:
            payload["format"] = dict(request.schema)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._host}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            raise LlmError(f"ollama call failed: {exc}. Is `ollama serve` running?") from exc

        text = str(body.get("response", ""))
        return LlmResponse(
            text=text,
            usage=Usage(
                input_tokens=int(body.get("prompt_eval_count", 0) or 0),
                output_tokens=int(body.get("eval_count", 0) or 0),
                llm_calls=1,
            ),
            model=str(body.get("model", self._model)),
            stop_reason="end_turn" if body.get("done") else "max_tokens",
            parsed=_maybe_json(text),
        )


# --------------------------------------------------------------------------- #
# Fake
# --------------------------------------------------------------------------- #


class FakeLlmClient:
    """Scripted client for tests.

    Responses are matched by substring against the prompt, so a test can say "when
    the prompt mentions citations, return this JSON" without caring about the exact
    prompt text. Records every request so tests can assert on what was sent.
    """

    def __init__(
        self,
        responses: Sequence[str | LlmResponse] | None = None,
        *,
        by_substring: Mapping[str, str | LlmResponse] | None = None,
        default: str = "",
    ) -> None:
        self._queue = list(responses or [])
        self._by_substring = dict(by_substring or {})
        self._default = default
        self.requests: list[LlmRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def last_prompt(self) -> str:
        return self.requests[-1].prompt if self.requests else ""

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)

        for needle, response in self._by_substring.items():
            if needle in request.prompt or needle in request.system:
                return _as_response(response)
        if self._queue:
            return _as_response(self._queue.pop(0))
        return _as_response(self._default)


def _as_response(value: str | LlmResponse) -> LlmResponse:
    if isinstance(value, LlmResponse):
        return value
    return LlmResponse(
        text=value, usage=Usage(llm_calls=1), model="fake", parsed=_maybe_json(value)
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _maybe_json(text: str) -> Mapping[str, Any] | None:
    """Parse a JSON object out of a response, tolerating markdown fences.

    Structured outputs make this unnecessary on Claude, but Ollama-backed small
    models wrap JSON in fences no matter what the prompt says, and the judge and
    the answerer both need to keep working there.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if match := _FENCE.search(stripped):
        stripped = match.group(1).strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        stripped = stripped[start : end + 1]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_client(
    provider: str, *, model: str, ollama_model: str, ollama_host: str, api_key: str | None = None
) -> LlmClient:
    if provider == "fake":
        return FakeLlmClient()
    if provider == "ollama":
        return OllamaClient(model=ollama_model, host=ollama_host)
    return AnthropicClient(default_model=model, api_key=api_key)
