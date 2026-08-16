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
from rag.errors import ConfigError, LlmError
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
# OpenAI
# --------------------------------------------------------------------------- #

# Our effort ladder is Claude-shaped (five levels); OpenAI's reasoning_effort has
# four. xhigh and max collapse onto OpenAI's ceiling.
_OPENAI_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class OpenAiClient:
    """OpenAI models via the official SDK's chat-completions surface.

    Structured output uses the `json_schema` response format with `strict`
    enforcement, which mirrors what the Anthropic client gets from
    `output_config.format`: the answer either validates against the citation
    schema or the call fails loudly.

    `reasoning_effort` is sent for the configured effort; if the target model
    rejects the parameter (not every OpenAI model accepts it), the request is
    retried once without it rather than failing the answer.
    """

    def __init__(self, default_model: str = "gpt-5.6-sol", api_key: str | None = None) -> None:
        self._default_model = default_model
        self._api_key = api_key
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return f"openai:{self._default_model}"

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - install-dependent
            raise LlmError("the `openai` package is required for OpenAiClient") from exc
        # No api_key argument when unset: the SDK resolves OPENAI_API_KEY itself.
        self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        return self._client

    def complete(self, request: LlmRequest) -> LlmResponse:
        client = self._load()
        model = request.model or self._default_model

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
            "reasoning_effort": _OPENAI_EFFORT.get(request.effort, "high"),
        }
        if request.schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": dict(request.schema),
                    "strict": True,
                },
            }

        try:
            completion = self._create(client, kwargs)
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"openai call failed: {exc}") from exc

        return _from_openai(completion)

    def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        """One create call, retrying once without `reasoning_effort` if rejected."""
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc)
            if "reasoning_effort" in message and "reasoning_effort" in kwargs:
                log.warning(
                    "model rejected reasoning_effort; retrying without it",
                    fields={"model": kwargs.get("model", "")},
                )
                retry_kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
                return client.chat.completions.create(**retry_kwargs)
            raise


def _from_openai(completion: Any) -> LlmResponse:
    choice = completion.choices[0] if getattr(completion, "choices", None) else None
    if choice is None:
        raise LlmError("openai returned no choices")

    finish = str(getattr(choice, "finish_reason", "") or "stop")
    # Map OpenAI finish reasons onto our stop vocabulary so the answerer's
    # truncation and refusal handling works identically across providers.
    stop_reason = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}.get(
        finish, "end_turn"
    )

    message = getattr(choice, "message", None)
    refusal = getattr(message, "refusal", None) if message is not None else None
    if refusal:
        stop_reason = "refusal"
    text = "" if stop_reason == "refusal" else str(getattr(message, "content", "") or "")

    raw_usage = getattr(completion, "usage", None)
    usage = Usage(
        input_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
        llm_calls=1,
    )
    return LlmResponse(
        text=text,
        usage=usage,
        model=str(getattr(completion, "model", "")),
        stop_reason=stop_reason,
        parsed=_maybe_json(text),
    )


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
        except ValueError as exc:
            # A proxy in front of Ollama can answer 200 with an HTML error page.
            # That is a bad body from a live endpoint, not a dead server, hence a
            # message without the "is it running" hint.
            raise LlmError(f"ollama returned a non-JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise LlmError(f"ollama returned a non-object JSON body: {type(body).__name__}")

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
    provider: str,
    *,
    model: str,
    ollama_model: str,
    ollama_host: str,
    api_key: str | None = None,
    openai_api_key: str | None = None,
) -> LlmClient:
    if provider == "fake":
        # The fake is a test double; tests construct FakeLlmClient directly and
        # inject it. A config that reaches this seam with provider=fake (a copied
        # test fixture, say) would otherwise run a pipeline that silently answers
        # nothing, so it fails loudly instead.
        raise ConfigError(
            "generate.provider='fake' is test-only; tests construct FakeLlmClient "
            "directly. Set provider to 'openai', 'anthropic' or 'ollama'."
        )
    if provider == "openai":
        return OpenAiClient(default_model=model, api_key=openai_api_key)
    if provider == "ollama":
        return OllamaClient(model=ollama_model, host=ollama_host)
    return AnthropicClient(default_model=model, api_key=api_key)
