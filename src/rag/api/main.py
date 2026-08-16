"""The chatbot HTTP endpoint.

A thin adapter over the same `Pipeline` the CLI drives; no retrieval or policy
logic lives here. Two rules shape the surface. First, every `Answer` maps to
HTTP 200: a refusal is the system working as designed, and the `status` field,
not the HTTP status, carries the outcome. HTTP errors are reserved for genuine
faults: a missing index (503 with the command that fixes it) and unexpected
crashes (500 with a trace id, never a stack trace). Second, the heavy state
(index bundle, pipeline) is built lazily on the first request that needs it and
cached on `app.state`, so constructing the app never touches the filesystem and
tests can build one without an index on disk.
"""

from __future__ import annotations

import os
import threading
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from rag.app import IndexBundle, build_pipeline, load_index
from rag.config import Config
from rag.domain import Answer
from rag.errors import IndexError_
from rag.observability import get_logger, new_trace_id
from rag.pipeline import Pipeline

log = get_logger("api")

CONFIG_ENV_VAR = "RAG_CONFIG"

_INDEX_HINT = "run `rag ingest && rag index` to build the index, then restart the server"


# --------------------------------------------------------------------------- #
# Wire models
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    label: str
    quote: str
    chunk_id: str


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    llm_calls: int


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: list[CitationOut]
    sources: list[str]
    trace_id: str
    usage: UsageOut

    @classmethod
    def from_answer(cls, answer: Answer) -> AskResponse:
        return cls(
            status=answer.status.value,
            answer=answer.text,
            citations=[
                CitationOut(label=c.label, quote=c.quote, chunk_id=c.chunk_id)
                for c in answer.citations
            ],
            sources=list(answer.sources),
            trace_id=answer.trace_id,
            usage=UsageOut(
                input_tokens=answer.usage.input_tokens,
                output_tokens=answer.usage.output_tokens,
                llm_calls=answer.usage.llm_calls,
            ),
        )


class HealthResponse(BaseModel):
    status: str
    chunks: int
    config_hash: str


class CorpusDoc(BaseModel):
    doc_id: str
    title: str
    chunks: int


# --------------------------------------------------------------------------- #
# Lazy state
# --------------------------------------------------------------------------- #


class _LazyState:
    """Heavy serving state, built on the first request that needs it.

    The bundle and pipeline start as the factory overrides (None in production)
    and fill in lazily, so building the app never opens the index. One reentrant
    lock covers both builds: `get_pipeline` builds the bundle under the same
    lock, and two concurrent first requests must not each load the index or race
    a flush of the embed cache.
    """

    def __init__(
        self, config: Config, bundle: IndexBundle | None, pipeline: Pipeline | None
    ) -> None:
        self.config = config
        self._bundle = bundle
        self._pipeline = pipeline
        self._lock = threading.RLock()

    def get_bundle(self) -> IndexBundle:
        with self._lock:
            if self._bundle is None:
                self._bundle = load_index(self.config)
            return self._bundle

    def get_pipeline(self) -> Pipeline:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = build_pipeline(self.config, self.get_bundle())
            return self._pipeline


def _state(request: Request) -> _LazyState:
    return cast(_LazyState, request.app.state.rag)


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_app(
    config: Config | None = None,
    pipeline: Pipeline | None = None,
    bundle: IndexBundle | None = None,
) -> FastAPI:
    """Build the app. `pipeline` and `bundle` exist so tests can inject fakes.

    With `config` unset, the path in the RAG_CONFIG environment variable is
    loaded when present, else defaults apply.
    """
    if config is None:
        config = Config.load(os.environ.get(CONFIG_ENV_VAR) or None)

    app = FastAPI(title="research-rag")
    app.state.rag = _LazyState(config=config, bundle=bundle, pipeline=pipeline)

    @app.exception_handler(IndexError_)
    async def _missing_index(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc), "hint": _INDEX_HINT})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Minted here, not taken from the pipeline: by the time an exception
        # escapes, any request trace has already been unbound. The body carries
        # only the id; the detail stays in the server log.
        trace_id = new_trace_id()
        log.error(
            "unhandled error",
            fields={"trace_id": trace_id, "error": type(exc).__name__, "detail": str(exc)},
        )
        return JSONResponse(
            status_code=500, content={"detail": "internal error", "trace_id": trace_id}
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def chat_page() -> HTMLResponse:
        return HTMLResponse(_CHAT_PAGE)

    @app.get("/health")
    def health(request: Request) -> HealthResponse:
        state = _state(request)
        return HealthResponse(
            status="ok", chunks=len(state.get_bundle().store), config_hash=state.config.hash()
        )

    @app.get("/corpus")
    def corpus(request: Request) -> list[CorpusDoc]:
        store = _state(request).get_bundle().store
        return [
            CorpusDoc(doc_id=doc_id, title=doc_chunks[0].doc_title, chunks=len(doc_chunks))
            for doc_id, doc_chunks in store.by_doc().items()
        ]

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        answer = _state(request).get_pipeline().ask(body.question)
        return AskResponse.from_answer(answer)

    return app


# --------------------------------------------------------------------------- #
# Chat page
# --------------------------------------------------------------------------- #

# A convenience view for poking at the pipeline, not a product. Self-contained
# on purpose: no external assets, so it works on an air-gapped box and never
# mixes third-party script into a page that renders model output. Everything the
# server returns lands in the DOM via textContent, never innerHTML.
_CHAT_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>research-rag</title>
<style>
  body {
    margin: 0 auto; max-width: 44rem; padding: 2.5rem 1.25rem;
    background: #10141a; color: #e2e8f0;
    font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  h1 { font-size: 1.25rem; margin: 0 0 0.25rem; }
  p.hint { color: #8b98a9; margin: 0 0 1.5rem; font-size: 0.9rem; }
  form { display: flex; gap: 0.5rem; }
  input {
    flex: 1; padding: 0.65rem 0.9rem; border-radius: 0.5rem;
    border: 1px solid #2b3442; background: #1a212b; color: inherit; font: inherit;
  }
  input:focus { outline: 1px solid #4c7dd0; }
  button {
    padding: 0.65rem 1.1rem; border: 0; border-radius: 0.5rem;
    background: #2f6fdb; color: #fff; font: inherit; cursor: pointer;
  }
  button:disabled { opacity: 0.45; cursor: wait; }
  #badge {
    display: none; margin-top: 1.25rem; padding: 0.15rem 0.7rem;
    border-radius: 999px; background: #3a2b17; color: #e8a13c; font-size: 0.8rem;
  }
  #answer { margin-top: 1rem; white-space: pre-wrap; }
  #sources { color: #8b98a9; font-size: 0.9rem; }
  #sources li { margin: 0.25rem 0; }
</style>
</head>
<body>
<h1>research-rag</h1>
<p class="hint">Ask a question about the indexed papers. Answers cite their sources.</p>
<form id="form">
  <input id="question" maxlength="2000" placeholder="How does selective head gating work?"
         autocomplete="off" required>
  <button id="send" type="submit">Ask</button>
</form>
<span id="badge"></span>
<div id="answer"></div>
<ol id="sources"></ol>
<script>
  const form = document.getElementById('form');
  const question = document.getElementById('question');
  const send = document.getElementById('send');
  const badge = document.getElementById('badge');
  const answer = document.getElementById('answer');
  const sources = document.getElementById('sources');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    send.disabled = true;
    badge.style.display = 'none';
    sources.replaceChildren();
    answer.textContent = 'Thinking...';
    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: question.value}),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data);
        answer.textContent = 'Error ' + res.status + ': ' + detail;
        return;
      }
      answer.textContent = data.answer;
      if (data.status !== 'ok') {
        badge.textContent = data.status;
        badge.style.display = 'inline-block';
      }
      for (const label of data.sources) {
        const item = document.createElement('li');
        item.textContent = label;
        sources.append(item);
      }
    } catch (err) {
      answer.textContent = 'Request failed: ' + err;
    } finally {
      send.disabled = false;
    }
  });
</script>
</body>
</html>
"""

# The instance uvicorn serves: `uvicorn rag.api.main:app`.
app = create_app()
