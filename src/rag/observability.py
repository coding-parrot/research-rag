"""Structured logging and trace ids.

Policy: we log identifiers, scores, decisions and counts. We never log full chunk
text or full answers, because the corpus and the queries are the two things most
likely to be sensitive in a deployed system. `preview()` is the only sanctioned way
to put document text into a log line.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

PREVIEW_CHARS = 80


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def current_trace_id() -> str:
    return _trace_id.get()


@contextmanager
def trace(trace_id: str | None = None) -> Iterator[str]:
    """Bind a trace id for the duration of a request."""
    tid = trace_id or new_trace_id()
    token = _trace_id.set(tid)
    try:
        yield tid
    finally:
        _trace_id.reset(token)


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Truncated, single-line rendering of text that is safe to log."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the trace id attached automatically."""

    # Core keys the formatter itself owns. Caller-supplied structured fields must
    # never overwrite them: a field named "msg" or "level" would rewrite the
    # line's message or severity and corrupt log-based alerting. Reserved keys
    # win; the colliding field keeps its data under a "fields_" prefix. The set
    # is fixed (not "keys currently in payload") because trace_id is absent when
    # no trace is bound, and a spoofed trace_id must not slip through that gap.
    _RESERVED = frozenset({"ts", "level", "logger", "msg", "trace_id", "exc"})

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if tid := _trace_id.get():
            payload["trace_id"] = tid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Anything passed via logger.info("...", extra={"fields": {...}})
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                payload[f"fields_{key}" if key in self._RESERVED else key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO, *, json_output: bool = False) -> None:
    """Idempotent root logger setup."""
    root = logging.getLogger("rag")
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(levelname)-7s %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.LoggerAdapter[logging.Logger]:
    """Logger that accepts structured fields: log.info('msg', fields={'k': v})."""
    return _FieldAdapter(logging.getLogger(f"rag.{name}"), {})


class _FieldAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        fields = kwargs.pop("fields", None)
        if fields:
            kwargs.setdefault("extra", {})["fields"] = fields
        return msg, kwargs


@contextmanager
def timed(log: logging.LoggerAdapter[logging.Logger], stage: str, **fields: Any) -> Iterator[None]:
    """Log the wall-clock duration of a stage, marking failures as failures.

    A crashed stage must not emit the same "done" line as a completed one:
    anyone grepping logs for stage completion would count the crash as a finish.
    The exception is re-raised untouched; only the log line differs. BaseException
    is deliberate, so even KeyboardInterrupt cannot produce a false "done".
    """
    start = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        log.error(
            f"{stage} failed",
            fields={
                "stage": stage,
                "elapsed_ms": elapsed_ms,
                "error": type(exc).__name__,
                **fields,
            },
        )
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    log.info(f"{stage} done", fields={"stage": stage, "elapsed_ms": elapsed_ms, **fields})
