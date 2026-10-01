"""Tracing through Langfuse (v4, built on OpenTelemetry), masked before export.

Observability never breaks a request (engineering rules §5): without keys, or
with tracing switched off, every call here is a no-op, and a Langfuse failure is
logged and swallowed. Everything Langfuse records passes through
``mask_trace_data`` first, so image data, bytes, emails and phone numbers never
leave the process, while prices, calories, dates and IDs stay readable.

Langfuse uploads base64 images to its media store *before* its mask hook runs,
so ``init_telemetry`` switches media upload off; masking alone would not keep
user photos in-process.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, ConfigDict, Field

from food_concierge import __version__
from food_concierge.config import ModelRole, Settings
from food_concierge.errors import AppError

if TYPE_CHECKING:
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter

logger = logging.getLogger(__name__)

ObservationType = Literal["span", "generation", "agent", "tool", "chain", "retriever", "evaluator", "guardrail"]

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

# Bounds keep masking cheap on any payload and cut cycles: deeper levels, extra items and long text are cut.
MAX_DEPTH = 10
MAX_ITEMS = 100
MAX_TEXT_CHARS = 4000

_DATA_URI = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")
# Long unbroken base64 runs are image or file content even without a data: prefix.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A phone number: 10-15 digits, optionally led by +, with single spaces, dots, dashes or brackets between them.
# It may not start or end inside a word, ID, decimal or path, nor run into a time ("2026-10-01 12:30").
_PHONE = re.compile(r"(?<![\w.+/#-])\+?\(?\d(?:[ .()-]{0,2}\d){9,14}(?![\w:/]|[.-]\d)")
_DATE_PREFIX = re.compile(r"\d{4}-\d{2}-\d{2}")


def _mask_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    return candidate if _DATE_PREFIX.match(candidate) else "[phone]"


def mask_text(text: str) -> str:
    """Replace image data, emails and phone numbers in ``text``, then cut it to ``MAX_TEXT_CHARS``."""
    text = _DATA_URI.sub("[image]", text)
    text = _BASE64_RUN.sub("[binary]", text)
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub(_mask_phone, text)
    if len(text) > MAX_TEXT_CHARS:
        text = f"{text[:MAX_TEXT_CHARS]}…[+{len(text) - MAX_TEXT_CHARS} chars]"
    return text


def mask_value(value: Any, depth: int = 0) -> Any:
    """A JSON-ready copy of ``value`` with sensitive content masked; the input is never modified."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return f"[bytes: {len(value)}]"
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return mask_value(value.value, depth)
    if depth >= MAX_DEPTH:
        return "[truncated: too deep]"
    if isinstance(value, BaseModel):
        return mask_value(value.model_dump(mode="python"), depth)
    if isinstance(value, Mapping):
        items = list(value.items())
        masked = {mask_text(str(key)): mask_value(item, depth + 1) for key, item in items[:MAX_ITEMS]}
        if len(items) > MAX_ITEMS:
            masked["[truncated]"] = f"+{len(items) - MAX_ITEMS} items"
        return masked
    if isinstance(value, list | tuple | set | frozenset):
        elements = list(value)
        masked_list = [mask_value(item, depth + 1) for item in elements[:MAX_ITEMS]]
        if len(elements) > MAX_ITEMS:
            masked_list.append(f"[+{len(elements) - MAX_ITEMS} items]")
        return masked_list
    # Unknown objects are exported as their text, masked: never as an unmasked structure.
    return mask_text(str(value))


def mask_trace_data(*, data: Any, **_: Any) -> Any:
    """Langfuse ``mask`` hook. Fails closed: if masking itself fails, nothing of the payload is exported."""
    try:
        return mask_value(data)
    except Exception:
        logger.warning("trace masking failed; payload dropped", exc_info=True)
        return "[masking failed]"


# ---------------------------------------------------------------------------
# Trace metadata
# ---------------------------------------------------------------------------


class TraceMeta(BaseModel):
    """Metadata attached to every traced model call; one shape so dashboards can group by it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str | None = None
    role: ModelRole | None = None
    provider: str | None = None  # the provider that answered, or the last one tried
    model: str | None = None
    fallback_used: bool = False
    attempts: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    prompt_version: str | None = None
    degraded: bool = False
    error_code: str | None = None  # set when no provider answered


class ModelAttempt(BaseModel):
    provider: str
    model: str
    error_code: str | None = None


class ModelCallRecorder(BaseCallbackHandler):
    """LangChain callback recording each provider attempt of a model call.

    Pass it in ``config={"callbacks": [...]}``; every provider tried along the
    fallback chain starts its own run, so attempts and fallbacks are counted
    exactly. Build ``TraceMeta`` from it with ``meta()``.
    """

    # Recording must never fail the model call it observes.
    raise_error = False

    def __init__(self) -> None:
        self.attempts: list[ModelAttempt] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._runs: dict[UUID, ModelAttempt] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        attempt = ModelAttempt(
            provider=str(metadata.get("ls_provider", "unknown")),
            model=str(metadata.get("ls_model_name", "unknown")),
        )
        self.attempts.append(attempt)
        self._runs[run_id] = attempt

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._runs.pop(run_id, None)
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.input_tokens += int(usage.get("input_tokens", 0))
                self.output_tokens += int(usage.get("output_tokens", 0))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        attempt = self._runs.pop(run_id, None)
        if attempt is not None:
            attempt.error_code = _error_code(error)

    def meta(
        self,
        *,
        role: ModelRole | None = None,
        request_id: str | None = None,
        prompt_version: str | None = None,
        degraded: bool = False,
    ) -> TraceMeta:
        last = self.attempts[-1] if self.attempts else None
        answered = last is not None and last.error_code is None
        return TraceMeta(
            request_id=request_id,
            role=role,
            provider=last.provider if last else None,
            model=last.model if last else None,
            fallback_used=answered and len(self.attempts) > 1,
            attempts=len(self.attempts),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            prompt_version=prompt_version,
            degraded=degraded,
            error_code=last.error_code if last else None,
        )


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def _error_code(error: BaseException) -> str:
    return error.code if isinstance(error, AppError) else type(error).__name__


def _metadata(meta: TraceMeta | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    if isinstance(meta, TraceMeta):
        return meta.model_dump(exclude_none=True)
    return dict(meta)


def _swallow(action: str) -> None:
    logger.warning("tracing failed; continuing without it", extra={"action": action}, exc_info=True)


class Observation:
    """A span or generation in progress. Updates never raise; on the no-op tracer they do nothing."""

    def __init__(self, span: Any = None) -> None:
        self._span = span

    @property
    def trace_id(self) -> str | None:
        return getattr(self._span, "trace_id", None)

    def update(
        self,
        *,
        output: Any = None,
        meta: TraceMeta | Mapping[str, Any] | None = None,
        level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
        status_message: str | None = None,
    ) -> None:
        if self._span is None:
            return
        try:
            self._span.update(output=output, metadata=_metadata(meta), level=level, status_message=status_message)
        except Exception:
            _swallow("update")


class Telemetry:
    """The process's tracer. ``Telemetry()`` without a client is the no-op tracer."""

    def __init__(self, client: Langfuse | None = None, *, public_key: str | None = None) -> None:
        self._client = client
        # The LangChain handler finds its client by public key; without one it may pick a disabled client.
        self._public_key = public_key

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        as_type: ObservationType = "span",
        input: Any = None,
        meta: TraceMeta | Mapping[str, Any] | None = None,
    ) -> Generator[Observation]:
        """Trace the block as a child of the current observation, or as a new trace.

        An exception from the block marks the observation as an error (with the
        app error code, not the message or stack) and is re-raised unchanged.
        """
        context: AbstractContextManager[Any] | None = None
        observation = Observation()
        if self._client is not None:
            try:
                context = self._client.start_as_current_observation(
                    name=name, as_type=as_type, input=input, metadata=_metadata(meta)
                )
                observation = Observation(context.__enter__())
            except Exception:
                context = None
                _swallow("start")
        try:
            yield observation
        except BaseException as exc:
            observation.update(level="ERROR", status_message=_error_code(exc))
            raise
        finally:
            if context is not None:
                try:
                    # Never hand the exception to OpenTelemetry: it would export the message and stack trace.
                    context.__exit__(None, None, None)
                except Exception:
                    _swallow("end")

    def callbacks(self) -> list[BaseCallbackHandler]:
        """LangChain callbacks that trace model and chain runs; empty when tracing is off."""
        if self._client is None:
            return []
        try:
            from langfuse.langchain import CallbackHandler

            return [CallbackHandler(public_key=self._public_key)]
        except Exception:
            _swallow("callback")
            return []

    def current_trace_id(self) -> str | None:
        if self._client is None:
            return None
        try:
            return self._client.get_current_trace_id()
        except Exception:
            _swallow("trace_id")
            return None

    def flush(self) -> None:
        """Send buffered spans now; call before a CLI or test exits."""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            _swallow("flush")

    def shutdown(self) -> None:
        if self._client is None:
            return
        try:
            self._client.shutdown()
        except Exception:
            _swallow("shutdown")


_telemetry = Telemetry()


def init_telemetry(
    settings: Settings,
    *,
    tracer_provider: TracerProvider | None = None,
    span_exporter: SpanExporter | None = None,
) -> Telemetry:
    """Create the process tracer from settings and make it the one ``get_telemetry`` returns.

    Without both Langfuse keys, or with ``TRACING_ENABLED=false``, the tracer is
    a no-op. ``tracer_provider`` and ``span_exporter`` are injectable for tests.
    """
    global _telemetry
    client = _build_client(settings, tracer_provider=tracer_provider, span_exporter=span_exporter)
    public_key = settings.langfuse_public_key.get_secret_value() if client and settings.langfuse_public_key else None
    _telemetry = Telemetry(client, public_key=public_key)
    return _telemetry


def get_telemetry() -> Telemetry:
    return _telemetry


def _build_client(
    settings: Settings,
    *,
    tracer_provider: TracerProvider | None,
    span_exporter: SpanExporter | None,
) -> Langfuse | None:
    if not settings.tracing_enabled:
        logger.info("tracing off", extra={"reason": "disabled"})
        return None
    public_key = settings.langfuse_public_key.get_secret_value() if settings.langfuse_public_key else ""
    secret_key = settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else ""
    if not (public_key and secret_key):
        logger.info("tracing off", extra={"reason": "no_keys"})
        return None
    # Read by the Langfuse client when it starts; see the module docstring.
    os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] = "false"
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=settings.langfuse_host,
            environment=settings.environment,
            release=__version__,
            sample_rate=settings.trace_sample_rate,
            mask=mask_trace_data,
            tracer_provider=tracer_provider,
            span_exporter=span_exporter,
        )
    except Exception:
        _swallow("init")
        return None
    logger.info("tracing on", extra={"host": settings.langfuse_host, "sample_rate": settings.trace_sample_rate})
    return client
