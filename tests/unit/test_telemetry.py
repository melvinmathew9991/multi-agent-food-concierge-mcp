"""Tracing: masking in both directions, no-op modes, swallowed failures and exported spans (no network)."""

import json
import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from pydantic import BaseModel, SecretStr

from food_concierge import errors, telemetry
from food_concierge.config import Settings
from food_concierge.logging_setup import TraceContextFilter
from food_concierge.models.fake import ScriptedChatModel
from food_concierge.telemetry import (
    MAX_DEPTH,
    MAX_ITEMS,
    MAX_TEXT_CHARS,
    ModelCallRecorder,
    Telemetry,
    TraceMeta,
    init_telemetry,
    mask_text,
    mask_trace_data,
    mask_value,
)

IMAGE_B64 = "iVBORw0KGgo" + "A" * 400

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail me at priya.k+food@example.co.in please", "mail me at [email] please"),
        ("call +91 98765 43210 after 6", "call [phone] after 6"),
        ("call 9876543210", "call [phone]"),
        ("US line (555) 123-4567 ext", "US line [phone] ext"),
        ("ring +44 20 7946 0958.", "ring [phone]."),
        (f"photo: data:image/jpeg;base64,{IMAGE_B64} end", "photo: [image] end"),
        (f"raw {IMAGE_B64}", "raw [binary]"),
    ],
    ids=["email", "intl-phone", "plain-mobile", "us-phone", "uk-phone", "data-uri", "bare-base64"],
)
def test_sensitive_text_is_masked(text: str, expected: str) -> None:
    assert mask_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Paneer tikka ₹349, serves 2",
        "Total $12.50 for 2 items",
        "620 kcal, 32 g protein, 1,250 kJ",
        "ordered 2026-10-01 12:30, delivered 2026-10-01T13:05:00Z",
        "date range 2026-10-01 - 2026-10-07",
        "dish_0042 from rest-17 in 400 g",
        "trace 4bf92f3577b34da6a3ce929d0e0e4736",
        "id 123e4567-e89b-12d3-a456-426614174000",
        "version 1.2.3 at 10:45:30",
        "score 0.8312 from 1240 reviews",
    ],
    ids=[
        "price-inr",
        "price-usd",
        "nutrition",
        "datetimes",
        "date-range",
        "ids",
        "trace-id",
        "uuid",
        "version",
        "stats",
    ],
)
def test_ordinary_values_are_not_masked(text: str) -> None:
    assert mask_text(text) == text


def test_long_text_is_cut() -> None:
    masked = mask_text("a " * MAX_TEXT_CHARS)

    assert masked.startswith("a a ")
    assert masked.endswith(f"…[+{MAX_TEXT_CHARS} chars]")


class _Colour(Enum):
    RED = "red"


class _Order(BaseModel):
    email: str
    total: float


def test_values_are_masked_recursively_and_made_json_ready() -> None:
    data = {
        "messages": [{"role": "user", "content": [{"type": "image_url", "url": f"data:image/png;base64,{IMAGE_B64}"}]}],
        "image": b"\x89PNG....",
        "contact": ("a@b.io", 9876543210),
        "order": _Order(email="x@y.com", total=12.5),
        "when": datetime(2026, 10, 1, 12, 30, tzinfo=UTC),
        "day": date(2026, 10, 1),
        "id": uuid.UUID("123e4567-e89b-12d3-a456-426614174000"),
        "colour": _Colour.RED,
        "tags": {"vegan"},
        "ok": True,
        "none": None,
        "who@example.com": 1,
    }

    masked = mask_value(data)

    assert masked == {
        "messages": [{"role": "user", "content": [{"type": "image_url", "url": "[image]"}]}],
        "image": "[bytes: 8]",
        "contact": ["[email]", 9876543210],  # numbers are data, not text: only strings are scanned
        "order": {"email": "[email]", "total": 12.5},
        "when": "2026-10-01T12:30:00+00:00",
        "day": "2026-10-01",
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "colour": "red",
        "tags": ["vegan"],
        "ok": True,
        "none": None,
        "[email]": 1,
    }
    json.dumps(masked)
    assert data["contact"] == ("a@b.io", 9876543210)  # the input is untouched


def test_unknown_objects_are_exported_as_masked_text() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "Opaque(owner=a@b.io)"

    assert mask_value(Opaque()) == "Opaque(owner=[email])"


def test_masking_is_bounded() -> None:
    nested: dict[str, Any] = {}
    cursor = nested
    for _ in range(MAX_DEPTH + 5):
        cursor["next"] = {}
        cursor = cursor["next"]
    cyclic: list[Any] = []
    cyclic.append(cyclic)

    deep = mask_value(nested)
    for _ in range(MAX_DEPTH):
        deep = deep["next"]

    assert deep == "[truncated: too deep]"
    assert json.dumps(mask_value(cyclic))
    assert mask_value(list(range(MAX_ITEMS + 3)))[-1] == "[+3 items]"
    assert mask_value({str(i): i for i in range(MAX_ITEMS + 2)})["[truncated]"] == "+2 items"


def test_the_hook_fails_closed(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def broken(value: Any, depth: int = 0) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(telemetry, "mask_value", broken)

    assert mask_trace_data(data={"email": "a@b.io"}) == "[masking failed]"
    assert "trace masking failed" in caplog.text


# ---------------------------------------------------------------------------
# No-op modes and swallowed failures
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tracing_enabled": False, "langfuse_public_key": SecretStr("pk"), "langfuse_secret_key": SecretStr("sk")},
        {},
        {"langfuse_public_key": SecretStr("pk-only")},
    ],
    ids=["switched-off", "no-keys", "one-key"],
)
def test_tracing_is_a_no_op_without_configuration(overrides: dict[str, Any]) -> None:
    tracer = init_telemetry(_settings(**overrides))

    with tracer.observe("request", input="hi") as observation:
        observation.update(output="hello", meta={"provider": "groq"})

    assert tracer is telemetry.get_telemetry()
    assert not tracer.enabled
    assert observation.trace_id is None
    assert tracer.callbacks() == []
    assert tracer.current_trace_id() is None
    tracer.flush()
    tracer.shutdown()


def _failed_actions(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [getattr(record, "action", "") for record in caplog.records if record.name == "food_concierge.telemetry"]


class _BrokenClient:
    """A Langfuse client whose every call fails, as when the SDK or the server misbehaves."""

    def __getattr__(self, name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"langfuse {name} failed")

        return fail


class _BrokenSpan:
    trace_id = "t" * 32

    def update(self, **kwargs: Any) -> None:
        raise RuntimeError("update failed")


class _BrokenExitContext:
    def __enter__(self) -> _BrokenSpan:
        return _BrokenSpan()

    def __exit__(self, *exc: object) -> None:
        raise RuntimeError("end failed")


class _ClientWithBrokenSpans:
    def start_as_current_observation(self, **kwargs: Any) -> _BrokenExitContext:
        return _BrokenExitContext()


def test_langfuse_failures_never_reach_the_caller(caplog: pytest.LogCaptureFixture) -> None:
    tracer = Telemetry(_BrokenClient())  # type: ignore[arg-type]

    with tracer.observe("request") as observation:
        ran = True
    tracer.flush()
    tracer.shutdown()

    assert ran
    assert observation.trace_id is None
    assert tracer.current_trace_id() is None
    assert _failed_actions(caplog) == ["start", "flush", "shutdown", "trace_id"]


def test_span_update_and_end_failures_are_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    tracer = Telemetry(_ClientWithBrokenSpans())  # type: ignore[arg-type]

    with tracer.observe("request") as observation:
        observation.update(output="x")

    assert observation.trace_id == "t" * 32
    assert _failed_actions(caplog) == ["update", "end"]


def test_callback_failure_returns_no_callbacks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import langfuse.langchain

    def broken(**kwargs: Any) -> Any:
        raise RuntimeError("no langchain")

    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", broken)

    assert Telemetry(_BrokenClient()).callbacks() == []  # type: ignore[arg-type]
    assert _failed_actions(caplog) == ["callback"]


def test_client_start_failure_leaves_tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import langfuse

    def broken(**kwargs: Any) -> Any:
        raise RuntimeError("bad host")

    monkeypatch.setattr(langfuse, "Langfuse", broken)
    settings = _settings(langfuse_public_key=SecretStr("pk"), langfuse_secret_key=SecretStr("sk"))

    assert not init_telemetry(settings).enabled


# ---------------------------------------------------------------------------
# Exported spans (real Langfuse client, in-memory exporter)
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter) -> Iterator[Telemetry]:
    # Restored after the test; init_telemetry must switch media upload off itself.
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "true")
    # Langfuse keeps one client per public key, so each test gets its own.
    settings = _settings(
        langfuse_public_key=SecretStr(f"pk-lf-{uuid.uuid4()}"),
        langfuse_secret_key=SecretStr("sk-lf-test"),
        environment="ci",
    )
    traced = init_telemetry(settings, tracer_provider=TracerProvider(), span_exporter=exporter)
    yield traced
    traced.shutdown()
    monkeypatch.setattr(telemetry, "_telemetry", Telemetry())


def _exported(tracer: Telemetry, exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    tracer.flush()
    return list(exporter.get_finished_spans())


def _attributes_text(spans: list[ReadableSpan]) -> str:
    return " ".join(str(value) for span in spans for value in (span.attributes or {}).values())


def test_exported_spans_are_masked(tracer: Telemetry, exporter: InMemorySpanExporter) -> None:
    query = {
        "text": "vegan thali under ₹349, email me at priya@example.com",
        "image": f"data:image/png;base64,{IMAGE_B64}",
    }

    with tracer.observe("request", input=query, meta=TraceMeta(request_id="req-1", provider="groq")) as observation:
        observation.update(output="Call 9876543210 to order dish_0042 (620 kcal)", meta={"stage": "answer"})

    exported = _attributes_text(_exported(tracer, exporter))
    assert tracer.enabled
    assert "priya@example.com" not in exported
    assert IMAGE_B64[:40] not in exported
    assert "9876543210" not in exported
    for kept in ("thali under", "349", "dish_0042", "620 kcal", "req-1", "groq", "answer"):
        assert kept in exported
    assert "[email]" in exported
    assert "[image]" in exported


def test_media_upload_is_switched_off(tracer: Telemetry) -> None:
    import os

    assert os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] == "false"


def test_nested_observations_share_one_trace(tracer: Telemetry, exporter: InMemorySpanExporter) -> None:
    with tracer.observe("request") as outer:
        inside = tracer.current_trace_id()
        with tracer.observe("retrieve", as_type="retriever") as inner:
            pass

    spans = _exported(tracer, exporter)
    assert outer.trace_id is not None
    assert outer.trace_id == inner.trace_id == inside
    assert {format(span.context.trace_id, "032x") for span in spans} == {outer.trace_id}
    assert tracer.current_trace_id() is None


def test_failed_block_is_marked_without_message_or_stack(tracer: Telemetry, exporter: InMemorySpanExporter) -> None:
    with pytest.raises(errors.ProviderUnavailableError), tracer.observe("request"):
        raise errors.ProviderUnavailableError("upstream said: user a@b.io not found")

    (span,) = _exported(tracer, exporter)
    attributes = span.attributes or {}
    assert attributes["langfuse.observation.level"] == "ERROR"
    assert attributes["langfuse.observation.status_message"] == "provider_unavailable"
    assert not span.events  # no exception event carrying the message or stack trace
    assert "a@b.io" not in _attributes_text([span])


def test_langchain_runs_are_traced_and_masked(tracer: Telemetry, exporter: InMemorySpanExporter) -> None:
    model = ScriptedChatModel(script=[AIMessage("Try the dal, call 9876543210")])

    with tracer.observe("request"):
        model.invoke([HumanMessage("I am priya@example.com")], config={"callbacks": tracer.callbacks()})

    spans = _exported(tracer, exporter)
    exported = _attributes_text(spans)
    assert len(spans) >= 2  # the request and the model run
    assert "priya@example.com" not in exported
    assert "9876543210" not in exported
    assert "Try the dal" in exported


# ---------------------------------------------------------------------------
# Model call recorder
# ---------------------------------------------------------------------------


def _usage(inp: int, out: int) -> dict[str, int]:
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


def test_recorder_sees_a_fallback() -> None:
    primary = ScriptedChatModel(provider="groq", model_name="g", script=[errors.ProviderRateLimitedError()])
    backup = ScriptedChatModel(provider="gemini", model_name="m", script=[AIMessage("hi", usage_metadata=_usage(7, 3))])
    recorder = ModelCallRecorder()

    primary.with_fallbacks([backup], exceptions_to_handle=(errors.ProviderError,)).invoke(
        "hi", config={"callbacks": [recorder]}
    )

    assert recorder.meta(role="chat", request_id="r1", prompt_version="v1") == TraceMeta(
        request_id="r1",
        role="chat",
        provider="gemini",
        model="m",
        fallback_used=True,
        attempts=2,
        input_tokens=7,
        output_tokens=3,
        prompt_version="v1",
    )
    assert [a.error_code for a in recorder.attempts] == ["provider_rate_limited", None]


def test_recorder_when_every_provider_fails() -> None:
    primary = ScriptedChatModel(provider="groq", script=[errors.ProviderTimeoutError()])
    backup = ScriptedChatModel(provider="gemini", script=[ValueError("odd")])
    recorder = ModelCallRecorder()

    with pytest.raises(errors.ProviderTimeoutError):
        primary.with_fallbacks([backup]).invoke("hi", config={"callbacks": [recorder]})

    meta = recorder.meta(degraded=True)
    assert meta.provider == "gemini"
    assert meta.error_code == "ValueError"
    assert not meta.fallback_used
    assert meta.degraded


def test_recorder_ignores_unknown_runs_and_starts_empty() -> None:
    recorder = ModelCallRecorder()
    recorder.on_llm_error(RuntimeError("x"), run_id=uuid.uuid4())

    assert recorder.meta() == TraceMeta()


# ---------------------------------------------------------------------------
# trace_id in log lines
# ---------------------------------------------------------------------------


def _record() -> logging.LogRecord:
    return logging.makeLogRecord({"msg": "hello"})


def _span(sampled: bool) -> NonRecordingSpan:
    flags = TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT)
    return NonRecordingSpan(SpanContext(trace_id=0xABC, span_id=0x1, is_remote=False, trace_flags=flags))


def test_log_lines_inside_a_trace_carry_its_id() -> None:
    record = _record()
    with trace.use_span(_span(sampled=True)):
        assert TraceContextFilter().filter(record)

    assert getattr(record, "trace_id", None) == f"{0xABC:032x}"


def test_log_lines_outside_a_sampled_trace_have_no_id() -> None:
    outside, unsampled = _record(), _record()
    TraceContextFilter().filter(outside)
    with trace.use_span(_span(sampled=False)):
        TraceContextFilter().filter(unsampled)

    assert not hasattr(outside, "trace_id")
    assert not hasattr(unsampled, "trace_id")


def test_an_explicit_trace_id_is_kept() -> None:
    record = _record()
    record.trace_id = "given"
    with trace.use_span(_span(sampled=True)):
        TraceContextFilter().filter(record)

    assert getattr(record, "trace_id", None) == "given"
