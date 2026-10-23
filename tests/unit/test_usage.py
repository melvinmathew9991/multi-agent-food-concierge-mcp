"""Call and token caps per model and the photo budget: the ledger itself, then enforcement through the router."""

import json
import logging
import threading
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from typing import Any, cast

import boto3
import httpx2
import pytest
from botocore.stub import Stubber
from pydantic import SecretStr

from food_concierge import errors
from food_concierge.config import Settings
from food_concierge.models import deadline
from food_concierge.models.deadline import model_deadline
from food_concierge.models.router import build_chat_model, get_chat_model
from food_concierge.models.usage import PHOTO_LIMIT, PHOTOS, Budget, Caps, UsageLedger, budgets_for, get_usage_ledger
from food_concierge.telemetry import ModelCallRecorder

GROQ = "api.groq.com"
GEMINI = "generativelanguage.googleapis.com"
NOON = datetime(2026, 10, 6, 12, 0, tzinfo=UTC).timestamp()


class Clock:
    def __init__(self, now: float = NOON) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def ledger(clock: Clock) -> UsageLedger:
    return UsageLedger(clock=clock)


def one(key: str, caps: Caps) -> list[Budget]:
    return [Budget(key=key, caps=caps)]


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_daily_cap_resets_at_utc_midnight(ledger: UsageLedger, clock: Clock) -> None:
    caps = Caps(per_day=2)
    ledger.acquire("groq", one("groq", caps))
    clock.now += 3600
    ledger.acquire("groq", one("groq", caps))

    with pytest.raises(errors.BudgetExceededError) as caught:
        ledger.acquire("groq", one("groq", caps))
    assert caught.value.provider == "groq"
    assert caught.value.falls_back
    assert "Today's" in caught.value.message

    clock.now = datetime(2026, 10, 7, 0, 0, 1, tzinfo=UTC).timestamp()
    ledger.acquire("groq", one("groq", caps))
    assert ledger.usage("groq").calls_today == 1


def test_minute_cap_is_a_rolling_window(ledger: UsageLedger, clock: Clock) -> None:
    caps = Caps(per_minute=2)
    ledger.acquire("gemini", one("gemini", caps))
    clock.now += 30
    ledger.acquire("gemini", one("gemini", caps))
    clock.now += 29

    with pytest.raises(errors.BudgetExceededError, match="busy"):
        ledger.acquire("gemini", one("gemini", caps))

    clock.now += 1  # the first call is now a minute old
    ledger.acquire("gemini", one("gemini", caps))
    assert ledger.usage("gemini").calls_last_minute == 2
    assert ledger.usage("gemini").calls_today == 3


def test_token_cap_counts_reported_usage(ledger: UsageLedger, clock: Clock) -> None:
    caps = Caps(tokens_per_minute=1000)
    ledger.acquire("groq", one("groq", caps))
    ledger.record_tokens("groq", 600)
    ledger.acquire("groq", one("groq", caps))  # 600 < 1000: the next call may start
    ledger.record_tokens("groq", 500)

    with pytest.raises(errors.BudgetExceededError, match="busy"):
        ledger.acquire("groq", one("groq", caps))

    clock.now += 61
    ledger.acquire("groq", one("groq", caps))
    assert ledger.usage("groq").tokens_last_minute == 0


def test_running_token_total_measures_one_call_exactly(ledger: UsageLedger, clock: Clock) -> None:
    ledger.record_tokens("groq", 1900)
    clock.now += 50
    before = ledger.usage("groq")
    clock.now += 20  # the earlier call leaves the rolling minute while this one is in flight
    ledger.record_tokens("groq", 1920)
    after = ledger.usage("groq")

    assert after.tokens_last_minute - before.tokens_last_minute == 20  # the old, undercounting difference (#74)
    assert after.tokens_total - before.tokens_total == 1920
    ledger.reset()
    assert ledger.usage("groq").tokens_total == 0


def test_refused_attempts_are_not_counted(ledger: UsageLedger) -> None:
    caps = Caps(per_minute=1)
    ledger.acquire("groq", one("groq", caps))
    for _ in range(3):
        with pytest.raises(errors.BudgetExceededError):
            ledger.acquire("groq", one("groq", caps))

    assert ledger.usage("groq").calls_today == 1


def test_uncapped_zero_and_empty_usage(ledger: UsageLedger) -> None:
    for _ in range(500):
        ledger.acquire("ollama", one("ollama", Caps()))
    ledger.record_tokens("ollama", 0)
    ledger.record_tokens("ollama", -5)

    assert ledger.usage("ollama").tokens_last_minute == 0
    with pytest.raises(errors.BudgetExceededError):
        ledger.acquire("groq", one("groq", Caps(per_day=0)))  # 0 switches a provider off
    assert ledger.usage("never-called").model_dump() == {
        "calls_today": 0,
        "calls_last_minute": 0,
        "tokens_last_minute": 0,
        "tokens_last_day": 0,
        "tokens_total": 0,
    }


def test_providers_are_counted_separately(ledger: UsageLedger) -> None:
    caps = Caps(per_minute=1)
    ledger.acquire("groq", one("groq", caps))
    ledger.acquire("gemini", one("gemini", caps))

    ledger.reset()
    assert ledger.usage("groq").calls_today == 0


def test_concurrent_attempts_never_exceed_a_cap(ledger: UsageLedger) -> None:
    caps = Caps(per_minute=25)
    admitted: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        try:
            ledger.acquire("groq", one("groq", caps))
            ok = True
        except errors.BudgetExceededError:
            ok = False
        with lock:
            admitted.append(ok)

    threads = [threading.Thread(target=attempt) for _ in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert admitted.count(True) == 25


def test_daily_token_cap_is_a_rolling_day(ledger: UsageLedger, clock: Clock) -> None:
    caps = Caps(tokens_per_day=3000)
    ledger.acquire("groq", one("groq", caps))
    ledger.record_tokens("groq", 1900)
    clock.now += 12 * 3600
    ledger.acquire("groq", one("groq", caps))
    ledger.record_tokens("groq", 1900)

    with pytest.raises(errors.BudgetExceededError, match="Today's"):
        ledger.acquire("groq", one("groq", caps))
    assert ledger.usage("groq").tokens_last_minute == 1900

    clock.now += 12 * 3600  # the first reply is now a day old; Groq's window rolls, it does not reset at midnight
    ledger.acquire("groq", one("groq", caps))
    assert ledger.usage("groq").tokens_last_day == 1900


def test_a_refusing_budget_counts_the_attempt_nowhere(ledger: UsageLedger) -> None:
    model = Budget(key="groq/vision", caps=Caps(per_day=10))
    photos = Budget(key=PHOTOS, caps=Caps(per_day=1), day_message="No more photos today.")
    ledger.acquire("groq", [model, photos])

    with pytest.raises(errors.BudgetExceededError, match="No more photos today"):
        ledger.acquire("groq", [model, photos])

    assert ledger.usage("groq/vision").calls_today == 1
    assert ledger.usage(PHOTOS).calls_today == 1


def test_budgets_are_per_model_and_photos_only_on_hosted_vision() -> None:
    s = Settings(_env_file=None, groq_chat_model="groq-chat", groq_vision_model="groq-vision", daily_photo_limit=5)

    (chat,) = budgets_for(s, "groq", "chat")
    vision, photos = budgets_for(s, "groq", "vision")
    (local,) = budgets_for(s, "ollama", "vision")

    assert (chat.key, vision.key) == ("groq/groq-chat", "groq/groq-vision")
    assert chat.caps == vision.caps == Caps.for_provider(s, "groq")
    assert photos == Budget(key=PHOTOS, caps=Caps(per_day=5), day_message=PHOTO_LIMIT)
    assert "describe the dish in words" in PHOTO_LIMIT
    assert local.key == f"ollama/{s.ollama_vision_model}"
    assert local.caps == Caps()


def test_caps_come_from_settings() -> None:
    s = Settings(_env_file=None)

    assert Caps.for_provider(s, "groq") == Caps(
        per_day=900, per_minute=25, tokens_per_minute=8000, tokens_per_day=180_000
    )
    assert Caps.for_provider(s, "gemini") == Caps(per_day=200, per_minute=8)
    assert Caps.for_provider(s, "bedrock") == Caps(per_day=200)
    assert Caps.for_provider(s, "ollama") == Caps()
    assert Caps.for_provider(s, "fake") == Caps()


# ---------------------------------------------------------------------------
# Enforcement through the router
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key=SecretStr("gsk-test"),
        gemini_api_key=SecretStr("gemini-test"),
        groq_chat_model="groq-chat",
        groq_vision_model="groq-vision",
        gemini_chat_model="gemini-chat",
        minute_call_limit_groq=2,
    )


GROQ_CHAT = "groq/groq-chat"


def _completion(total_tokens: int = 7) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": total_tokens - 2, "completion_tokens": 2, "total_tokens": total_tokens},
    }


class Providers:
    def __init__(self, status: int = 200, total_tokens: int = 7) -> None:
        self.status = status
        self.total_tokens = total_tokens
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.host == GROQ and self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "slow down"}})
        return httpx2.Response(200, json=_completion(self.total_tokens))

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))

    def async_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))


def test_a_capped_provider_falls_back_without_a_request(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    fake = Providers()
    model = get_chat_model("chat", settings, http_client=fake.client())

    with caplog.at_level(logging.WARNING, logger="food_concierge.models.router"):
        for _ in range(3):
            model.invoke("hi")

    assert fake.hosts() == [GROQ, GROQ, GEMINI]
    assert [getattr(r, "error_code", None) for r in caplog.records] == ["budget_exceeded"]
    assert get_usage_ledger().usage(GROQ_CHAT).calls_today == 2


async def test_async_calls_are_capped_too(settings: Settings) -> None:
    fake = Providers()
    model = get_chat_model("chat", settings, http_async_client=fake.async_client())

    for _ in range(3):
        await model.ainvoke("hi")

    assert fake.hosts() == [GROQ, GROQ, GEMINI]


def test_every_provider_capped_raises_the_budget_error(settings: Settings) -> None:
    s = settings.model_copy(update={"minute_call_limit_groq": 0, "minute_call_limit_gemini": 0})
    fake = Providers()

    with pytest.raises(errors.BudgetExceededError) as caught:
        get_chat_model("chat", s, http_client=fake.client()).invoke("hi")

    assert fake.requests == []
    assert caught.value.provider == "groq"


def test_failed_attempts_use_quota(settings: Settings) -> None:
    fake = Providers(status=429)

    get_chat_model("chat", settings, http_client=fake.client()).invoke("hi")

    assert get_usage_ledger().usage(GROQ_CHAT).calls_today == 1  # the provider counted the rejected request too


def test_reported_tokens_reach_the_token_cap(settings: Settings) -> None:
    s = settings.model_copy(update={"minute_call_limit_groq": 25, "minute_token_limit_groq": 1000})
    fake = Providers(total_tokens=600)
    model = get_chat_model("chat", s, http_client=fake.client())

    for _ in range(3):
        model.invoke("hi")

    assert get_usage_ledger().usage(GROQ_CHAT).tokens_last_minute == 1200
    assert fake.hosts() == [GROQ, GROQ, GEMINI]


def test_streamed_usage_is_counted_when_reported(settings: Settings) -> None:
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    events = [
        {**body, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]},
        {**body, "choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 2, "total_tokens": 42}},
    ]
    stream = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, text=stream))
    model = build_chat_model(
        "groq",
        "chat",
        settings,
        http_client=httpx2.Client(transport=transport),
        http_async_client=httpx2.AsyncClient(transport=transport),
    )

    list(model.stream("hi"))

    assert get_usage_ledger().usage(GROQ_CHAT).model_dump() == {
        "calls_today": 1,
        "calls_last_minute": 1,
        "tokens_last_minute": 42,
        "tokens_last_day": 42,
        "tokens_total": 42,
    }


async def test_async_streamed_usage_is_counted_when_reported(settings: Settings) -> None:
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    events = [
        {**body, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]},
        {**body, "choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 2, "total_tokens": 42}},
    ]
    stream = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, text=stream))
    model = build_chat_model("groq", "chat", settings, http_async_client=httpx2.AsyncClient(transport=transport))

    [chunk async for chunk in model.astream("hi")]

    assert get_usage_ledger().usage(GROQ_CHAT).tokens_last_minute == 42


def _stream_with_usage(
    *usages: dict[str, int] | None,
    usage_only: dict[str, int] | None = None,
    requests: list[httpx2.Request] | None = None,
) -> httpx2.Client:
    """One content chunk per entry, each carrying that usage, then an optional usage-only chunk."""
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    events: list[dict[str, Any]] = []
    for i, usage in enumerate(usages):
        finish = "stop" if i == len(usages) - 1 else None
        delta = {"role": "assistant", "content": f"w{i} "}
        events.append({**body, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage})
    if usage_only:
        events.append({**body, "choices": [], "usage": usage_only})
    stream = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"

    def reply(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        return httpx2.Response(200, text=stream)

    return httpx2.Client(transport=httpx2.MockTransport(reply))


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def test_streams_ask_for_usage(settings: Settings) -> None:
    requests: list[httpx2.Request] = []
    model = build_chat_model("groq", "chat", settings, http_client=_stream_with_usage(None, requests=requests))

    list(model.stream("hi"))

    assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    "client",
    [
        # Gemini: a running total on every chunk (probed 2026-10-23).
        lambda: _stream_with_usage(_usage(16, 13), _usage(16, 37), _usage(16, 63)),
        # Groq: the final totals on the last content chunk and again on a usage-only chunk.
        lambda: _stream_with_usage(None, _usage(16, 63), usage_only=_usage(16, 63)),
    ],
    ids=["running-totals", "repeated-totals"],
)
def test_a_streams_usage_is_counted_once(settings: Settings, client: Callable[[], httpx2.Client]) -> None:
    model = build_chat_model("groq", "chat", settings, http_client=client())
    recorder = ModelCallRecorder()

    chunks = list(model.stream("hi", config={"callbacks": [recorder]}))

    assert get_usage_ledger().usage(GROQ_CHAT).tokens_last_minute == 79
    assert [c.usage_metadata["total_tokens"] for c in chunks if c.usage_metadata] == [79]
    assert (recorder.input_tokens, recorder.output_tokens) == (16, 63)


def test_a_stream_the_caller_abandons_still_counts_what_it_reported(settings: Settings) -> None:
    model = build_chat_model("groq", "chat", settings, http_client=_stream_with_usage(_usage(16, 13), _usage(16, 37)))

    stream = cast(Generator[Any], model.stream("hi"))
    next(stream)
    stream.close()

    assert get_usage_ledger().usage(GROQ_CHAT).tokens_last_minute == 29


def test_an_attempt_refused_by_the_deadline_uses_no_quota(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_now = Clock(1000.0)
    monkeypatch.setattr(deadline, "_clock", fake_now)
    model = build_chat_model("groq", "chat", settings, http_client=Providers().client())

    with model_deadline(1):
        fake_now.now += 1
        with pytest.raises(errors.DeadlineExceededError):
            model.invoke("hi")

    assert get_usage_ledger().usage(GROQ_CHAT).calls_today == 0


def test_photos_stop_at_the_photo_budget_and_chat_goes_on(settings: Settings) -> None:
    s = settings.model_copy(update={"daily_photo_limit": 1})
    fake = Providers()
    vision = build_chat_model("groq", "vision", s, http_client=fake.client())
    chat = build_chat_model("groq", "chat", s, http_client=fake.client())

    vision.invoke("describe this")
    with pytest.raises(errors.BudgetExceededError, match="photo limit"):
        vision.invoke("describe this")
    chat.invoke("hi")

    assert fake.hosts() == [GROQ, GROQ]
    ledger = get_usage_ledger()
    assert ledger.usage(PHOTOS).calls_today == 1
    assert ledger.usage(PHOTOS).tokens_total == 7  # photo tokens are counted too
    assert ledger.usage("groq/groq-vision").calls_today == 1
    assert ledger.usage(GROQ_CHAT).calls_today == 1


def test_models_on_one_provider_are_capped_separately(settings: Settings) -> None:
    s = settings.model_copy(update={"minute_call_limit_groq": 1})
    fake = Providers()

    build_chat_model("groq", "chat", s, http_client=fake.client()).invoke("hi")
    build_chat_model("groq", "vision", s, http_client=fake.client()).invoke("describe this")

    assert fake.hosts() == [GROQ, GROQ]  # one call a minute each: Groq limits every model on its own


def test_bedrock_is_capped_too() -> None:
    s = Settings(
        _env_file=None,
        allow_paid_providers=True,
        chat_provider="bedrock",
        fallback_providers=[],
        bedrock_chat_model_id="anthropic.test-model",
        daily_call_limit_bedrock=0,
    )
    client = boto3.client(
        "bedrock-runtime", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )
    model = build_chat_model("bedrock", "chat", s, bedrock_client=client)

    with Stubber(client), pytest.raises(errors.BudgetExceededError):  # no stubbed responses: no request is made
        model.invoke("hi")
