"""Call and token caps per provider: the ledger itself, then enforcement through the router."""

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

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
from food_concierge.models.usage import Caps, UsageLedger, get_usage_ledger

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


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_daily_cap_resets_at_utc_midnight(ledger: UsageLedger, clock: Clock) -> None:
    caps = Caps(per_day=2)
    ledger.acquire("groq", caps)
    clock.now += 3600
    ledger.acquire("groq", caps)

    with pytest.raises(errors.BudgetExceededError) as caught:
        ledger.acquire("groq", caps)
    assert caught.value.provider == "groq"
    assert caught.value.falls_back
    assert "Today's" in caught.value.message

    clock.now = datetime(2026, 10, 7, 0, 0, 1, tzinfo=UTC).timestamp()
    ledger.acquire("groq", caps)
    assert ledger.usage("groq").calls_today == 1


def test_minute_cap_is_a_rolling_window(ledger: UsageLedger, clock: Clock) -> None:
    caps = Caps(per_minute=2)
    ledger.acquire("gemini", caps)
    clock.now += 30
    ledger.acquire("gemini", caps)
    clock.now += 29

    with pytest.raises(errors.BudgetExceededError, match="busy"):
        ledger.acquire("gemini", caps)

    clock.now += 1  # the first call is now a minute old
    ledger.acquire("gemini", caps)
    assert ledger.usage("gemini").calls_last_minute == 2
    assert ledger.usage("gemini").calls_today == 3


def test_token_cap_counts_reported_usage(ledger: UsageLedger, clock: Clock) -> None:
    caps = Caps(tokens_per_minute=1000)
    ledger.acquire("groq", caps)
    ledger.record_tokens("groq", 600)
    ledger.acquire("groq", caps)  # 600 < 1000: the next call may start
    ledger.record_tokens("groq", 500)

    with pytest.raises(errors.BudgetExceededError, match="busy"):
        ledger.acquire("groq", caps)

    clock.now += 61
    ledger.acquire("groq", caps)
    assert ledger.usage("groq").tokens_last_minute == 0


def test_refused_attempts_are_not_counted(ledger: UsageLedger) -> None:
    caps = Caps(per_minute=1)
    ledger.acquire("groq", caps)
    for _ in range(3):
        with pytest.raises(errors.BudgetExceededError):
            ledger.acquire("groq", caps)

    assert ledger.usage("groq").calls_today == 1


def test_uncapped_zero_and_empty_usage(ledger: UsageLedger) -> None:
    for _ in range(500):
        ledger.acquire("ollama", Caps())
    ledger.record_tokens("ollama", 0)
    ledger.record_tokens("ollama", -5)

    assert ledger.usage("ollama").tokens_last_minute == 0
    with pytest.raises(errors.BudgetExceededError):
        ledger.acquire("groq", Caps(per_day=0))  # 0 switches a provider off
    assert ledger.usage("never-called").model_dump() == {
        "calls_today": 0,
        "calls_last_minute": 0,
        "tokens_last_minute": 0,
    }


def test_providers_are_counted_separately(ledger: UsageLedger) -> None:
    caps = Caps(per_minute=1)
    ledger.acquire("groq", caps)
    ledger.acquire("gemini", caps)

    ledger.reset()
    assert ledger.usage("groq").calls_today == 0


def test_concurrent_attempts_never_exceed_a_cap(ledger: UsageLedger) -> None:
    caps = Caps(per_minute=25)
    admitted: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        try:
            ledger.acquire("groq", caps)
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


def test_caps_come_from_settings() -> None:
    s = Settings(_env_file=None)

    assert Caps.for_provider(s, "groq") == Caps(per_day=900, per_minute=25, tokens_per_minute=8000)
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
        gemini_chat_model="gemini-chat",
        minute_call_limit_groq=2,
    )


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
    assert get_usage_ledger().usage("groq").calls_today == 2


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

    assert get_usage_ledger().usage("groq").calls_today == 1  # the provider counted the rejected request too


def test_reported_tokens_reach_the_token_cap(settings: Settings) -> None:
    s = settings.model_copy(update={"minute_call_limit_groq": 25, "minute_token_limit_groq": 1000})
    fake = Providers(total_tokens=600)
    model = get_chat_model("chat", s, http_client=fake.client())

    for _ in range(3):
        model.invoke("hi")

    assert get_usage_ledger().usage("groq").tokens_last_minute == 1200
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

    assert get_usage_ledger().usage("groq").model_dump() == {
        "calls_today": 1,
        "calls_last_minute": 1,
        "tokens_last_minute": 42,
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

    assert get_usage_ledger().usage("groq").tokens_last_minute == 42


def test_an_attempt_refused_by_the_deadline_uses_no_quota(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_now = Clock(1000.0)
    monkeypatch.setattr(deadline, "_clock", fake_now)
    model = build_chat_model("groq", "chat", settings, http_client=Providers().client())

    with model_deadline(1):
        fake_now.now += 1
        with pytest.raises(errors.DeadlineExceededError):
            model.invoke("hi")

    assert get_usage_ledger().usage("groq").calls_today == 0


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
