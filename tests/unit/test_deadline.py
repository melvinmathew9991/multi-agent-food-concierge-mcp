"""The wall-clock deadline across model calls: attempts are cut to the time left, or not started."""

import json
import logging
from typing import Any, Literal

import boto3
import httpx2
import pytest
from botocore.stub import Stubber
from pydantic import BaseModel, SecretStr

from food_concierge import errors
from food_concierge.config import Settings
from food_concierge.models import deadline
from food_concierge.models.deadline import MIN_ATTEMPT_S, attempt_timeout, model_deadline, time_left
from food_concierge.models.router import build_chat_model, get_chat_model
from food_concierge.models.structured import get_structured_model

GROQ = "api.groq.com"
GEMINI = "generativelanguage.googleapis.com"


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(deadline, "_clock", fake)
    return fake


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key=SecretStr("gsk-test"),
        gemini_api_key=SecretStr("gemini-test"),
        groq_chat_model="groq-chat",
        gemini_chat_model="gemini-chat",
    )


def _completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


class Providers:
    """Replies per host from a queue; each request takes ``seconds`` on the fake clock. Records read timeouts."""

    def __init__(self, clock: Clock, seconds: float = 0.0, **replies: list[dict[str, Any]]) -> None:
        self.clock = clock
        self.seconds = seconds
        self.replies = {GROQ: replies.get("groq", []), GEMINI: replies.get("gemini", [])}
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.clock.now += self.seconds
        return httpx2.Response(200, json=self.replies[request.url.host].pop(0))

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    def timeouts(self) -> list[float]:
        return [r.extensions["timeout"]["read"] for r in self.requests]

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))

    def async_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))


def _text(content: str = "hello") -> dict[str, Any]:
    return _completion({"content": content})


# ---------------------------------------------------------------------------
# The deadline itself
# ---------------------------------------------------------------------------


def test_no_deadline_outside_the_block(clock: Clock) -> None:
    assert time_left() is None
    with model_deadline(8):
        clock.now += 3
        assert time_left() == 5
    assert time_left() is None


def test_a_nested_deadline_never_extends_the_outer_one(clock: Clock) -> None:
    with model_deadline(5):
        with model_deadline(20):
            assert time_left() == 5
        with model_deadline(2):
            assert time_left() == 2
        assert time_left() == 5


def test_attempts_are_cut_to_the_time_left(clock: Clock) -> None:
    assert attempt_timeout(3.0, "groq") == 3.0  # no deadline: the attempt keeps its own timeout
    with model_deadline(8):
        assert attempt_timeout(3.0, "groq") == 3.0
        clock.now += 6.5
        assert attempt_timeout(3.0, "groq") == 1.5
        clock.now += 1.5 - MIN_ATTEMPT_S / 2
        with pytest.raises(errors.DeadlineExceededError) as caught:
            attempt_timeout(3.0, "groq")
    assert caught.value.provider == "groq"
    assert not caught.value.falls_back


# ---------------------------------------------------------------------------
# Guarded models
# ---------------------------------------------------------------------------


def test_requests_are_unchanged_outside_a_deadline(clock: Clock, settings: Settings) -> None:
    fake = Providers(clock, groq=[_text()])

    build_chat_model("groq", "chat", settings, http_client=fake.client()).invoke("hi")

    assert fake.timeouts() == [settings.groq_timeout_s]


def test_a_request_gets_only_the_time_left(clock: Clock, settings: Settings) -> None:
    fake = Providers(clock, groq=[_text(), _text()])
    model = build_chat_model("groq", "chat", settings, http_client=fake.client())

    with model_deadline(4):
        model.invoke("hi")
        clock.now += 2.5
        model.invoke("hi")

    assert fake.timeouts() == [3.0, 1.5]


async def test_async_and_streamed_requests_get_only_the_time_left(clock: Clock, settings: Settings) -> None:
    stream = 'data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'  # noqa: E501
    timeouts: list[float] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        timeouts.append(request.extensions["timeout"]["read"])
        if json.loads(request.content).get("stream"):
            return httpx2.Response(200, text=stream, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=_text())

    transport = httpx2.MockTransport(reply)
    model = build_chat_model(
        "groq",
        "chat",
        settings,
        http_client=httpx2.Client(transport=transport),
        http_async_client=httpx2.AsyncClient(transport=transport),
    )
    with model_deadline(2):
        await model.ainvoke("hi")
        list(model.stream("hi"))
        [chunk async for chunk in model.astream("hi")]

    assert timeouts == [2.0, 2.0, 2.0]


def test_no_request_starts_once_the_deadline_has_passed(
    clock: Clock, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    fake = Providers(clock)
    model = get_chat_model("chat", settings, http_client=fake.client())

    with model_deadline(1), caplog.at_level(logging.WARNING, logger="food_concierge.models.router"):
        clock.now += 1
        with pytest.raises(errors.DeadlineExceededError) as caught:
            model.invoke("hi")

    assert fake.requests == []
    assert caught.value.provider == "groq"  # and Gemini was not tried: the deadline does not fall back
    assert [getattr(r, "error_code", None) for r in caplog.records] == ["deadline_exceeded"]


def _bedrock() -> Any:
    s = Settings(
        _env_file=None,
        allow_paid_providers=True,
        chat_provider="bedrock",
        fallback_providers=[],
        bedrock_chat_model_id="anthropic.test-model",
    )
    client = boto3.client(
        "bedrock-runtime", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )
    return build_chat_model("bedrock", "chat", s, bedrock_client=client), client


def test_bedrock_does_not_start_once_the_deadline_has_passed(clock: Clock) -> None:
    model, client = _bedrock()
    streaming = model.model_copy(update={"disable_streaming": False})

    with Stubber(client), model_deadline(1):  # no stubbed responses: any request would fail the test
        clock.now += 1
        with pytest.raises(errors.DeadlineExceededError):
            model.invoke("hi")
        with pytest.raises(errors.DeadlineExceededError):
            list(streaming._stream([]))


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------


class MealRequest(BaseModel):
    diet: Literal["veg", "vegan", "any"]


def _tool_call(diet: str) -> dict[str, Any]:
    function = {"name": "MealRequest", "arguments": json.dumps({"diet": diet})}
    call = {"id": "call-1", "type": "function", "function": function}
    return _completion({"content": None, "tool_calls": [call]})


def test_typed_output_keeps_repair_and_fallback_inside_the_request_deadline(clock: Clock, settings: Settings) -> None:
    # Each request takes 2.5 s. Without the deadline: Groq 3 s + 3 s, Gemini 5 s, 11 s worst case.
    fake = Providers(clock, seconds=2.5, groq=[_tool_call("bad"), _tool_call("bad")], gemini=[_tool_call("vegan")])

    result = get_structured_model(MealRequest, "router", settings, http_client=fake.client()).invoke("vegan")

    assert result.diet == "vegan"
    assert fake.hosts() == [GROQ, GROQ, GEMINI]
    assert fake.timeouts() == [3.0, 3.0, 3.0]  # Gemini's 5 s cut to the 3 s left of the 8 s deadline
    assert time_left() is None


async def test_typed_output_stops_when_the_deadline_is_spent(clock: Clock, settings: Settings) -> None:
    fake = Providers(clock, seconds=4.0, groq=[_tool_call("bad"), _tool_call("bad")])

    model = get_structured_model(MealRequest, "router", settings, http_async_client=fake.async_client())
    with pytest.raises(errors.DeadlineExceededError) as caught:
        await model.ainvoke("vegan")

    assert fake.hosts() == [GROQ, GROQ]
    assert fake.timeouts() == [3.0, 3.0]
    assert caught.value.provider == "gemini"  # Groq's failed repair fell back; Gemini had no time to start
