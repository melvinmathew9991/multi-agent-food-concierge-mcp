"""Contract tests for the model router over a mock HTTP transport (no network)."""

import json
import logging
from collections.abc import Callable
from typing import Any

import boto3
import httpx2
import pytest
from botocore.stub import Stubber
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import SecretStr

from food_concierge import errors
from food_concierge.config import Settings
from food_concierge.models.fake import ScriptedChatModel
from food_concierge.models.router import build_chat_model, get_chat_model

GROQ = "api.groq.com"
GEMINI = "generativelanguage.googleapis.com"

Reply = Callable[[httpx2.Request], httpx2.Response]


def _completion(content: str | None = "hello", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def ok(content: str | None = "hello", **kw: Any) -> Reply:
    return lambda request: httpx2.Response(200, json=_completion(content, **kw))


def status(code: int) -> Reply:
    return lambda request: httpx2.Response(code, json={"error": {"message": "provider detail"}})


def raises(exc: Exception) -> Reply:
    def reply(request: httpx2.Request) -> httpx2.Response:
        raise exc

    return reply


class FakeProviders:
    """Routes requests by host to a per-provider reply and records every request."""

    def __init__(self, **replies: Reply) -> None:
        self.replies = {GROQ: replies.get("groq", ok()), GEMINI: replies.get("gemini", ok())}
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.replies[request.url.host](request)

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    def body(self, index: int = 0) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.requests[index].content)
        return loaded

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))

    def async_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key=SecretStr("gsk-test"),
        gemini_api_key=SecretStr("gemini-test"),
        groq_chat_model="groq-chat",
        gemini_chat_model="gemini-chat",
        gemini_vision_model="gemini-vision",
    )


def test_primary_provider_answers(settings: Settings) -> None:
    fake = FakeProviders()

    reply = get_chat_model("chat", settings, http_client=fake.client()).invoke("hi")

    assert reply.content == "hello"
    assert fake.hosts() == [GROQ]
    body = fake.body()
    assert body["model"] == "groq-chat"
    assert body["max_completion_tokens"] == settings.max_output_tokens
    assert body["reasoning_effort"] == "low"
    assert body["temperature"] == settings.chat_temperature
    assert fake.requests[0].headers["authorization"] == "Bearer gsk-test"


@pytest.mark.parametrize(
    "failure",
    [status(429), status(500), status(503), status(401), status(404), raises(httpx2.ReadTimeout("slow"))],
    ids=["rate-limited", "server-error", "unavailable", "auth", "model-gone", "timeout"],
)
def test_failures_fall_back_without_retrying_the_primary(settings: Settings, failure: Reply) -> None:
    fake = FakeProviders(groq=failure)

    reply = get_chat_model("chat", settings, http_client=fake.client()).invoke("hi")

    assert reply.content == "hello"
    # The next provider is the retry: exactly one attempt at Groq, even for a 429 with Retry-After.
    assert fake.hosts() == [GROQ, GEMINI]


def test_rejected_request_does_not_fall_back(settings: Settings) -> None:
    fake = FakeProviders(groq=status(400))

    with pytest.raises(errors.ProviderRequestError) as caught:
        get_chat_model("chat", settings, http_client=fake.client()).invoke("hi")

    assert fake.hosts() == [GROQ]
    assert caught.value.provider == "groq"
    assert "provider detail" not in caught.value.message


def test_empty_reply_counts_as_a_failure(settings: Settings) -> None:
    fake = FakeProviders(groq=ok(content=""))

    assert get_chat_model("chat", settings, http_client=fake.client()).invoke("hi").content == "hello"
    assert fake.hosts() == [GROQ, GEMINI]


def test_all_providers_failing_raises_an_app_error(settings: Settings) -> None:
    fake = FakeProviders(groq=status(503), gemini=ok(content=""))

    with pytest.raises(errors.ProviderUnavailableError) as caught:
        get_chat_model("chat", settings, http_client=fake.client()).invoke("hi")

    assert caught.value.provider == "groq"  # LangChain re-raises the first provider's error


def test_operator_problems_are_logged_at_error(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeProviders(groq=status(401))

    with caplog.at_level(logging.WARNING, logger="food_concierge.models.router"):
        get_chat_model("chat", settings, http_client=fake.client()).invoke("hi")

    record = next(r for r in caplog.records if r.message == "model call failed")
    assert record.levelno == logging.ERROR
    assert (record.provider, record.error_code) == ("groq", "provider_auth_error")  # type: ignore[attr-defined]


def test_tools_bind_across_the_whole_chain(settings: Settings) -> None:
    @tool
    def search_dishes(query: str) -> str:
        """Search the catalog."""
        return query

    call = {"id": "call-1", "type": "function", "function": {"name": "search_dishes", "arguments": '{"query":"curry"}'}}
    fake = FakeProviders(groq=status(429), gemini=ok(content=None, tool_calls=[call]))

    model = get_chat_model("router", settings, http_client=fake.client())
    reply = model.bind_tools([search_dishes]).invoke("find curry")  # type: ignore[attr-defined]

    assert isinstance(reply, AIMessage)
    assert reply.tool_calls[0]["args"] == {"query": "curry"}
    assert [body["tools"][0]["function"]["name"] for body in (fake.body(0), fake.body(1))] == ["search_dishes"] * 2
    assert fake.body(1)["temperature"] == 0.0  # routing is deterministic


def test_roles_without_a_model_skip_that_provider(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeProviders()

    with caplog.at_level(logging.WARNING, logger="food_concierge.models.router"):
        get_chat_model("vision", settings, http_client=fake.client()).invoke("describe")

    assert fake.hosts() == [GEMINI]
    assert fake.body()["model"] == "gemini-vision"
    assert "reasoning_effort" not in fake.body()
    assert any(r.message == "providers skipped for role" for r in caplog.records)


def test_no_model_for_the_role_is_a_config_error() -> None:
    with pytest.raises(errors.ConfigError, match="No provider"):
        get_chat_model("chat", Settings(_env_file=None))


def test_ollama_uses_its_local_endpoint_without_reasoning_effort() -> None:
    requests: list[httpx2.Request] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_completion())

    s = Settings(_env_file=None, chat_provider="ollama", fallback_providers=[])
    get_chat_model("chat", s, http_client=httpx2.Client(transport=httpx2.MockTransport(reply))).invoke("hi")

    assert str(requests[0].url) == "http://localhost:11434/v1/chat/completions"
    assert "reasoning_effort" not in json.loads(requests[0].content)


def test_retries_apply_only_to_the_last_provider(settings: Settings) -> None:
    s = settings.model_copy(update={"provider_max_retries": 1})

    first = build_chat_model("groq", "chat", s, is_last=False)
    last = build_chat_model("gemini", "chat", s, is_last=True)

    assert (first.max_retries, last.max_retries) == (0, 1)  # type: ignore[attr-defined]
    assert (first.request_timeout, last.request_timeout) == (3.0, 5.0)  # type: ignore[attr-defined]


async def test_async_calls_fall_back_too(settings: Settings) -> None:
    fake = FakeProviders(groq=status(429))

    reply = await get_chat_model("chat", settings, http_async_client=fake.async_client()).ainvoke("hi")

    assert reply.content == "hello"
    assert fake.hosts() == [GROQ, GEMINI]


def test_streaming_failures_are_translated(settings: Settings) -> None:
    fake = FakeProviders(groq=status(400))
    model = build_chat_model("groq", "chat", settings, http_client=fake.client())

    with pytest.raises(errors.ProviderRequestError):
        list(model.stream("hi"))


async def test_async_streaming_failures_are_translated(settings: Settings) -> None:
    fake = FakeProviders(groq=status(400))
    model = build_chat_model("groq", "chat", settings, http_async_client=fake.async_client())

    with pytest.raises(errors.ProviderRequestError):
        [chunk async for chunk in model.astream("hi")]


def test_fake_provider_builds_a_scripted_model() -> None:
    s = Settings(_env_file=None, chat_provider="fake", fallback_providers=[])

    assert isinstance(get_chat_model("chat", s), ScriptedChatModel)


def test_paid_provider_cannot_be_built_without_opt_in(settings: Settings) -> None:
    with pytest.raises(errors.ConfigError, match="ALLOW_PAID_PROVIDERS"):
        build_chat_model("openai", "chat", settings)


def test_openai_uses_its_default_endpoint_when_allowed() -> None:
    requests: list[httpx2.Request] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_completion())

    s = Settings(
        _env_file=None,
        allow_paid_providers=True,
        chat_provider="openai",
        fallback_providers=[],
        openai_api_key=SecretStr("sk-test"),
        openai_chat_model="gpt-test",
    )
    get_chat_model("chat", s, http_client=httpx2.Client(transport=httpx2.MockTransport(reply))).invoke("hi")

    assert requests[0].url.host == "api.openai.com"


def _bedrock_settings() -> Settings:
    return Settings(
        _env_file=None,
        allow_paid_providers=True,
        chat_provider="bedrock",
        fallback_providers=[],
        bedrock_chat_model_id="anthropic.test-model",
    )


def _bedrock_client() -> Any:
    return boto3.client(
        "bedrock-runtime", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )


def test_bedrock_converse_contract() -> None:
    client = _bedrock_client()
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "hello from bedrock"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
                "metrics": {"latencyMs": 10},
            },
        )
        reply = get_chat_model("chat", _bedrock_settings(), bedrock_client=client).invoke("hi")

    assert reply.content == "hello from bedrock"


def test_bedrock_throttling_is_a_rate_limit() -> None:
    client = _bedrock_client()
    with Stubber(client) as stub:
        stub.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429)
        with pytest.raises(errors.ProviderRateLimitedError):
            get_chat_model("chat", _bedrock_settings(), bedrock_client=client).invoke("hi")


def test_bedrock_streaming_failures_are_translated() -> None:
    client = _bedrock_client()
    model = build_chat_model("bedrock", "chat", _bedrock_settings(), bedrock_client=client)
    with Stubber(client) as stub:
        # langchain-aws streams only for providers it has verified, and otherwise calls Converse.
        stub.add_client_error("converse", service_error_code="ValidationException", http_status_code=400)
        with pytest.raises(errors.ProviderRequestError):
            list(model.stream("hi"))


def _sse(*pieces: str) -> Reply:
    def chunk(content: str, finish: str | None) -> str:
        delta = {"role": "assistant", "content": content}
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [choice]}
        return f"data: {json.dumps(body)}\n\n"

    events = [chunk(p, "stop" if i == len(pieces) - 1 else None) for i, p in enumerate(pieces)]
    stream = "".join(events) + "data: [DONE]\n\n"
    return lambda request: httpx2.Response(200, text=stream, headers={"content-type": "text/event-stream"})


def test_streaming_yields_chunks(settings: Settings) -> None:
    fake = FakeProviders(groq=_sse("hel", "lo"))
    model = build_chat_model("groq", "chat", settings, http_client=fake.client())

    assert "".join(str(chunk.content) for chunk in model.stream("hi")) == "hello"


async def test_async_streaming_yields_chunks(settings: Settings) -> None:
    fake = FakeProviders(groq=_sse("hel", "lo"))
    model = build_chat_model("groq", "chat", settings, http_async_client=fake.async_client())

    assert "".join([str(chunk.content) async for chunk in model.astream("hi")]) == "hello"


def test_app_errors_from_the_sdk_layer_pass_through(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def over_budget(*args: Any, **kwargs: Any) -> Any:
        raise errors.BudgetExceededError()

    monkeypatch.setattr("langchain_openai.ChatOpenAI._generate", over_budget)

    with pytest.raises(errors.BudgetExceededError) as caught:
        build_chat_model("groq", "chat", settings).invoke("hi")
    assert caught.value.__cause__ is None


def test_bedrock_native_streaming_failures_are_translated() -> None:
    client = _bedrock_client()
    model = build_chat_model("bedrock", "chat", _bedrock_settings(), bedrock_client=client)
    streaming = model.model_copy(update={"disable_streaming": False})
    with Stubber(client) as stub:
        stub.add_client_error("converse_stream", service_error_code="ThrottlingException", http_status_code=429)
        with pytest.raises(errors.ProviderRateLimitedError):
            list(streaming._stream([]))
