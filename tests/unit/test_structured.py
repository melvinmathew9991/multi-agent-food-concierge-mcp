"""Typed output: validation, one repair, then fallback; contract tests over a mock HTTP transport."""

import json
from typing import Any, Literal

import httpx2
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel, SecretStr, ValidationError

from food_concierge import errors
from food_concierge.config import Settings
from food_concierge.models import capabilities
from food_concierge.models.capabilities import ProviderCapabilities
from food_concierge.models.fake import ScriptedChatModel
from food_concierge.models.structured import (
    MAX_PROBLEM_CHARS,
    MAX_REPAIR_ECHO_CHARS,
    _echo,
    _problems,
    get_structured_model,
    structured_runnable,
)
from food_concierge.telemetry import ModelCallRecorder

GROQ = "api.groq.com"
GEMINI = "generativelanguage.googleapis.com"
OLLAMA = "localhost"


class MealRequest(BaseModel):
    diet: Literal["veg", "vegan", "any"]
    max_calories: int | None = None


GOOD_ARGS = {"diet": "vegan", "max_calories": 600}
BAD_ARGS = {"diet": "carnivore", "max_calories": "lots"}


def _tool_reply(args: dict[str, Any]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "MealRequest", "args": args, "id": "call-1"}])


def _scripted(*steps: AIMessage | BaseException) -> ScriptedChatModel:
    return ScriptedChatModel(provider="groq", script=list(steps))


def _last_text(model: ScriptedChatModel) -> str:
    return str(model.calls[-1][-1].content)


# ---------------------------------------------------------------------------
# Repair logic (scripted model)
# ---------------------------------------------------------------------------


def test_valid_output_needs_one_call() -> None:
    model = _scripted(_tool_reply(GOOD_ARGS))

    result = structured_runnable(model, MealRequest, provider="groq").invoke("vegan dinner under 600 kcal")

    assert result == MealRequest(diet="vegan", max_calories=600)
    assert len(model.calls) == 1


def test_invalid_output_is_repaired_once(caplog: pytest.LogCaptureFixture) -> None:
    model = _scripted(_tool_reply(BAD_ARGS), _tool_reply(GOOD_ARGS))
    question = [SystemMessage("Extract the request."), HumanMessage("vegan dinner")]

    result = structured_runnable(model, MealRequest, provider="groq").invoke(question)

    assert result.diet == "vegan"
    assert len(model.calls) == 2
    repair = model.calls[1]
    assert repair[:2] == question  # the original conversation, then the repair note
    note = _last_text(model)
    assert "diet:" in note
    assert "max_calories:" in note
    assert "carnivore" in note  # the model's own reply is echoed so it can fix it
    assert [getattr(r, "attempt", None) for r in caplog.records] == ["first"]


def test_reply_without_structured_part_is_repaired() -> None:
    model = _scripted(AIMessage("Sure! You want vegan food."), _tool_reply(GOOD_ARGS))

    structured_runnable(model, MealRequest, provider="groq").invoke("vegan dinner")

    note = _last_text(model)
    assert "no structured output" in note
    assert "Sure! You want vegan food." in note


def test_failed_repair_raises_a_fallback_error(caplog: pytest.LogCaptureFixture) -> None:
    model = _scripted(_tool_reply(BAD_ARGS), _tool_reply(BAD_ARGS))

    with pytest.raises(errors.ProviderResponseError) as caught:
        structured_runnable(model, MealRequest, provider="groq").invoke("vegan dinner")

    assert caught.value.provider == "groq"
    assert caught.value.falls_back
    assert len(model.calls) == 2
    assert [getattr(r, "attempt", None) for r in caplog.records] == ["first", "repair"]


async def test_async_repair_and_failure() -> None:
    repaired = _scripted(_tool_reply(BAD_ARGS), _tool_reply(GOOD_ARGS))
    first_try = _scripted(_tool_reply(GOOD_ARGS))
    broken = _scripted(_tool_reply(BAD_ARGS), AIMessage("still wrong"))

    assert (await structured_runnable(repaired, MealRequest, provider="groq").ainvoke("x")).diet == "vegan"
    assert (await structured_runnable(first_try, MealRequest, provider="groq").ainvoke("x")).diet == "vegan"
    with pytest.raises(errors.ProviderResponseError):
        await structured_runnable(broken, MealRequest, provider="groq").ainvoke("x")


def test_prompt_values_are_accepted() -> None:
    model = _scripted(_tool_reply(BAD_ARGS), _tool_reply(GOOD_ARGS))
    prompt = ChatPromptValue(messages=[HumanMessage("vegan dinner")])

    structured_runnable(model, MealRequest, provider="groq").invoke(prompt)

    assert model.calls[1][0] == HumanMessage("vegan dinner")


def test_repair_note_is_bounded() -> None:
    with pytest.raises(ValidationError) as caught:
        MealRequest.model_validate({"diet": "x" * 5000, "max_calories": "y"})

    assert len(_problems(caught.value)) <= MAX_PROBLEM_CHARS
    assert "x" * 50 not in _problems(caught.value)  # the values themselves are not repeated
    assert _problems(ValueError("Invalid json output: {")) == "Invalid json output: {"
    assert len(_echo(AIMessage("z" * 10_000))) == MAX_REPAIR_ECHO_CHARS
    assert _echo(None) == ""


def test_a_repair_is_not_counted_as_a_fallback() -> None:
    model = _scripted(_tool_reply(BAD_ARGS), _tool_reply(GOOD_ARGS))
    recorder = ModelCallRecorder()

    structured_runnable(model, MealRequest, provider="groq").invoke("x", config={"callbacks": [recorder]})

    meta = recorder.meta()
    assert meta.attempts == 2
    assert not meta.fallback_used


# ---------------------------------------------------------------------------
# Contract tests over HTTP
# ---------------------------------------------------------------------------


def _completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def tool_call(args: dict[str, Any]) -> dict[str, Any]:
    call = {"id": "call-1", "type": "function", "function": {"name": "MealRequest", "arguments": json.dumps(args)}}
    return _completion({"content": None, "tool_calls": [call]})


def content(text: str) -> dict[str, Any]:
    return _completion({"content": text})


class Providers:
    """Replies per host from a queue, recording every request body."""

    def __init__(self, **replies: list[dict[str, Any] | int]) -> None:
        self.replies = {
            GROQ: replies.get("groq", []),
            GEMINI: replies.get("gemini", []),
            OLLAMA: replies.get("ollama", []),
        }
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self.replies[request.url.host].pop(0)
        if isinstance(reply, int):
            return httpx2.Response(reply, json={"error": {"message": "provider detail"}})
        return httpx2.Response(200, json=reply)

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    def body(self, index: int) -> dict[str, Any]:
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
    )


def test_groq_is_asked_through_a_tool_call(settings: Settings) -> None:
    fake = Providers(groq=[tool_call(GOOD_ARGS)])

    result = get_structured_model(MealRequest, "router", settings, http_client=fake.client()).invoke("vegan")

    assert result == MealRequest(diet="vegan", max_calories=600)
    body = fake.body(0)
    assert body["tools"][0]["function"]["name"] == "MealRequest"
    assert "response_format" not in body
    assert (body["temperature"], body["seed"]) == (0.0, settings.model_seed)


def test_bad_output_is_repaired_then_falls_back(settings: Settings) -> None:
    fake = Providers(groq=[tool_call(BAD_ARGS), tool_call(BAD_ARGS)], gemini=[tool_call(GOOD_ARGS)])

    result = get_structured_model(MealRequest, "router", settings, http_client=fake.client()).invoke("vegan")

    assert result.diet == "vegan"
    assert fake.hosts() == [GROQ, GROQ, GEMINI]
    assert "seed" not in fake.body(2)  # Gemini's endpoint is not sent a seed


async def test_async_fallback(settings: Settings) -> None:
    fake = Providers(groq=[tool_call(BAD_ARGS), content("no")], gemini=[tool_call(GOOD_ARGS)])

    model = get_structured_model(MealRequest, "router", settings, http_async_client=fake.async_client())

    assert (await model.ainvoke("vegan")).diet == "vegan"
    assert fake.hosts() == [GROQ, GROQ, GEMINI]


def test_every_provider_failing_raises_an_app_error(settings: Settings) -> None:
    fake = Providers(groq=[tool_call(BAD_ARGS)] * 2, gemini=[tool_call(BAD_ARGS)] * 2)

    with pytest.raises(errors.ProviderResponseError):
        get_structured_model(MealRequest, "router", settings, http_client=fake.client()).invoke("vegan")

    assert fake.hosts() == [GROQ, GROQ, GEMINI, GEMINI]


def test_rejected_request_does_not_repair_or_fall_back(settings: Settings) -> None:
    fake = Providers(groq=[400])

    with pytest.raises(errors.ProviderRequestError):
        get_structured_model(MealRequest, "router", settings, http_client=fake.client()).invoke("vegan")

    assert fake.hosts() == [GROQ]


def test_ollama_is_asked_for_a_json_schema() -> None:
    settings = Settings(_env_file=None, chat_provider="ollama", fallback_providers=[])
    fake = Providers(ollama=[content("{not json"), content(json.dumps(GOOD_ARGS))])

    result = get_structured_model(MealRequest, "router", settings, http_client=fake.client()).invoke("vegan")

    assert result.diet == "vegan"
    first = fake.body(0)
    assert first["response_format"]["type"] == "json_schema"
    assert "tools" not in first
    assert "{not json" in fake.body(1)["messages"][-1]["content"]


def test_chat_role_keeps_its_temperature_and_no_seed(settings: Settings) -> None:
    fake = Providers(groq=[tool_call(GOOD_ARGS)])

    get_structured_model(MealRequest, "chat", settings, http_client=fake.client()).invoke("vegan")

    assert fake.body(0)["temperature"] == settings.chat_temperature
    assert "seed" not in fake.body(0)


def test_providers_without_tools_or_a_model_are_left_out(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    no_tools = ProviderCapabilities(
        tools=False, structured_method=None, vision=False, streaming=True, usage=True, seed=False
    )
    monkeypatch.setitem(capabilities.CAPABILITIES, "groq", no_tools)
    fake = Providers(gemini=[tool_call(GOOD_ARGS)])

    get_structured_model(MealRequest, "router", settings, http_client=fake.client()).invoke("vegan")

    assert fake.hosts() == [GEMINI]
    with pytest.raises(errors.ConfigError, match="structured vision output"):
        get_structured_model(MealRequest, "vision", settings)
