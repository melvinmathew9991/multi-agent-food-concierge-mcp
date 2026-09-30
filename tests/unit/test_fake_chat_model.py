import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from food_concierge import errors
from food_concierge.models.fake import ScriptedChatModel, ScriptExhaustedError


@tool
def search_dishes(query: str) -> str:
    """Search the catalog."""
    return query


class Route(BaseModel):
    target: str


def test_replays_the_script_in_order_and_records_calls() -> None:
    model = ScriptedChatModel(script=[AIMessage(content="first"), AIMessage(content="second")])

    assert model.invoke("hello").content == "first"
    assert model.invoke([HumanMessage(content="again")]).content == "second"
    assert [call[0].content for call in model.calls] == ["hello", "again"]
    assert model.remaining == 0


def test_scripted_exception_is_raised() -> None:
    model = ScriptedChatModel(script=[errors.ProviderRateLimitedError(provider="groq"), AIMessage(content="ok")])

    with pytest.raises(errors.ProviderRateLimitedError):
        model.invoke("hello")
    assert model.invoke("hello").content == "ok"


def test_calling_past_the_script_fails_the_test() -> None:
    model = ScriptedChatModel(script=[])

    with pytest.raises(ScriptExhaustedError, match="called 1 times"):
        model.invoke("hello")


def test_tool_calls_survive_binding() -> None:
    call = {"name": "search_dishes", "args": {"query": "vegan curry"}, "id": "call-1"}
    model = ScriptedChatModel(script=[AIMessage(content="", tool_calls=[call])])

    reply = model.bind_tools([search_dishes]).invoke("find me a vegan curry")

    assert isinstance(reply, AIMessage)
    assert reply.tool_calls[0]["args"] == {"query": "vegan curry"}
    assert model.bound_tools == [search_dishes]


def test_structured_output_parses_a_scripted_tool_call() -> None:
    call = {"name": "Route", "args": {"target": "recommender"}, "id": "call-1"}
    model = ScriptedChatModel(script=[AIMessage(content="", tool_calls=[call])])

    assert model.with_structured_output(Route).invoke("route this") == Route(target="recommender")


def test_replayed_messages_are_copies() -> None:
    scripted = AIMessage(content="same")
    model = ScriptedChatModel(script=[scripted])

    assert model.invoke("hello") is not scripted


async def test_async_invocation_uses_the_same_script() -> None:
    model = ScriptedChatModel(script=[AIMessage(content="async ok")])

    assert (await model.ainvoke("hello")).content == "async ok"
