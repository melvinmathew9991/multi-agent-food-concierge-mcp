"""Scripted chat model for offline tests.

Agent and router tests need exact control over what the model "says": plain
answers, tool calls, structured output and failures, in a fixed order.
LangChain's fake chat models cannot bind tools or raise, so this one replays a
script and records every call for assertions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LangSmithParams, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr, SkipValidation

ScriptStep = AIMessage | BaseException


class ScriptExhaustedError(AssertionError):
    """The code under test called the model more often than the test scripted."""


class ScriptedChatModel(BaseChatModel):
    """Replays ``script`` one step per call: an ``AIMessage`` is returned, an exception is raised."""

    # Validation would coerce scripted exceptions into messages; the script is test input, used as given.
    script: Annotated[list[ScriptStep], SkipValidation] = Field(default_factory=list)
    model_name: str = "fake-chat"
    provider: str = "fake"  # reported to callbacks as ``ls_provider``, so fallback tests can tell models apart

    _position: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = self.provider
        params["ls_model_name"] = self.model_name
        return params

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """The messages received by each call, in order."""
        return self._calls

    @property
    def bound_tools(self) -> list[Any]:
        return self._bound_tools

    @property
    def remaining(self) -> int:
        return len(self.script) - self._position

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        if self._position >= len(self.script):
            raise ScriptExhaustedError(
                f"Model called {len(self._calls)} times; the script has {len(self.script)} steps."
            )
        step = self.script[self._position]
        self._position += 1
        if isinstance(step, BaseException):
            raise step
        return ChatResult(generations=[ChatGeneration(message=step.model_copy())])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The script already decides which tool calls happen; binding only records what was offered.
        self._bound_tools.extend(tools)
        return self
