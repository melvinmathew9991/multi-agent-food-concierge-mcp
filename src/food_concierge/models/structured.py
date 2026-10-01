"""Typed model output: validated pydantic objects, one repair, then the next provider.

Each provider is asked in the way its endpoint supports best (``capabilities.py``).
When a reply does not validate, or has no structured part at all, the same
provider gets one repair turn: the original messages, its own reply and the
validation problems. If that fails too, the attempt raises
``ProviderResponseError``, which moves the fallback chain on. When every
provider fails, the caller gets that error and answers safely; malformed
model output never crashes a request (baseline defect: the original crashed
on bad JSON).

One repair, not a loop: each repair is another model call inside the same
request deadline, which the API layer enforces as a wall-clock limit.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

import httpx2
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel, ValidationError

from food_concierge.config import ModelRole, Settings, get_settings
from food_concierge.errors import ConfigError, ProviderResponseError
from food_concierge.models.capabilities import CAPABILITIES, StructuredMethod
from food_concierge.models.provider_errors import FALLBACK_ERRORS
from food_concierge.models.router import build_chat_model, has_model

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Enough of the bad reply and of the problems for the model to fix them, without echoing a whole essay back.
MAX_REPAIR_ECHO_CHARS = 2000
MAX_PROBLEM_CHARS = 600

REPAIR_INSTRUCTION = (
    "Your previous reply did not match the required format.\n"
    "Problems: {problems}\n"
    "Your previous reply: {reply}\n"
    "Answer the original request again, in the required format only."
)


def _as_messages(value: LanguageModelInput) -> list[BaseMessage]:
    if isinstance(value, PromptValue):
        return value.to_messages()
    if isinstance(value, str):
        return [HumanMessage(value)]
    return convert_to_messages(value)


def _problems(error: BaseException | None) -> str:
    if error is None:
        return "the reply had no structured output"
    if isinstance(error, ValidationError):
        # Field paths and messages only: the offending values are already in the echoed reply.
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'reply'}: {item['msg']}"
            for item in error.errors(include_input=False, include_url=False)
        )
    else:
        details = str(error)
    return details[:MAX_PROBLEM_CHARS]


def _echo(raw: Any) -> str:
    if not isinstance(raw, BaseMessage):
        return ""
    tool_calls = getattr(raw, "tool_calls", None) or getattr(raw, "invalid_tool_calls", None)
    text = str(tool_calls) if tool_calls else raw.text
    return text[:MAX_REPAIR_ECHO_CHARS]


def structured_runnable(
    model: BaseChatModel,
    schema: type[SchemaT],
    *,
    provider: str,
    method: StructuredMethod | None = None,
) -> Runnable[LanguageModelInput, SchemaT]:
    """``model`` returning validated ``schema`` objects, with one repair turn before giving up.

    The model is given the schema as plain JSON Schema and its reply is validated here. Given the pydantic
    class, some integrations validate inside the model call (``ChatOpenAI`` with ``json_schema``), so a
    malformed reply would skip the repair turn; validating in one place gives every provider the same path.
    """
    options: dict[str, Any] = {"include_raw": True}
    if method is not None:
        options["method"] = method
    # With include_raw the reply is a dict: {"raw", "parsed", "parsing_error"}.
    bound: Runnable[LanguageModelInput, Any] = model.with_structured_output(schema.model_json_schema(), **options)

    def validated(result: Any, attempt: str) -> tuple[SchemaT | None, BaseException | None]:
        error: BaseException | None = result.get("parsing_error")
        parsed = result.get("parsed")
        if error is None and parsed is not None:
            try:
                return schema.model_validate(parsed), None
            except ValidationError as invalid:
                error = invalid
        logger.warning(
            "structured output invalid",
            extra={"provider": provider, "attempt": attempt, "error_type": type(error).__name__ if error else "none"},
        )
        return None, error

    def repair_messages(messages: list[BaseMessage], raw: Any, error: BaseException | None) -> list[BaseMessage]:
        return [*messages, HumanMessage(REPAIR_INSTRUCTION.format(problems=_problems(error), reply=_echo(raw)))]

    def failed() -> ProviderResponseError:
        return ProviderResponseError("The model's reply did not match the expected format.", provider=provider)

    def invoke(value: LanguageModelInput, config: RunnableConfig) -> SchemaT:
        messages = _as_messages(value)
        first = bound.invoke(messages, config=config)
        parsed, error = validated(first, "first")
        if parsed is not None:
            return parsed
        second = bound.invoke(repair_messages(messages, first.get("raw"), error), config=config)
        repaired, _ = validated(second, "repair")
        if repaired is None:
            raise failed()
        return repaired

    async def ainvoke(value: LanguageModelInput, config: RunnableConfig) -> SchemaT:
        messages = _as_messages(value)
        first = await bound.ainvoke(messages, config=config)
        parsed, error = validated(first, "first")
        if parsed is not None:
            return parsed
        second = await bound.ainvoke(repair_messages(messages, first.get("raw"), error), config=config)
        repaired, _ = validated(second, "repair")
        if repaired is None:
            raise failed()
        return repaired

    return RunnableLambda(invoke, afunc=ainvoke, name=f"structured_{provider}")


def get_structured_model(
    schema: type[SchemaT],
    role: ModelRole,
    settings: Settings | None = None,
    *,
    http_client: httpx2.Client | None = None,
    http_async_client: httpx2.AsyncClient | None = None,
    bedrock_client: Any = None,
) -> Runnable[LanguageModelInput, SchemaT]:
    """Validated ``schema`` objects for ``role``, along the configured fallback chain.

    Raises ``ProviderResponseError`` (or the first provider's error) when no provider produces valid output.
    """
    settings = settings or get_settings()
    chain = [p for p in settings.provider_chain if has_model(settings, p, role) and CAPABILITIES[p].tools]
    if not chain:
        raise ConfigError(f"No provider in {settings.provider_chain} can produce structured {role} output.")
    runnables: Sequence[Runnable[LanguageModelInput, SchemaT]] = [
        structured_runnable(
            build_chat_model(
                provider,
                role,
                settings,
                is_last=index == len(chain) - 1,
                http_client=http_client,
                http_async_client=http_async_client,
                bedrock_client=bedrock_client,
            ),
            schema,
            provider=provider,
            method=CAPABILITIES[provider].structured_method,
        )
        for index, provider in enumerate(chain)
    ]
    if len(runnables) == 1:
        return runnables[0]
    return runnables[0].with_fallbacks(runnables[1:], exceptions_to_handle=FALLBACK_ERRORS)
