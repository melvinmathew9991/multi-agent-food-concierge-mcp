"""Chat model router: one model per role, with a provider fallback chain.

Groq, Gemini and Ollama are reached through their OpenAI-compatible endpoints
with ``ChatOpenAI``; Bedrock through ``ChatBedrockConverse`` and only when paid
providers are allowed. Each provider model is wrapped so that it

- raises application errors instead of SDK exceptions, which is what the
  fallback chain matches on (``FALLBACK_ERRORS``); a rejected request stops it;
- treats an empty reply as a failed attempt, because thinking models can spend
  the whole output budget on hidden reasoning and still return "success".

The chain itself is LangChain's ``with_fallbacks``; there is no retry loop here.
The next provider is the retry, so SDK retries apply only to the last provider.
Each attempt has its provider's timeout (``Settings.timeout_for``), and settings
validation keeps the whole chain inside ``request_deadline_s``. It is a per-attempt network timeout,
so the API layer still enforces the deadline as a wall-clock limit.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from typing import Any

import httpx2
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel, LangSmithParams, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from food_concierge.config import PAID_PROVIDERS, ModelRole, ProviderName, Settings, get_settings
from food_concierge.errors import AppError, ConfigError, ProviderError, ProviderResponseError
from food_concierge.models.fake import ScriptedChatModel
from food_concierge.models.provider_errors import FALLBACK_ERRORS, translate_provider_error

logger = logging.getLogger(__name__)

# Providers whose OpenAI-compatible endpoint accepts ``reasoning_effort``. Ollama's local models do not reason.
_ACCEPTS_REASONING_EFFORT: frozenset[str] = frozenset({"groq", "gemini", "openai"})


@contextmanager
def _translated(provider: str) -> Generator[None]:
    try:
        yield
    except AppError as app_error:
        _log_failure(app_error, provider)
        raise
    except Exception as exc:
        translated = translate_provider_error(exc, provider)
        _log_failure(translated, provider)
        raise translated from exc


def _log_failure(err: AppError, provider: str) -> None:
    needs_attention = isinstance(err, ProviderError) and err.needs_attention
    falls_back = isinstance(err, FALLBACK_ERRORS)
    logger.log(
        logging.ERROR if needs_attention else logging.WARNING,
        "model call failed",
        extra={"provider": provider, "error_code": err.code, "falls_back": falls_back},
    )


def _require_output(result: ChatResult, provider: str) -> ChatResult:
    for generation in result.generations:
        message = generation.message
        if message.content or (isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls)):
            return result
    err = ProviderResponseError("The model returned an empty reply.", provider=provider)
    _log_failure(err, provider)
    raise err


class GuardedChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` for any OpenAI-compatible endpoint, raising application errors."""

    provider_label: str = "openai"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        # Traces and ModelCallRecorder name the real provider, not "openai" for every compatible endpoint.
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = self.provider_label
        return params

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        with _translated(self.provider_label):
            result = super()._generate(*args, **kwargs)
        return _require_output(result, self.provider_label)

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        with _translated(self.provider_label):
            result = await super()._agenerate(*args, **kwargs)
        return _require_output(result, self.provider_label)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        with _translated(self.provider_label):
            yield from super()._stream(*args, **kwargs)

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        with _translated(self.provider_label):
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk


class GuardedChatBedrockConverse(ChatBedrockConverse):
    """``ChatBedrockConverse`` raising application errors (async runs these in an executor)."""

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "bedrock"
        return params

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        with _translated("bedrock"):
            result = super()._generate(*args, **kwargs)
        return _require_output(result, "bedrock")

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        with _translated("bedrock"):
            yield from super()._stream(*args, **kwargs)


def build_chat_model(
    provider: ProviderName,
    role: ModelRole,
    settings: Settings,
    *,
    is_last: bool = True,
    http_client: httpx2.Client | None = None,
    http_async_client: httpx2.AsyncClient | None = None,
    bedrock_client: Any = None,
) -> BaseChatModel:
    """One provider's model for ``role``. HTTP clients are injectable for tests and connection limits."""
    if provider in PAID_PROVIDERS and not settings.allow_paid_providers:
        # Settings validation already refuses this; a direct call must not bypass it.
        raise ConfigError(f"Provider '{provider}' is paid and ALLOW_PAID_PROVIDERS is false.")
    model = settings.require_model_id(provider, role)
    if provider == "fake":
        return ScriptedChatModel(model_name=model)

    retries = settings.provider_max_retries if is_last else 0
    timeout = settings.timeout_for(provider)
    temperature = settings.chat_temperature if role == "chat" else 0.0
    if provider == "bedrock":
        return GuardedChatBedrockConverse(
            model_id=model,
            region_name=settings.aws_region,
            max_tokens=settings.max_output_tokens,
            temperature=temperature,
            client=bedrock_client,
            config=BotoConfig(
                connect_timeout=timeout,
                read_timeout=timeout,
                retries={"max_attempts": retries + 1, "mode": "standard"},
            ),
        )

    endpoints: dict[str, tuple[str | None, str]] = {
        "groq": (settings.groq_base_url, settings.require_api_key("groq") if provider == "groq" else ""),
        "gemini": (settings.gemini_base_url, settings.require_api_key("gemini") if provider == "gemini" else ""),
        # Ollama ignores the key, but the SDK requires one.
        "ollama": (f"{settings.ollama_base_url.rstrip('/')}/v1", "ollama"),
        "openai": (None, settings.require_api_key("openai") if provider == "openai" else ""),
    }
    base_url, api_key = endpoints[provider]
    effort = settings.reasoning_effort(role) if provider in _ACCEPTS_REASONING_EFFORT else None
    return GuardedChatOpenAI(
        provider_label=provider,
        model_name=model,
        openai_api_key=SecretStr(api_key),
        openai_api_base=base_url,
        request_timeout=timeout,
        max_retries=retries,
        max_tokens=settings.max_output_tokens,
        temperature=temperature,
        reasoning_effort=effort,
        # These endpoints implement Chat Completions only; never switch to the Responses API.
        use_responses_api=False,
        http_client=http_client,
        http_async_client=http_async_client,
    )


def get_chat_model(
    role: ModelRole,
    settings: Settings | None = None,
    *,
    http_client: httpx2.Client | None = None,
    http_async_client: httpx2.AsyncClient | None = None,
    bedrock_client: Any = None,
) -> Runnable[LanguageModelInput, BaseMessage]:
    """The model for ``role``: the configured chat provider, falling back along ``FALLBACK_PROVIDERS``.

    Providers with no model for this role (e.g. no Groq vision model) are left out of the chain.
    ``bind_tools`` and ``with_structured_output`` on the result apply to every provider in it.
    When every provider fails, the first provider's error is raised.
    """
    settings = settings or get_settings()
    chain = [p for p in settings.provider_chain if _has_model(settings, p, role)]
    if not chain:
        raise ConfigError(f"No provider in {settings.provider_chain} has a {role} model configured.")
    skipped = [p for p in settings.provider_chain if p not in chain]
    if skipped:
        logger.warning("providers skipped for role", extra={"role": role, "providers": ",".join(skipped)})

    models = [
        build_chat_model(
            provider,
            role,
            settings,
            is_last=index == len(chain) - 1,
            http_client=http_client,
            http_async_client=http_async_client,
            bedrock_client=bedrock_client,
        )
        for index, provider in enumerate(chain)
    ]
    if len(models) == 1:
        return models[0]
    return models[0].with_fallbacks(models[1:], exceptions_to_handle=FALLBACK_ERRORS)


def _has_model(settings: Settings, provider: ProviderName, role: ModelRole) -> bool:
    try:
        settings.require_model_id(provider, role)
    except ConfigError:
        return False
    return True
