"""Translate provider SDK exceptions into the application error hierarchy.

One place decides what a provider failure means: whether the next provider in
the fallback chain may succeed (``ProviderError.falls_back``) and whether an
operator must act (``ProviderError.needs_attention``). Messages are the safe
defaults on the error classes; provider details stay on ``__cause__`` for logs.
"""

from __future__ import annotations

import json

import botocore.exceptions as boto_errors
import httpx
import httpx2
import openai
from pydantic import ValidationError

from food_concierge.errors import (
    AppError,
    ProviderAuthError,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderRateLimitedError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

# Bedrock reports failures as ClientError codes rather than exception types.
_BEDROCK_CODES: dict[str, type[ProviderError]] = {
    "ThrottlingException": ProviderRateLimitedError,
    "ServiceQuotaExceededException": ProviderRateLimitedError,
    "AccessDeniedException": ProviderAuthError,
    "UnrecognizedClientException": ProviderAuthError,
    "ExpiredTokenException": ProviderAuthError,
    "ValidationException": ProviderRequestError,
    "ModelTimeoutException": ProviderTimeoutError,
    "ResourceNotFoundException": ProviderModelNotFoundError,
    "ModelNotReadyException": ProviderUnavailableError,
    "ServiceUnavailableException": ProviderUnavailableError,
    "InternalServerException": ProviderUnavailableError,
}


def _from_openai_status(exc: openai.APIStatusError) -> type[ProviderError]:
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return ProviderAuthError
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitedError
    if isinstance(exc, openai.NotFoundError):
        return ProviderModelNotFoundError
    if isinstance(exc, openai.BadRequestError | openai.UnprocessableEntityError):
        return ProviderRequestError
    if exc.status_code == 408:
        return ProviderTimeoutError
    if exc.status_code >= 500:
        return ProviderUnavailableError
    return ProviderError


def _error_class(exc: BaseException) -> type[ProviderError]:
    # Order matters: APITimeoutError subclasses APIConnectionError.
    if isinstance(exc, openai.APITimeoutError | httpx.TimeoutException | httpx2.TimeoutException | TimeoutError):
        return ProviderTimeoutError
    if isinstance(exc, openai.APIConnectionError | httpx.TransportError | httpx2.TransportError | ConnectionError):
        return ProviderUnavailableError
    if isinstance(exc, openai.APIStatusError):
        return _from_openai_status(exc)
    if isinstance(
        exc,
        openai.LengthFinishReasonError
        | openai.ContentFilterFinishReasonError
        | openai.APIResponseValidationError
        | ValidationError
        | json.JSONDecodeError,
    ):
        return ProviderResponseError
    if isinstance(exc, boto_errors.ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return _BEDROCK_CODES.get(code, ProviderError)
    if isinstance(exc, boto_errors.ReadTimeoutError | boto_errors.ConnectTimeoutError):
        return ProviderTimeoutError
    if isinstance(exc, boto_errors.EndpointConnectionError | boto_errors.ConnectionClosedError):
        return ProviderUnavailableError
    if isinstance(exc, boto_errors.NoCredentialsError):
        return ProviderAuthError
    return ProviderError


def translate_provider_error(exc: BaseException, provider: str) -> AppError:
    """Map ``exc`` to an application error; app errors pass through unchanged.

    Callers raise the result ``from exc`` so the provider detail stays on the chain.
    """
    if isinstance(exc, AppError):
        return exc
    return _error_class(exc)(provider=provider)
