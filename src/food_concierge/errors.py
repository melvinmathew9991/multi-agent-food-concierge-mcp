"""Application error hierarchy.

Every error carries a stable ``code`` and the HTTP status the API maps it to, so
the API layer can translate errors generically instead of per-type branching.
Messages on these errors are safe to show to users; provider details belong in
the chained ``__cause__`` and the logs.
"""

from __future__ import annotations


class AppError(Exception):
    code: str = "internal_error"
    http_status: int = 500
    default_message: str = "Something went wrong on our side."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)

    @property
    def message(self) -> str:
        return str(self)


class ConfigError(AppError):
    code = "config_error"
    default_message = "The service is misconfigured."


class InputValidationError(AppError):
    code = "invalid_input"
    http_status = 422
    default_message = "The request is invalid."


class NotReadyError(AppError):
    code = "not_ready"
    http_status = 503
    default_message = "The search index is not ready yet."


class BudgetExceededError(AppError):
    code = "budget_exceeded"
    http_status = 429
    default_message = "Today's usage limit for this model has been reached."


class RetrievalError(AppError):
    code = "retrieval_error"
    default_message = "Search failed."


class ProviderError(AppError):
    code = "provider_error"
    http_status = 502
    default_message = "The model service returned an error."

    def __init__(self, message: str | None = None, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class ProviderAuthError(ProviderError):
    code = "provider_auth_error"
    default_message = "The model service rejected our credentials."


class ProviderRateLimitedError(ProviderError):
    code = "provider_rate_limited"
    http_status = 429
    default_message = "The model service is busy. Please retry shortly."


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    http_status = 504
    default_message = "The model service took too long to respond."


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"
    http_status = 503
    default_message = "The model service is unavailable."


class ProviderResponseError(ProviderError):
    code = "provider_bad_response"
    default_message = "The model returned an unusable response."
