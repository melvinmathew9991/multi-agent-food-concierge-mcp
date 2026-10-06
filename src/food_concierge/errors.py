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


class RetrievalError(AppError):
    code = "retrieval_error"
    default_message = "Search failed."


class ProviderError(AppError):
    code = "provider_error"
    http_status = 502
    default_message = "The model service returned an error."
    # falls_back: another provider may succeed where this one failed.
    # needs_attention: an operator must fix something (credentials, a retired model), so it is logged at ERROR.
    falls_back: bool = True
    needs_attention: bool = False

    def __init__(self, message: str | None = None, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class ProviderAuthError(ProviderError):
    code = "provider_auth_error"
    default_message = "The model service rejected our credentials."
    needs_attention = True


class ProviderRequestError(ProviderError):
    # The request itself was rejected (bad parameters, context too long); the next provider would reject it too.
    code = "provider_bad_request"
    default_message = "The model service could not handle this request."
    falls_back = False


class ProviderRateLimitedError(ProviderError):
    code = "provider_rate_limited"
    http_status = 429
    default_message = "The model service is busy. Please retry shortly."


class BudgetExceededError(ProviderError):
    # Our own cap on a provider (models/usage.py), reached before its quota is: another provider's quota is separate.
    code = "budget_exceeded"
    http_status = 429
    default_message = "Today's usage limit for this model has been reached."


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    http_status = 504
    default_message = "The model service took too long to respond."


class DeadlineExceededError(ProviderError):
    # Not a ProviderTimeoutError: the fallback chain matches by type, and with the deadline spent
    # no later provider can start either.
    code = "deadline_exceeded"
    http_status = 504
    default_message = "The request took too long."
    falls_back = False


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"
    http_status = 503
    default_message = "The model service is unavailable."


class ProviderModelNotFoundError(ProviderUnavailableError):
    # Free-tier models are retired or closed to new accounts without notice; the configured name needs updating.
    code = "provider_model_unavailable"
    default_message = "The configured model is not available."
    needs_attention = True


class ProviderResponseError(ProviderError):
    code = "provider_bad_response"
    default_message = "The model returned an unusable response."
