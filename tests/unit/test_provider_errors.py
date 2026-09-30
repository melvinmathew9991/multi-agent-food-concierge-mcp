import json

import botocore.exceptions as boto_errors
import httpx
import httpx2
import openai
import pytest
from pydantic import BaseModel, ValidationError

from food_concierge import errors
from food_concierge.models.provider_errors import translate_provider_error

# openai>=3 is built on httpx2, so its exceptions carry httpx2 requests and responses.
_REQUEST = httpx2.Request("POST", "https://provider.test/v1/chat/completions")


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    return cls("provider detail", response=httpx2.Response(status, request=_REQUEST), body=None)


def _bedrock_error(code: str) -> boto_errors.ClientError:
    return boto_errors.ClientError({"Error": {"Code": code, "Message": "provider detail"}}, "Converse")


def _validation_error() -> ValidationError:
    class Route(BaseModel):
        target: str

    try:
        Route.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("validation should have failed")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(openai.AuthenticationError, 401), errors.ProviderAuthError),
        (_status_error(openai.PermissionDeniedError, 403), errors.ProviderAuthError),
        (_status_error(openai.RateLimitError, 429), errors.ProviderRateLimitedError),
        (_status_error(openai.NotFoundError, 404), errors.ProviderModelNotFoundError),
        (_status_error(openai.BadRequestError, 400), errors.ProviderRequestError),
        (_status_error(openai.UnprocessableEntityError, 422), errors.ProviderRequestError),
        (_status_error(openai.APIStatusError, 408), errors.ProviderTimeoutError),
        (_status_error(openai.InternalServerError, 500), errors.ProviderUnavailableError),
        (_status_error(openai.APIStatusError, 503), errors.ProviderUnavailableError),
        (_status_error(openai.ConflictError, 409), errors.ProviderError),
        (openai.APITimeoutError(request=_REQUEST), errors.ProviderTimeoutError),
        (openai.APIConnectionError(request=_REQUEST), errors.ProviderUnavailableError),
        (httpx.ReadTimeout("slow"), errors.ProviderTimeoutError),
        (httpx.ConnectError("refused"), errors.ProviderUnavailableError),
        (httpx2.ReadTimeout("slow"), errors.ProviderTimeoutError),
        (httpx2.ConnectError("refused"), errors.ProviderUnavailableError),
        (TimeoutError(), errors.ProviderTimeoutError),
        (ConnectionRefusedError(), errors.ProviderUnavailableError),
        (json.JSONDecodeError("bad", "{", 0), errors.ProviderResponseError),
        (_validation_error(), errors.ProviderResponseError),
        (_bedrock_error("ThrottlingException"), errors.ProviderRateLimitedError),
        (_bedrock_error("AccessDeniedException"), errors.ProviderAuthError),
        (_bedrock_error("ValidationException"), errors.ProviderRequestError),
        (_bedrock_error("ModelTimeoutException"), errors.ProviderTimeoutError),
        (_bedrock_error("ResourceNotFoundException"), errors.ProviderModelNotFoundError),
        (_bedrock_error("ServiceUnavailableException"), errors.ProviderUnavailableError),
        (_bedrock_error("SomethingNew"), errors.ProviderError),
        (boto_errors.ReadTimeoutError(endpoint_url="https://bedrock.test"), errors.ProviderTimeoutError),
        (boto_errors.EndpointConnectionError(endpoint_url="https://bedrock.test"), errors.ProviderUnavailableError),
        (boto_errors.NoCredentialsError(), errors.ProviderAuthError),
        (RuntimeError("unexpected"), errors.ProviderError),
    ],
)
def test_provider_failures_map_to_app_errors(exc: BaseException, expected: type[errors.ProviderError]) -> None:
    err = translate_provider_error(exc, provider="groq")

    assert type(err) is expected
    assert isinstance(err, errors.ProviderError)
    assert err.provider == "groq"


def test_messages_never_carry_provider_detail() -> None:
    err = translate_provider_error(_status_error(openai.RateLimitError, 429), provider="groq")

    assert "provider detail" not in err.message
    assert err.message == errors.ProviderRateLimitedError.default_message


def test_app_errors_pass_through_unchanged() -> None:
    original = errors.BudgetExceededError()

    assert translate_provider_error(original, provider="groq") is original


def test_only_request_errors_stop_the_fallback_chain() -> None:
    # A rejected request would be rejected by the next provider too; every other failure may succeed elsewhere.
    stops = [cls for cls in _provider_error_classes() if not cls.falls_back]

    assert stops == [errors.ProviderRequestError]


def test_operator_problems_are_flagged() -> None:
    flagged = {cls for cls in _provider_error_classes() if cls.needs_attention}

    assert flagged == {errors.ProviderAuthError, errors.ProviderModelNotFoundError}


def _provider_error_classes() -> list[type[errors.ProviderError]]:
    found: list[type[errors.ProviderError]] = [errors.ProviderError]
    for cls in found:
        found.extend(cls.__subclasses__())
    return found
