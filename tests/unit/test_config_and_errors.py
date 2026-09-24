import pytest

from food_concierge import errors
from food_concierge.config import Settings


@pytest.fixture
def settings() -> Settings:
    # _env_file=None isolates tests from a developer's .env file.
    return Settings(_env_file=None)


def test_defaults_are_free_providers(settings: Settings) -> None:
    assert settings.chat_provider == "ollama"
    assert settings.embed_provider == "fastembed"
    assert settings.enable_admin is False


def test_csv_env_vars_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_CHAT_PROVIDERS", "bedrock, openai")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,https://b.example")

    s = Settings(_env_file=None)

    assert s.allowed_chat_providers == ["bedrock", "openai"]
    assert s.cors_origins == ["https://a.example", "https://b.example"]


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="chat_provider"):
        Settings(_env_file=None)


def test_daily_limits_apply_only_to_paid_providers(settings: Settings) -> None:
    assert settings.daily_call_limit("bedrock") == 200
    assert settings.daily_call_limit("ollama") is None


def test_missing_openai_key_raises_config_error(settings: Settings) -> None:
    with pytest.raises(errors.ConfigError, match="OPENAI_API_KEY"):
        settings.require_openai_key()


def test_secret_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    s = Settings(_env_file=None)

    assert "sk-test-secret" not in repr(s)
    assert s.require_openai_key() == "sk-test-secret"


def test_model_id_resolution(settings: Settings) -> None:
    assert settings.require_model_id("ollama", "vision") == settings.ollama_vision_model
    with pytest.raises(errors.ConfigError, match="bedrock"):
        settings.require_model_id("bedrock", "chat")


@pytest.mark.parametrize(
    ("error_cls", "status"),
    [
        (errors.InputValidationError, 422),
        (errors.NotReadyError, 503),
        (errors.BudgetExceededError, 429),
        (errors.ProviderRateLimitedError, 429),
        (errors.ProviderTimeoutError, 504),
        (errors.ProviderAuthError, 502),
        (errors.ProviderResponseError, 502),
    ],
)
def test_error_http_mapping(error_cls: type[errors.AppError], status: int) -> None:
    err = error_cls()
    assert err.http_status == status
    assert isinstance(err, errors.AppError)
    assert err.message


def test_provider_error_keeps_provider_name() -> None:
    err = errors.ProviderTimeoutError(provider="bedrock")
    assert err.provider == "bedrock"
    assert isinstance(err, errors.ProviderError)
