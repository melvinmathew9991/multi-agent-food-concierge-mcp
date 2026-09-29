from pathlib import Path

import pytest
from pydantic import ValidationError

from food_concierge import errors
from food_concierge.config import PAID_PROVIDERS, REPO_ROOT, KeyedProvider, Settings, get_settings


@pytest.fixture
def settings() -> Settings:
    # _env_file=None isolates tests from a developer's .env file.
    return Settings(_env_file=None)


def test_default_config_has_no_paid_provider(settings: Settings) -> None:
    configured = {settings.chat_provider, settings.embed_provider, *settings.fallback_providers}

    assert not configured & PAID_PROVIDERS
    assert settings.allow_paid_providers is False
    assert (settings.chat_provider, settings.fallback_providers) == ("groq", ["gemini"])


def test_paid_provider_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_PROVIDERS", "gemini,bedrock")
    with pytest.raises(ValidationError, match="ALLOW_PAID_PROVIDERS"):
        Settings(_env_file=None)

    monkeypatch.setenv("ALLOW_PAID_PROVIDERS", "true")
    assert Settings(_env_file=None).fallback_providers == ["gemini", "bedrock"]


def test_csv_env_vars_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_PROVIDERS", "gemini, ollama")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,https://b.example")

    s = Settings(_env_file=None)

    assert s.fallback_providers == ["gemini", "ollama"]
    assert s.cors_origins == ["https://a.example", "https://b.example"]


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_PROVIDER", "not-a-provider")
    with pytest.raises(ValidationError, match="chat_provider"):
        Settings(_env_file=None)


def test_env_example_matches_settings() -> None:
    example = REPO_ROOT / ".env.example"
    keys = [line.split("=", 1)[0] for line in example.read_text().splitlines() if line and not line.startswith("#")]

    assert keys, ".env.example has no variables"
    assert not {k.lower() for k in keys} - set(Settings.model_fields)
    # A copied example, with its blank values, must load and keep the zero-cost defaults.
    s = Settings(_env_file=example)
    assert s.aws_region == "us-east-1"
    assert s.groq_api_key is None


def test_unknown_env_file_key_is_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CHAT_PROVDER=ollama\n")

    with pytest.raises(ValidationError, match="chat_provder"):
        Settings(_env_file=env_file)


def test_daily_limits_apply_only_to_paid_providers(settings: Settings) -> None:
    assert settings.daily_call_limit("bedrock") == 200
    assert settings.daily_call_limit("ollama") is None


@pytest.mark.parametrize("provider", ["groq", "gemini", "openai"])
def test_missing_api_key_raises_config_error(settings: Settings, provider: KeyedProvider) -> None:
    with pytest.raises(errors.ConfigError, match=f"{provider.upper()}_API_KEY"):
        settings.require_api_key(provider)


def test_secret_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-secret")
    s = Settings(_env_file=None)

    assert "gsk-test-secret" not in repr(s)
    assert s.require_api_key("groq") == "gsk-test-secret"


def test_model_id_resolution(settings: Settings) -> None:
    assert settings.require_model_id("ollama", "vision") == settings.ollama_vision_model
    with pytest.raises(errors.ConfigError, match="bedrock"):
        settings.require_model_id("bedrock", "chat")


def test_derived_paths_and_limits(settings: Settings) -> None:
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.processed_dir == settings.data_dir / "processed"
    assert settings.max_image_bytes == 5 * 1024 * 1024


def test_get_settings_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no developer .env in the working directory

    assert get_settings() is get_settings()


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
