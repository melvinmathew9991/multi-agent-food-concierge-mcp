from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from food_concierge import errors
from food_concierge.config import PAID_PROVIDERS, REPO_ROOT, KeyedProvider, Settings, TraceContent, get_settings


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
    monkeypatch.setenv("REQUEST_DEADLINE_S", "12")  # three providers must fit the deadline
    with pytest.raises(ValidationError, match="ALLOW_PAID_PROVIDERS"):
        Settings(_env_file=None)

    monkeypatch.setenv("ALLOW_PAID_PROVIDERS", "true")
    assert Settings(_env_file=None).fallback_providers == ["gemini", "bedrock"]


def test_csv_env_vars_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_PROVIDERS", "gemini, ollama")
    monkeypatch.setenv("REQUEST_DEADLINE_S", "12")
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


def test_call_limits_cover_hosted_providers_only(settings: Settings) -> None:
    assert settings.daily_call_limit("bedrock") == 200
    assert settings.daily_call_limit("groq") == settings.daily_call_limit_groq
    assert settings.minute_call_limit("gemini") == settings.minute_call_limit_gemini
    assert settings.daily_call_limit("ollama") is None
    assert settings.minute_call_limit("ollama") is None


def test_fallback_chain_must_fit_the_request_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_PROVIDERS", "gemini,ollama")  # 3 s + 5 s + 4 s > 8 s

    with pytest.raises(ValidationError, match="add up to 12 s"):
        Settings(_env_file=None)

    monkeypatch.setenv("REQUEST_DEADLINE_S", "12")
    assert Settings(_env_file=None).provider_chain == ["groq", "gemini", "ollama"]


def test_retries_count_against_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_MAX_RETRIES", "1")  # the last provider (Gemini, 5 s) may try twice

    with pytest.raises(ValidationError, match="add up to 13 s"):
        Settings(_env_file=None)


def test_timeouts_are_per_provider(settings: Settings) -> None:
    assert settings.timeout_for("groq") == settings.groq_timeout_s
    assert settings.timeout_for("gemini") == settings.gemini_timeout_s
    assert settings.timeout_for("ollama") == settings.provider_timeout_s


def test_provider_chain_drops_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_PROVIDERS", "groq,gemini")

    assert Settings(_env_file=None).provider_chain == ["groq", "gemini"]


def test_reasoning_effort_by_role(settings: Settings) -> None:
    assert settings.reasoning_effort("router") == "low"
    assert settings.reasoning_effort("judge") == "medium"
    assert settings.reasoning_effort("vision") is None


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


def test_router_and_judge_roles_share_the_chat_model(settings: Settings) -> None:
    for role in ("router", "judge"):
        assert settings.require_model_id("ollama", role) == settings.ollama_chat_model


def test_unverified_free_tier_model_fails_loudly(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(errors.ConfigError, match="vision model configured for provider 'groq'"):
        settings.require_model_id("groq", "vision")

    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-test")
    assert Settings(_env_file=None).require_model_id("gemini", "router") == "gemini-test"


def test_trace_sample_rate_is_a_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACE_SAMPLE_RATE", "1.5")
    with pytest.raises(ValidationError, match="trace_sample_rate"):
        Settings(_env_file=None)


def test_caches_live_outside_the_repository(settings: Settings) -> None:
    assert not settings.model_cache_dir.is_relative_to(REPO_ROOT)
    assert not settings.photo_cache_dir.is_relative_to(REPO_ROOT)


def test_model_cache_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert Settings(_env_file=None).model_cache_dir == Path.home() / ".cache" / "food-concierge" / "models"
    assert Settings(_env_file=None).photo_cache_dir == Path.home() / ".cache" / "food-concierge" / "photos"


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


Environment = Literal["development", "ci", "production"]


@pytest.mark.parametrize(
    ("environment", "explicit", "exported"),
    [
        ("development", None, "full"),
        ("ci", None, "full"),
        ("production", None, "metadata"),
        ("production", "full", "full"),
        ("development", "metadata", "metadata"),
    ],
)
def test_production_traces_carry_metadata_only_by_default(
    environment: Environment, explicit: TraceContent | None, exported: TraceContent
) -> None:
    s = Settings(_env_file=None, environment=environment, trace_content=explicit)

    assert s.exported_trace_content() == exported
