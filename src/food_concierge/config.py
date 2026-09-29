"""Runtime settings, read from environment variables and an optional ``.env`` file.

Loading settings never fails because a provider's credentials are missing:
tests and local runs use Fake or Ollama. Credentials are checked when a provider
is actually built (``require_*`` helpers), so the failure names the real problem.
Loading does fail on a paid provider without ``ALLOW_PAID_PROVIDERS=true`` and on
unknown keys in ``.env``, so a typo can't silently change behaviour.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from food_concierge.errors import ConfigError

ProviderName = Literal["groq", "gemini", "ollama", "fake", "openai", "bedrock"]
EmbedProviderName = Literal["fastembed", "ollama", "fake", "openai", "bedrock"]
KeyedProvider = Literal["groq", "gemini", "openai"]

PAID_PROVIDERS: frozenset[str] = frozenset({"openai", "bedrock"})

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_fallbacks() -> list[ProviderName]:
    return ["gemini"]


def _default_cors_origins() -> list[str]:
    return ["http://localhost:8501"]


class Settings(BaseSettings):
    # extra="forbid" rejects unknown keys in .env; unrelated OS environment variables are never read.
    # env_ignore_empty lets a copied .env.example leave values blank without overriding defaults.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="forbid", env_ignore_empty=True)

    # Providers (zero cost by default: free tiers and local models only)
    chat_provider: ProviderName = "groq"
    fallback_providers: Annotated[list[ProviderName], NoDecode] = Field(default_factory=_default_fallbacks)
    embed_provider: EmbedProviderName = "fastembed"
    allow_paid_providers: bool = False

    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_embed_model: str = "nomic-embed-text"

    # Paid providers: implemented and stub-tested only
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = ""
    openai_embed_model: str = "text-embedding-3-small"

    aws_region: str = "us-east-1"
    bedrock_chat_model_id: str = ""
    bedrock_vision_model_id: str = ""
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    # LLMOps
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3000"

    # Request limits and timeouts
    provider_timeout_s: float = Field(default=30.0, gt=0)
    provider_max_retries: int = Field(default=2, ge=0, le=5)
    max_output_tokens: int = Field(default=400, gt=0, le=4096)
    max_query_chars: int = Field(default=500, gt=0)
    max_image_mb: float = Field(default=5.0, gt=0)
    max_history_turns: int = Field(default=4, ge=0)

    # Cost guardrails (calls per UTC day, per paid provider)
    daily_call_limit_bedrock: int = Field(default=200, ge=0)
    daily_call_limit_openai: int = Field(default=200, ge=0)
    cache_ttl_hours: int = Field(default=168, ge=0)

    # API access: one token per scope (public / agent / admin)
    api_token_public: SecretStr | None = None
    api_token_agent: SecretStr | None = None
    api_token_admin: SecretStr | None = None
    rate_limit_per_min: int = Field(default=20, gt=0)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=_default_cors_origins)

    # Paths
    data_dir: Path = REPO_ROOT / "data"
    log_level: str = "INFO"

    @field_validator("fallback_providers", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        # Env vars arrive as "a,b"; pydantic-settings otherwise expects JSON.
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _paid_providers_need_opt_in(self) -> Settings:
        if self.allow_paid_providers:
            return self
        configured = {self.chat_provider, self.embed_provider, *self.fallback_providers}
        paid = sorted(configured & PAID_PROVIDERS)
        if paid:
            raise ValueError(f"Paid providers {paid} are configured but ALLOW_PAID_PROVIDERS is false.")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def max_image_bytes(self) -> int:
        return int(self.max_image_mb * 1024 * 1024)

    def daily_call_limit(self, provider: ProviderName) -> int | None:
        """Daily call cap for paid providers; ``None`` means uncapped (free providers)."""
        return {
            "bedrock": self.daily_call_limit_bedrock,
            "openai": self.daily_call_limit_openai,
        }.get(provider)

    def require_api_key(self, provider: KeyedProvider) -> str:
        key = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
        }[provider]
        if key is None or not key.get_secret_value():
            raise ConfigError(f"{provider.upper()}_API_KEY is not set.")
        return key.get_secret_value()

    def require_model_id(self, provider: ProviderName, purpose: Literal["chat", "vision"]) -> str:
        # Free-tier model names for Groq and Gemini are verified and added in Phase 1.
        # OpenAI uses one multimodal model for both purposes; Bedrock and Ollama may split them.
        model_ids: dict[tuple[str, str], str] = {
            ("bedrock", "chat"): self.bedrock_chat_model_id,
            ("bedrock", "vision"): self.bedrock_vision_model_id or self.bedrock_chat_model_id,
            ("openai", "chat"): self.openai_chat_model,
            ("openai", "vision"): self.openai_chat_model,
            ("ollama", "chat"): self.ollama_chat_model,
            ("ollama", "vision"): self.ollama_vision_model,
            ("fake", "chat"): "fake-chat",
            ("fake", "vision"): "fake-vision",
        }
        model_id = model_ids.get((provider, purpose), "")
        if not model_id:
            raise ConfigError(f"No {purpose} model configured for provider '{provider}'.")
        return model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
