"""Runtime settings, read from environment variables and an optional ``.env`` file.

Loading settings never fails because a provider's credentials are missing:
tests and local runs use Fake or Ollama. Credentials are checked when a provider
is actually built (``require_*`` helpers), so the failure names the real problem.
Loading does fail on a paid provider without ``ALLOW_PAID_PROVIDERS=true`` and on
unknown keys in ``.env``, so a typo can't silently change behaviour.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from food_concierge.errors import ConfigError

ProviderName = Literal["groq", "gemini", "ollama", "fake", "openai", "bedrock"]
EmbedProviderName = Literal["fastembed", "ollama", "fake", "openai", "bedrock"]
KeyedProvider = Literal["groq", "gemini", "openai"]
# What a model is used for. Router and judge share the chat model until Phase 5/6 measurements
# justify a separate one; callers ask by role so that change stays inside config.
ModelRole = Literal["chat", "vision", "router", "judge"]
ReasoningEffort = Literal["low", "medium", "high"]
# What exported traces contain: masked inputs and outputs, or names, timings, models, usage and metadata only.
TraceContent = Literal["full", "metadata"]

PAID_PROVIDERS: frozenset[str] = frozenset({"openai", "bedrock"})

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_fallbacks() -> list[ProviderName]:
    return ["gemini"]


def _default_cors_origins() -> list[str]:
    return ["http://localhost:8501"]


def _cache_base() -> Path:
    # Outside the repository: model files and photos are large and the checkout may sit in a synced folder.
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "food-concierge"


def _default_model_cache_dir() -> Path:
    return _cache_base() / "models"


def _default_photo_cache_dir() -> Path:
    return _cache_base() / "photos"


class Settings(BaseSettings):
    # extra="forbid" rejects unknown keys in .env; unrelated OS environment variables are never read.
    # env_ignore_empty lets a copied .env.example leave values blank without overriding defaults.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="forbid", env_ignore_empty=True)

    # Providers (zero cost by default: free tiers and local models only)
    chat_provider: ProviderName = "groq"
    fallback_providers: Annotated[list[ProviderName], NoDecode] = Field(default_factory=_default_fallbacks)
    embed_provider: EmbedProviderName = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # BGE's retrieval instruction; fastembed does not add it. Optional for v1.5, so Phase 3 measures both.
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = Field(default=64, gt=0)
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    allow_paid_providers: bool = False

    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # Chosen from the Phase 1 model profile (ADR-0006, eval/results/model_profile_2026-10-01.json).
    # Groq has no vision model; Gemini vision stays blank until measured on own photos.
    groq_chat_model: str = "openai/gpt-oss-20b"
    groq_vision_model: str = ""
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    gemini_vision_model: str = ""

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
    tracing_enabled: bool = True
    trace_sample_rate: float = Field(default=1.0, ge=0, le=1)
    environment: Literal["development", "ci", "production"] = "development"
    # Blank: "metadata" in production, where public users' messages carry health details (allergies, diet)
    # that masking cannot find, and "full" elsewhere (docs/data-handling.md).
    trace_content: TraceContent | None = None

    # Model calls. The deadline bounds one model call across the whole fallback chain and matches the
    # PRD p95 target. The next provider is the retry: SDK retries (default 0) apply only to the last
    # provider, so a 429 moves on at once instead of sleeping on Retry-After.
    request_deadline_s: float = Field(default=8.0, gt=0)
    provider_timeout_s: float = Field(default=4.0, gt=0)
    # Per-provider attempt timeouts where latency differs: on 2026-09-30 Groq answered in ~0.4 s and
    # Gemini Flash-Lite in 2.3-3.5 s (small samples; the Phase 1 model profile re-measures them).
    groq_timeout_s: float = Field(default=3.0, gt=0)
    gemini_timeout_s: float = Field(default=5.0, gt=0)
    provider_max_retries: int = Field(default=0, ge=0, le=2)
    chat_temperature: float = Field(default=0.2, ge=0, le=2)
    # Routing, extraction, vision and judging run at temperature 0 with this seed where the provider accepts one,
    # so the same input gives the same decision and evaluation runs are repeatable.
    model_seed: int = Field(default=7, ge=0)
    # Thinking models spend output tokens on hidden reasoning; low effort keeps the visible answer in budget.
    reasoning_effort_chat: ReasoningEffort = "low"
    reasoning_effort_router: ReasoningEffort = "low"
    reasoning_effort_judge: ReasoningEffort = "medium"
    max_output_tokens: int = Field(default=400, gt=0, le=4096)
    max_query_chars: int = Field(default=500, gt=0)
    max_image_mb: float = Field(default=5.0, gt=0)
    max_history_turns: int = Field(default=4, ge=0)

    # Caps per provider, enforced in models/usage.py: paid ones bound cost, free ones stay under the
    # free-tier quotas. Groq's sit under its rate-limit headers (1,000 requests a day, 8,000 tokens a minute
    # per model; ADR-0006); Gemini reports none, so its caps are conservative. Local providers are uncapped.
    # 0 refuses every call to that provider.
    daily_call_limit_bedrock: int = Field(default=200, ge=0)
    daily_call_limit_openai: int = Field(default=200, ge=0)
    daily_call_limit_groq: int = Field(default=900, ge=0)
    daily_call_limit_gemini: int = Field(default=200, ge=0)
    minute_call_limit_groq: int = Field(default=25, ge=0)
    minute_call_limit_gemini: int = Field(default=8, ge=0)
    minute_token_limit_groq: int = Field(default=8000, ge=0)
    cache_ttl_hours: int = Field(default=168, ge=0)

    # API access: one token per scope (public / agent / admin)
    api_token_public: SecretStr | None = None
    api_token_agent: SecretStr | None = None
    api_token_admin: SecretStr | None = None
    rate_limit_per_min: int = Field(default=20, gt=0)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=_default_cors_origins)

    # Paths
    data_dir: Path = REPO_ROOT / "data"
    model_cache_dir: Path = Field(default_factory=_default_model_cache_dir)
    photo_cache_dir: Path = Field(default_factory=_default_photo_cache_dir)  # catalog photos and thumbnails
    log_level: str = "INFO"

    @field_validator("fallback_providers", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        # Env vars arrive as "a,b"; pydantic-settings otherwise expects JSON.
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _chain_fits_deadline(self) -> Settings:
        # Worst case for one model request: every provider times out once, and the last one also uses its
        # retries. Calls that make several requests (typed output's repair) are bounded at run time instead,
        # by models/deadline.py.
        timeouts = [self.timeout_for(provider) for provider in self.provider_chain]
        worst_case = sum(timeouts) + timeouts[-1] * self.provider_max_retries
        if worst_case > self.request_deadline_s:
            raise ValueError(
                f"Provider timeouts add up to {worst_case:g} s across the fallback chain, more than "
                "REQUEST_DEADLINE_S; shorten the timeouts, the chain or the retries."
            )
        return self

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
    def provider_chain(self) -> list[ProviderName]:
        """The chat provider followed by its fallbacks, each once, in order."""
        return list(dict.fromkeys([self.chat_provider, *self.fallback_providers]))

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
        """Calls allowed per UTC day; ``None`` means uncapped (local and fake providers)."""
        return {
            "bedrock": self.daily_call_limit_bedrock,
            "openai": self.daily_call_limit_openai,
            "groq": self.daily_call_limit_groq,
            "gemini": self.daily_call_limit_gemini,
        }.get(provider)

    def minute_call_limit(self, provider: ProviderName) -> int | None:
        """Calls allowed per minute on free tiers; ``None`` means no per-minute cap."""
        return {
            "groq": self.minute_call_limit_groq,
            "gemini": self.minute_call_limit_gemini,
        }.get(provider)

    def minute_token_limit(self, provider: ProviderName) -> int | None:
        """Tokens allowed per rolling minute where the provider limits tokens; ``None`` means uncapped."""
        return {"groq": self.minute_token_limit_groq}.get(provider)

    def exported_trace_content(self) -> TraceContent:
        """What traces export: ``TRACE_CONTENT`` if set, otherwise metadata only in production."""
        if self.trace_content is not None:
            return self.trace_content
        return "metadata" if self.environment == "production" else "full"

    def require_api_key(self, provider: KeyedProvider) -> str:
        key = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
        }[provider]
        if key is None or not key.get_secret_value():
            raise ConfigError(f"{provider.upper()}_API_KEY is not set.")
        return key.get_secret_value()

    def timeout_for(self, provider: ProviderName) -> float:
        """Timeout for one attempt at ``provider``."""
        return {"groq": self.groq_timeout_s, "gemini": self.gemini_timeout_s}.get(provider, self.provider_timeout_s)

    def reasoning_effort(self, role: ModelRole) -> ReasoningEffort | None:
        """Reasoning effort for thinking models; vision descriptions do not reason."""
        return {
            "chat": self.reasoning_effort_chat,
            "router": self.reasoning_effort_router,
            "judge": self.reasoning_effort_judge,
        }.get(role)

    def require_model_id(self, provider: ProviderName, role: ModelRole) -> str:
        # OpenAI uses one multimodal model for both purposes; the others may split them.
        purpose = "vision" if role == "vision" else "chat"
        model_ids: dict[tuple[str, str], str] = {
            ("groq", "chat"): self.groq_chat_model,
            ("groq", "vision"): self.groq_vision_model,
            ("gemini", "chat"): self.gemini_chat_model,
            ("gemini", "vision"): self.gemini_vision_model,
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
            raise ConfigError(f"No {role} model configured for provider '{provider}'.")
        return model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
