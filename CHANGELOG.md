# Changelog

Notable changes, grouped by delivery phase (`docs/planning/phases.md`). Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) and are tagged at the milestones listed in the plan.

## [Unreleased]

### Phase 1: Models, embeddings, telemetry

#### Added
- Hash-checked lock files: `requirements.lock` (runtime) and `requirements-dev.lock` (CI), generated with `uv`. CI installs from the dev lock, fails when either lock no longer matches `pyproject.toml`, and audits both.
- LangChain (core, OpenAI, AWS), Langfuse, fastembed and pytest-asyncio, with SDKs capped below their next major version.
- Settings for model names per provider and role (`chat`, `vision`, `router`, `judge`); Groq and Gemini names stay blank until verified, and a blank name fails when the model is built.
- A per-request deadline (8 s, the PRD p95 target), per-day and per-minute call caps for the Groq and Gemini free tiers, tracing switches, and a model cache directory outside the repository.
- Provider error translation (`models/provider_errors.py`): OpenAI-compatible, httpx and Bedrock failures become app errors that say whether the next provider may succeed and whether an operator must act. New `ProviderRequestError` (never falls back) and `ProviderModelNotFoundError` (retired or closed models).
- `ScriptedChatModel`, a fake chat model that replays scripted replies, tool calls and failures and records every call, for router and agent tests.
- Model router (`models/router.py`): one model per role, Groq → Gemini by default through their OpenAI-compatible endpoints, Ollama locally, Bedrock and OpenAI only with `ALLOW_PAID_PROVIDERS`. Failures become app errors and move to the next provider, except rejected requests; empty replies count as failures; reasoning effort is set per role; providers without a model for a role are skipped. Contract tests run over an injected HTTP transport and a Bedrock stub.
- Per-provider attempt timeouts (Groq 3 s, Gemini 5 s); the settings check adds them up along the fallback chain against the deadline.
- Local embeddings (`models/embeddings.py`): fastembed `BAAI/bge-small-en-v1.5` with separate query and document embedding, a configurable BGE query instruction and normalised vectors; an `EmbedderFingerprint` (model, snapshot revision, model-file hash, dim, prefix) that refuses a mismatched index; a hashing fake embedder whose rankings are meaningful offline; a cross-encoder reranker wrapper and an overlap fake for Phase 3.
- Tracing (`telemetry.py`) over the Langfuse v4 SDK: spans and generations, a LangChain callback, flush and shutdown. Off without keys or with `TRACING_ENABLED=false`; Langfuse failures are logged and never reach the request, and a failed block is marked with its error code only, never its message or stack trace.
- Trace masking before export: image data URIs, long base64, bytes, emails and phone numbers are replaced; prices, calories, dates, times and IDs are kept, tested in both directions. Bounded in depth, items and text length, and fails closed. Langfuse media upload is switched off, because it runs before the mask hook.
- `TraceMeta`, one metadata shape for traced model calls, and `ModelCallRecorder`, a LangChain callback that records each provider attempt along the fallback chain (provider, model, error code, tokens), so a fallback is visible in traces and logs.
- Log lines inside a sampled trace carry its `trace_id`.
- `docker-compose.yml` for local Langfuse: pinned images, ports bound to 127.0.0.1, no default secrets (`.env.langfuse.example`), project and API keys created on first start.
- Typed model output (`models/structured.py`): `get_structured_model(Schema, role)` returns validated pydantic objects along the fallback chain. An invalid reply, or one with no structured part, gets one repair turn (the validation problems and the model's own reply); if that fails too the next provider is tried, and when every provider fails the caller gets `ProviderResponseError` instead of a crash. Replies are validated in one place, so every provider takes the same repair path.
- Provider capability table (`models/capabilities.py`): tools, structured-output method, vision, streaming, usage and seed per provider. Tool calling for Groq, Gemini and Bedrock; JSON schema for Ollama and OpenAI. Provisional until the model profile measures it.
- Image cleaning (`models/images.py`): every inline image a model is sent is re-encoded without metadata (EXIF with GPS and device, XMP, ICC, text chunks), with its orientation applied. Only JPEG, PNG and WebP within `MAX_IMAGE_MB` and 40 MP are accepted; bad images are refused before any request.
- `MODEL_SEED`: roles other than `chat` (routing, extraction, vision, judging) run at temperature 0 with a seed where the provider accepts one.
- Model profile (`scripts/model_profile.py`, live, never in CI): 20 fixed prompts (`eval/datasets/model_profile.yaml`: constraint extraction, routing, tool calls) through the production code path, scoring structured-output validity on the first try and after repair, correctness, tool accuracy, latency against the production timeout, tokens and rate-limit headers, with Wilson 95% intervals. Results in `eval/results/`.
- ADR-0006: Groq `openai/gpt-oss-20b` then Gemini `gemini-3.5-flash-lite` are the default chat models, chosen from two profile runs (40/40 and 39/40 correct, p95 0.73 s and 1.25 s, no 429 or 503). Gemini 3.5/3.8 Flash ran out of free quota within minutes, 3.1 Flash-Lite returned 503s, Gemini 2.5 Flash-Lite returned 404, Gemma rejects `reasoning_effort`, and local Llama models produced valid but wrong output (8/20). No vision default yet.
- Nightly workflow: the real embedding model is checked against committed reference vectors, since fastembed cannot pin a model revision. A `model_download` test marker keeps model downloads out of PR CI.

#### Changed
- `GROQ_CHAT_MODEL` and `GEMINI_CHAT_MODEL` default to the ADR-0006 models instead of blank.
- `ModelCallRecorder` counts a fallback only when another provider answered, so a repair on the same provider is not a fallback.
- Model runs report their real provider (`groq`, `gemini`, `ollama`, `bedrock`) to callbacks and traces instead of `openai` for every OpenAI-compatible endpoint.
- `langchain` (required by Langfuse's LangChain callback; brings LangGraph 1.2 for Phase 5) and `opentelemetry-api` are direct dependencies. `websockets` is pinned at 16.1.1, the cap set by `langgraph-sdk`.
- Provider timeout lowered from 30 s to 4 s (Groq and Gemini have their own) and SDK retries from 2 to 0, applied only to the last provider: the old defaults allowed about three minutes per model call, and a retried 429 slept on Retry-After instead of moving to the next provider.
- Unknown provider failures are treated as the provider being unavailable, so they move to the next provider.

### Phase 0: Foundation

#### Added
- Package layout (`food_concierge`), settings, application error hierarchy and key=value logging.
- CI on push and pull request: ruff lint and format, strict mypy, offline tests, pip-audit, gitleaks secret scan, and a 5 MB tracked-file limit. A weekly scheduled run scans the full history.
- Planning documents: PRD, architecture, design, engineering rules and delivery phases (revision 3), plus the baseline defect log and an audit reference index.
- Ignore rules for credentials, user data, local service state and material that is not redistributed.
- This changelog and the ADR index (`docs/adr/`).

#### Changed
- Settings follow `.env.example`: Groq by default with Gemini as fallback, one API token per scope, and Langfuse keys. A test keeps the two in sync.
- Paid providers (OpenAI, Bedrock) are refused at startup unless `ALLOW_PAID_PROVIDERS=true`.
- Unknown keys in `.env` are rejected, and blank values keep their defaults.
- The package version has a single source (`food_concierge.__version__`).

#### Security
- Log field values are quoted and escaped, so user text can't inject extra log lines or fields.
- The third-party CI action is pinned to a commit.
- `main` is protected by a repository ruleset: no force-push or deletion, and changes go through pull requests with green checks.
