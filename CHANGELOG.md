# Changelog

Notable changes, grouped by delivery phase (`docs/planning/phases.md`). Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) and are tagged at the milestones listed in the plan.

## [Unreleased]

### Phase 2: Data, safety and ingestion flows (in progress)

#### Added
- Catalog vocabularies (`ingestion/taxonomy.py`):
  - 15 allergen keys: the union of EU-14 and US Big-9;
  - one table mapping free-text terms onto the keys, where a term maps to every key it may mean ("nuts" → peanuts and tree nuts, "shellfish" → crustaceans and molluscs);
  - wheat implies gluten;
  - diets (vegan, vegetarian as lacto-ovo, non-vegetarian), cuisines and menu categories as closed sets, so a typo is an error rather than a new value.
- Catalog row schemas (`ingestion/schemas.py`) for `restaurants.csv` and `menu.csv`:
  - serves as a range;
  - a `contains_egg` flag behind the `eggless` filter;
  - restaurant allergen labels that tell "not labelled" (blank) apart from "declared none";
  - a hand-checked `true_allergens` column that accepts canonical keys only and stays out of the menu items the system uses.
- Catalog loader (`ingestion/loader.py`). It checks every row before raising and reports each problem with file, line and column. Errors include:
  - a diet that contradicts the ingredients (ghee in a vegan dish, chicken in a vegetarian one, an unflagged egg);
  - unknown allergen terms and duplicate or orphaned rows;
  - header mismatches.
  Calories more than 25% away from 4P + 4C + 9F are warnings.
- A 12-dish fixture catalog (`tests/fixtures/catalog/`) with fictional restaurants, deliberate label gaps and a dish served at two restaurants.
- Allergen tagging (`ingestion/allergens.py`) from three sources, each tag stored with its source, level and evidence:
  - the restaurant label → `contains`;
  - a versioned ingredient lexicon (`data/lexicon/allergens.yaml`, with Indian terms such as ghee, maida, hing, khoya and kasundi) → `contains`;
  - the photo description → `may_contain`.

  At each position the longest phrase wins, so peanut butter is peanuts and makki atta is not wheat. Opaque ingredients (masala mixes, chutneys, unnamed sauces) mark a dish unverified.
- The allergen safety gate (Phase 2 DoD), in the tests: every hand-checked allergen of every dish is tagged, or the dish is unverified, with unverified capped at 25% of dishes. `scripts/allergen_report.py` prints precision per allergen.
- Catalog photos from Wikimedia Commons (`ingestion/photos.py`, `scripts/fetch_photos.py`). Only CC0, public domain, CC BY and CC BY-SA files are kept; NC, ND, GFDL-only, non-free, restricted and unknown licences are refused.
  - Candidates are written to `data/raw/photo_review.csv`, with a local review page for choosing one photo per dish or "no suitable photo".
  - Approved photos are recorded in `data/raw/attributions.csv` with author, licence and SHA-256, and the loader validates them.
  - Files are fetched into a new `PHOTO_CACHE_DIR` outside the repository, verified against their hash, and thumbnailed to WebP of at most 480 px.
  - Requests carry a User-Agent naming the project, and are paced to one per second.
  - The review page remembers choices across reloads, and `fetch_photos.py review` rebuilds it from the sheet without querying Commons.
  - `approve` checks every chosen row before downloading anything, so a bad row fails in seconds instead of after the downloads.
- Owner-approved catalog photos: 131 of 150 dishes, recorded in `data/raw/attributions.csv`; 19 dishes have no suitable photo.
- The SQLite catalog (`storage/catalog_db.py`): restaurants, dishes, items, allergen tags with source and level, images, and descriptions. It is built into a temporary file and swapped in atomically.
  - `dish_key` groups the same dish across restaurants.
  - The document text for search is semantic only (name, description, cuisine, category, ingredients, diet); prices and calories stay in indexed columns.
  - A `meta` table records the document-text version, lexicon version and source-data hash. The hand-checked ground truth never enters the database.
  - `CatalogStore.candidate_ids(filters)` applies diet, eggless, excluded allergens, maximum calories, maximum price and cuisine as SQL. When allergens matter, unverified dishes are left out unless the caller asks for them. A property test checks 400 random filter combinations against a plain-Python oracle.
- Search indexes. FAISS (`IndexIDMap2(IndexFlatIP)`), Qdrant and BM25 sit behind one search interface in `services/retrieval.py`.
  - Hard filters are decided once in SQL; each backend only restricts its search to those ids.
  - `storage/indexes.py` writes vectors (`.npy`, pickling disabled), item ids and the FAISS index with a manifest: embedder fingerprint, dimension, item count, catalog data hash, document-text version and file hashes. Loading refuses an index built with another embedder, from other data, or with changed files.
  - Qdrant runs in memory, because its on-disk local mode persists with pickle.
  - A property test runs 300 random filter combinations on all three backends: no hit violates a filter, and none is missed.
- New dependencies `qdrant-client` and `rank-bm25`, approved in the engineering rules; both lock files were regenerated.
- `Filters.admits`, the hard filters restated in Python, for re-checking a proposed dish.
- The offline build as Prefect flows (`flows/`): `ingest` (validate, tag allergens, build the catalog), `build_index` (embed, build the indexes) and `build` (both), run with `python -m food_concierge.flows build`.
  - Each step is skipped when its output matches its inputs, so a re-run with unchanged inputs does nothing.
  - Index steps retry only on transient network failures.
  - Task results are not persisted, so Prefect pickles nothing, and Prefect's anonymous analytics are switched off.
  - The nightly workflow runs the build with the real embedder and checks that a second run does nothing.
- `prefect` in the `ingest` extra, approved in the engineering rules; the API's runtime lock is unchanged.
- `docs/data-card.md`: sources, what is synthetic and how, derived data, lineage from source hash to index manifest, and known limitations.
- The authored catalog (`data/raw/`): 150 dishes at 15 fictional restaurants, two thirds of them Indian, with prices in INR. It was written in three batches, and the owner reviewed each batch's allergen ground truth.
  - Nutrition is invented and consistent with 4P + 4C + 9F.
  - About a third of dishes have no restaurant label, and some labels are deliberately wrong.
  - Six dishes are served at two restaurants.
  - `data/raw/README.md` documents the columns, the label gaps, and the recipe assumptions behind the ground truth, such as hing and soy sauce containing wheat, and kimchi containing fish and shrimp.

#### Fixed
- Photo attributions accept Commons files with an uppercase extension (`.JPG`), and `http://` licence links are stored as `https://`.
- Thumbnails can be built from Commons originals of up to 250 MP (two approved photos are 200 MP). The higher limit applies only while thumbnailing approved, hash-pinned files; uploads keep Pillow's default.

#### Changed
- Engineering rules §1: `qdrant-client` is used in memory only, because its on-disk local mode persists with pickle.
- Architecture and data-handling documents updated to match what Phase 2 built: the build command, in-memory Qdrant, filtering through SQL candidate ids, `doc_text` with category, hash-checked re-runs, Wikimedia Commons as an offline source, and Prefect analytics off.

### Phase 1: Models, embeddings, telemetry (closed 2026-10-06)

One model layer for every later phase:
- Groq → Gemini through OpenAI-compatible endpoints, Ollama locally, and Bedrock and OpenAI only with `ALLOW_PAID_PROVIDERS`.
- Typed output with one repair, a wall-clock request deadline, and free-tier call and token caps.
- Local fastembed embeddings with a fingerprint that refuses a mismatched index.
- Langfuse tracing: masked in development, metadata-only in production.

Models were chosen from a measured profile (ADR-0006) and verified by a live smoke run. Vision is deferred to Phase 2.

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
- Live smoke test (`scripts/smoke_live.py`, run by hand, never in CI) for the Phase 1 definition of done:
  - one real call each to Groq, Gemini and Ollama;
  - a forced Groq authentication failure answered by Gemini, with the fallback recorded;
  - one typed-output call;
  - traces read back from the Langfuse server, checking that planted fake contact and card details arrive masked with `TRACE_CONTENT=full`, and that no content arrives at all with `TRACE_CONTENT=metadata` (in a child process, since Langfuse keeps one client per key);
  - optionally, local vision on up to five of your own photos.

  Results go to `eval/results/smoke_<UTC time>.json`, without keys or file names, with the commit marked `-dirty` if tracked files had uncommitted changes. Traces are read back through the Langfuse v4 observations API (`/api/public/v2/observations`); v4 servers no longer serve `/api/public/traces`.

#### Changed
- `GROQ_CHAT_MODEL` and `GEMINI_CHAT_MODEL` default to the ADR-0006 models instead of blank.
- `ModelCallRecorder` counts a fallback only when another provider answered, so a repair on the same provider is not a fallback.
- Model runs report their real provider (`groq`, `gemini`, `ollama`, `bedrock`) to callbacks and traces instead of `openai` for every OpenAI-compatible endpoint.
- `langchain` (required by Langfuse's LangChain callback; brings LangGraph 1.2 for Phase 5) and `opentelemetry-api` are direct dependencies. `websockets` is pinned at 16.1.1, the cap set by `langgraph-sdk`.
- Provider timeout lowered from 30 s to 4 s (Groq and Gemini have their own) and SDK retries from 2 to 0, applied only to the last provider: the old defaults allowed about three minutes per model call, and a retried 429 slept on Retry-After instead of moving to the next provider.
- Unknown provider failures are treated as the provider being unavailable, so they move to the next provider.
- CI fails when offline tests cover less than 100% of `src/` (the Phase 0 and Phase 1 definition of done; it was checked by hand before).

#### Fixed
- The embedder fingerprint can no longer weaken silently. It hashes the files fastembed actually loads (the model file from its description, which may be in a subfolder, plus any extra files such as external weights), instead of the alphabetically first `*.onnx` at the top of the snapshot. A model whose files can't be identified is refused with `NotReadyError`, instead of being fingerprinted as `unknown` with an empty hash that would match any other unknown model. For the configured BGE model the fingerprint is unchanged, so the committed reference stays valid. The nightly check now also fails when the fingerprint changes upstream, even if the vectors don't.
- The free-tier call caps are enforced; until now they were settings that nothing read. Every model attempt is counted against its provider's caps per UTC day and per rolling minute (`models/usage.py`), plus Groq's tokens per rolling minute (new `MINUTE_TOKEN_LIMIT_GROQ`, default 8,000, from the ADR-0006 rate-limit headers). A capped attempt is refused before any request with `BudgetExceededError`, which is now a provider error that falls back to the next provider. Failed attempts count, since the provider counts them; attempts refused by the deadline don't. Counts are per process and restart with it. Streamed replies count toward call caps, but toward the token cap only when the endpoint reports usage, which the router does not request yet.
- Typed output stays inside `REQUEST_DEADLINE_S`. The settings check covers one request per provider, but a repair makes two, so the default chain could take 16 s against an 8 s deadline. A wall-clock deadline (`models/deadline.py`, `model_deadline`) now wraps every typed call: each later attempt gets only the time left, and one with no time left is not started (`DeadlineExceededError`, which does not fall back). The agent and API can open the same deadline around a whole request.
- An empty streamed reply now counts as a failed attempt, as an empty non-streamed reply already did. Chunks are held back until the first one with content, so the next provider can still take over.

#### Security
- Production traces carry no message content. Masking finds emails, phones and images, but not names, addresses or allergies, and allergies are in almost every request. New `TRACE_CONTENT` (`full` | `metadata`; blank means `metadata` in production and `full` elsewhere). In `metadata` mode an export-stage hook replaces inputs and outputs with `[content not exported]` and keeps only an allow-list of structural attributes (names, timings, model, usage, level, status, metadata), so content an SDK adds under a new name is dropped by default.
- Payment card numbers (13–19 digits, Luhn-checked) are masked as `[card]`. Before, a spaced 16-digit card came out as `[phone] 1111`, mislabelled and with its last digits kept.
- `docs/data-handling.md`: what user data the service touches, where each piece goes (Groq, Gemini's free tier, Langfuse), the providers' training and retention terms, retention and deletion per destination, and a GDPR / DPDP Act mapping. Langfuse Cloud Hobby has no configurable retention, and Gemini's free tier may use prompts to improve Google's products; both are recorded there with sources.
- Docs no longer say "PII masked": README, PRD F27, architecture and engineering rules §7.5 now say what is masked and that production traces are metadata-only.
- The repair turn quotes the problems and the model's previous reply in marked data blocks, with block markers removed from the quoted text, so text echoed from the catalog or tools cannot pose as instructions (engineering rules §2.6b).

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
