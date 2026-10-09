# Data handling

What user data this service touches, where each piece goes, what protects it and how long it stays. Current as of Phase 1 (model layer and tracing); rows marked *planned* arrive in the phase named. Change this file whenever a new kind of data, a new destination or a new external service is added.

This is an engineering record, not legal advice.

## Why it matters here

Diners tell a food concierge about their allergies, intolerances and diet ("no peanuts, my son is coeliac"). That is health information, a special category under GDPR Art. 9, and personal data under India's Digital Personal Data Protection Act 2023. It is in almost every request, so it can't be masked away without making the request meaningless. The design minimises where it goes instead.

## Data inventory

| Data | May contain | Sent to | Kept by this service | Controls |
|---|---|---|---|---|
| Chat messages | dietary needs and allergies (health), names, places, contact details | the chat model provider: Groq, Gemini on fallback | not stored; traces as below | input limit 500 chars (`MAX_QUERY_CHARS`); traces metadata-only in production |
| Photos | faces, location and device in EXIF | the vision model provider | never stored (P7) | re-encoded without EXIF, XMP, ICC or text chunks before sending (`models/images.py`); JPEG, PNG and WebP only, ≤ 5 MB and 40 MP |
| Model replies | echoes of the message | user; traces | not stored; traces as below | numbers and allergen facts come from tools, not the model (rules §2.3) |
| Stated preferences (*planned, P5*) | diet, allergies | stored locally | until the user deletes them | only what the user explicitly stated; view and delete in the UI (rules §2.6) |
| Trace metadata | request ID, role, provider, model, attempts, tokens, prompt version, error code | Langfuse | Langfuse retention, below | by rule never user text |
| Logs | event names, provider, error codes, trace ID | stdout of the host | host log retention | never prompts at INFO, secrets or image bytes (rules §3.4); values escaped |
| API tokens and keys | credentials | — | env and platform secrets only | git-ignored, gitleaks, push protection (rules §7.1) |

## External services

| Service | Receives | Their terms (checked 2026-10-06) | Consequence |
|---|---|---|---|
| Groq (primary model) | messages, replies | Does not train on inputs or outputs and keeps no inference data by default ([Groq: your data](https://console.groq.com/docs/your-data)) | none beyond the provider list in the UI notice |
| Ollama, local (catalog photo descriptions, offline) | catalog photos from Wikimedia Commons; no user data | runs on the owner's machine; nothing leaves it | run by hand (`scripts/describe_photos.py`), never by the app or CI |
| Gemini API, free tier (fallback model) | messages, replies; photos once a Gemini vision model is configured | On the free tier Google uses prompts and responses to improve its products, and human reviewers may read them; paid-tier terms apply to users in the EEA, Switzerland and the UK ([Gemini API terms](https://ai.google.dev/gemini-api/terms)) | the P7 data-use notice must say so before a user sends anything; weigh this before making Gemini the vision default |
| Langfuse Cloud, Hobby (traces, from P4) | trace metadata in production; masked content elsewhere | 30 days of data access; project data-retention policies need Pro, Enterprise or self-hosted Enterprise, not Hobby ([Langfuse data retention](https://langfuse.com/docs/administration/data-retention)) | no deletion window to rely on, hence `TRACE_CONTENT=metadata` in production |
| Langfuse self-hosted (development) | masked content | data in local Docker volumes; no retention policy in the open-source edition | `docker compose --env-file .env.langfuse down -v` deletes all traces |
| Wikimedia Commons (catalog photos, offline) | search queries made of dish names, and file downloads; no user data | requests identify the project by User-Agent and are paced to one per second; only CC0, public-domain, CC BY and CC BY-SA files are kept | run by hand (`scripts/fetch_photos.py`), never by the app or CI; files cached outside the repository |
| Prefect (offline build) | nothing: flows run locally | the local server's anonymous usage analytics are on by default | switched off (`PREFECT_SERVER_ANALYTICS_ENABLED=false`) |

Paid providers (OpenAI, Bedrock) are off by default (`ALLOW_PAID_PROVIDERS=false`) and not covered here.

## Traces

Two layers, both in `telemetry.py`:

1. **Masking, always on.** Every input, output and metadata value is masked before export (`mask_trace_data`): data-URI and long base64 images → `[image]`/`[binary]`, bytes → `[bytes: n]`, emails → `[email]`, payment cards (Luhn-checked) → `[card]`, phone numbers → `[phone]`. Prices, calories, dates, times and IDs are kept; tests check both directions. Text is cut at 4,000 characters. If masking fails, the payload is dropped. Langfuse media upload is off, because it runs before the mask hook. **Masking finds patterns: names, addresses and health details pass through it.**
2. **What is exported at all (`TRACE_CONTENT`).**
   - `full`: masked inputs and outputs are exported. Default in `development` and `ci`.
   - `metadata`: inputs and outputs are replaced by `[content not exported]`, and only an allow-list of attributes leaves the process: names, timings, model, parameters, usage, level, status, environment, release and metadata (`metadata_only_spans`). An attribute the allow-list doesn't name is dropped, so content that a future SDK adds under a new name is not exported by default. Default in `production`.

A failed block is marked with its error code, never its message or stack trace.

Production traces therefore answer "what happened, how fast, with which model, at what cost, did it fail" but not "what did the user say". A bad production answer is investigated by reproducing it locally, or from feedback the user chose to send (P7).

## Retention and deletion

| Where | How long | How to delete |
|---|---|---|
| Photos | the request only | not stored (P7 checks the uploader is cleared) |
| Messages and replies | the request only; the checkpointer window from P5 | P5 thread deletion (*planned*) |
| Preferences (*planned, P5*) | until deleted | the user deletes them in the UI |
| Traces, Langfuse Cloud Hobby | 30 days visible | content is not exported in production; delete the project to remove everything |
| Traces, local Langfuse | until the volumes are removed | `docker compose --env-file .env.langfuse down -v` |
| Logs | the host's retention | contain no message content |

## Requirement mapping

Engineering mapping, not legal advice; the full review is Phase 9 (`docs/responsible-ai/system-card.md`).

| Principle | Where it is met |
|---|---|
| Data minimisation (GDPR Art. 5(1)(c); DPDP Act purpose limitation) | metadata-only production traces; no message or photo storage; input limits |
| Special-category (health) data (GDPR Art. 9) | not stored except preferences the user states and can delete; not exported to traces in production |
| Data protection by design and by default (GDPR Art. 25) | production default is the strictest mode; allow-list export; masking fails closed |
| Notice before collection (DPDP Act; GDPR Art. 13) | P7 data-use notice and AI-interaction disclosure (*planned*) |
| Erasure | preference and thread deletion (P5, *planned*); trace content not exported in production |
| Third-party processing | provider terms above; paid providers off by default |

## Rules for contributors

- Never put user text in trace metadata, log fields or span names. Metadata is exported in every mode.
- New trace fields go through `mask_trace_data`; new Langfuse attributes stay dropped in `metadata` mode until added to the allow-list on purpose, with a test.
- A new external service, a new kind of user data or a change to a provider's terms updates this file in the same PR.
