# Optimization, scalability and reproducibility

> Living document. How the project spends its scarce resources (free-tier quota, latency, the owner's time) and keeps every result reproducible. Requirements: `prd.md` §7 (non-functional). Binding rules: `engineering-rules.md` §9. Sequencing: each item names the phase that delivers it (`phases.md`). Revision 1 (2026-10-11).

## 0. Principles

1. **Optimise the binding constraint.** Here that is the free-tier LLM quota: about 1,000 requests a day and 8,000 tokens a minute per Groq model (ADR-0006, ADR-0007). Next come end-to-end latency and the owner's review time. CPU and memory are not binding at 150 dishes.
2. **Remove calls before shrinking them.** A request answered without an LLM costs nothing and cannot hallucinate. Cheaper prompts come second.
3. **Measure first, against a budget.** Every optimisation names the metric it moves, where that metric is measured, and the decision rule for keeping it. Without a measurement, the default does not change.
4. **Every optimisation can be rolled back by configuration.** A flag or setting, never a code revert.
5. **Safety is never traded for efficiency.** Hard filters and the verifier stay deterministic. Nothing is cached in a way that could return a dish that breaks the asking user's constraints.

## 1. Budgets

The targets that the sections below are measured against. "Set in P*n*" means the budget is fixed from that phase's measurements and then enforced.

| Area | Budget | Today | Measured / enforced by |
|---|---|---|---|
| Retrieval-only server time | p95 ≤ 300 ms on the Space (PRD); ≤ 150 ms on the reference machine | rerank median about 51, 120 and 280 ms for the top 5, 10 and 20; full-stage p95 measured for ADR-0001 | P3 stage timings; P4 load check |
| Single recommendation | p95 ≤ 8 s (PRD) | — | P6 agent eval, Langfuse |
| Meal plan | p95 ≤ 20 s (PRD) | — | P6 |
| LLM calls a request | fast path 0; recommendation ≤ 3; meal plan ≤ 8 tool calls and ≤ 2 repairs (rules §2.4). The original approach made up to 11 | — | call-count gate on scripted-agent scenarios (P5) |
| Tokens a request (in + out) | set in P5 for each path, from measurement | — | trace metadata; P6 report p50/p95 |
| Tokens a photo | ≤ 1,950 now; target from the image-size experiment (§2.4) | p50 1,921 at 1,024 px | ADR-0007 measurement |
| Groq quota headroom | ≤ 900 calls a day per process (cap below the 1,000 limit); token caps below Groq's figure. Vision is also capped by Groq at about 200,000 tokens a day, or about 100 photos | calls and tokens enforced per model (180,000 tokens a rolling day each); photos capped at 90 a day (#49) | `models/usage.py` |
| PR CI | ≤ 5 min | about 3 min, with the retrieval gate | GitHub Actions |
| Nightly | ≤ 20 min | about 2 min, with the cache drift and reproduction checks | GitHub Actions |
| Container image and cold start | set in P4 | — | P4 build, `/ready` timing |
| Repository | no file over 5 MB; evaluation caches under 2 MB | enforced; caches 479 KiB | CI size check |

## 2. Model and token usage (product)

**Already in place:**
- a model chosen per role;
- low reasoning effort for routing and extraction;
- `max_output_tokens` 400;
- temperature 0 and a seed;
- call and token caps per model, and a daily photo budget;
- an 8 s deadline across the fallback chain;
- an empty reply counted as a failed attempt;
- photo descriptions cached by SHA-256 at ingestion;
- no LLM call per item (rules §3.1).

The levers below are listed by impact.

### 2.1 Don't call (P5)
- **Fast path:** a query whose constraints parse deterministically, and that needs no reasoning, is answered by retrieval alone. Examples: "vegan biryani under ₹300", or a known dish name.
- **Rule-based constraint extraction first:** patterns for diet, "no X" allergens, kcal and price caps, and cuisine run before any model. The LLM extractor runs only when rules leave text unexplained, and inferred constraints are confirmed by the user (A15).
- **Metric:** the share of requests with zero LLM calls, measured on the P6 scenario set.

### 2.2 Fewer calls (P5)
- **One structured call** returns intent and constraints together, replacing the original's separate rewrite and router calls.
- **Specialists run only when the supervisor routes to them.** The single-agent baseline stays, so this can be measured (ADR-0004).
- **The verifier is deterministic.** A failed verification triggers at most 2 repairs, then a safe fallback.
- **Gate:** scripted-agent tests assert the maximum LLM calls for each path. They are deterministic with the fake model, so an extra call fails CI.

### 2.3 Smaller calls (P5)
- **Static content first:** system prompt and tool schemas come first and are stable byte for byte, so a provider's prefix caching can apply. Whether Groq or Gemini actually discounts cached prefixes is checked in the P5 profile, not assumed.
- **Tool results are trimmed** to the fields the next step needs: top 3–5 dishes as id, name and a one-line summary, never full descriptions.
- **Bounded history:** `max_history_turns` 4, with older turns summarised (contextual memory, P5).
- **Prompt size budget:** each versioned prompt carries a character budget, and a test fails when a template exceeds it.
- **Output caps per role** come from measured reply sizes, not one global 400.

### 2.4 Cheaper images (P4)
- **Downscale user photos before any vision call.** Today only the catalog path resizes, to `DESCRIBE_PX` = 1,024. The P4 vision service applies the same step to uploads. A 12-megapixel phone photo would otherwise cost several times the tokens, or be resized by the provider in ways we don't control. Done in #53: `services/vision.py` shrinks uploads to `VISION_IMAGE_PX` (1,024) during validation.
- **Image-size experiment:** compare 768 px with 1,024 px on the 30-photo owner-scored sample (ADR-0007's protocol). For Qwen-style vision models, image tokens scale with pixel count, so 768 px should cost about 45% fewer image tokens. 480 px already failed: the model mistook a paneer curry for meat stew. *Done 2026-10-23 (#53): on Groq the premise failed. 768 px cost the same tokens as 1,024 px on all 30 photos (within 2%); the count follows the aspect ratio (wide about 900, square about 1,420, 4:3 about 1,930). The size stays at 1,024 px, and the 768 px descriptions were not scored (ADR-0007 update).*
- **Decision rule:** adopt the smaller size if dish-type correctness drops by at most 1/30 and false allergen terms do not rise. Rollback is the setting. (Not reached: with no token saving there was nothing to trade accuracy for.)
- **Per-session description cache** keyed by image SHA-256, so a user re-sending a photo costs nothing. Done in #53: in memory, keyed by SHA-256, model, prompt version and size, for `cache_ttl_hours`, at most `VISION_CACHE_ENTRIES`. Sessions arrive in P5, so it is shared by the process; a hit needs the identical file.

### 2.5 Response cache (P5)
- **Exact-match cache:** keyed by normalised query + resolved filters + prompt version + model + retrieval configuration + catalog data hash, with TTL `cache_ttl_hours`. It lives in process (LRU, bounded), so no paid service is needed. The key includes the filters, so a cached answer is only reused for identical constraints.
- **Semantic (embedding-similarity) cache rejected:** "nut-free curry" and "curry" are close in embedding space, and a cache hit across them would be a safety failure (rules §9.5).

### 2.6 Model routing and quota (P4–P5)
- **Smallest capable model per role:** the model that passes the P1 profile's structured-output and tool-call bar. Escalation happens only through repair → fallback, already in the router.
- **Per-model keys in the usage ledger** (ADR-0007 carry-over), with a separate photo budget, so photos cannot starve chat. Done in #49: 90 photos a UTC day, and 180,000 tokens a rolling day per Groq model.
- **Per-request token budget** in addition to the call budgets: when it is exhausted, the request degrades to a retrieval-only answer marked `degraded=true`.
- **Concurrency limit per provider:** an async semaphore keeps bursts under the per-minute caps instead of collecting 429s.

### 2.7 Measuring it (P5–P6)
- **Every trace carries:** calls, tokens in and out, model, cache hit or miss, fast path taken, fallback, degraded (`TraceMeta` already holds most of these).
- **The P6 report shows:** tokens a request (p50/p95) per path, calls a request, the cache hit rate, zero-LLM share, fallback rate and shadow cost (F11). Each is set against the original approach's figures.

## 3. Code execution and performance

| Practice | Status | Phase |
|---|---|---|
| Filter in SQL before vector search; batch embeddings; rerank only the top 10 or 20 (depth chosen in ADR-0001) | done | P3 |
| Lazy imports of heavy libraries (ONNX Runtime, fastembed) for CLI and tests | done | — |
| Load the embedder, reranker and indexes **once at startup**; `/ready` reports ready only after a warm-up query | partly: loaded once at start-up (#113); the warm-up query before `/ready` is planned | P4 |
| ONNX Runtime thread count as a setting, tuned to the Space's cores | planned | P4 |
| Async provider calls with per-provider concurrency limits (§2.6) | planned | P4–P5 |
| Stage timings (embed, search, fuse, rerank) on every search; LLM and tool spans in Langfuse | done | P3, P4 |
| Profile (cProfile) **only when a budget in §1 breaks**, not ahead of need | rule | — |
| Images are never kept in memory beyond the request; bytes never logged | done | — |
| Offline test suite under the CI budget; slow model tests behind markers (`model_download`, `live`) | done | — |

## 4. Scalability: what changes at each order of magnitude

The design is sized for the demo. The interfaces are the ones that let it grow without rewrites.

| Dimension | Today | About 10× | 100× and more | Trigger to move |
|---|---|---|---|---|
| Catalog items | 150; exact search O(n·d) | unchanged | HNSW (FAISS `IndexHNSWFlat`, or Qdrant as a server) behind `VectorStore` | search p95 above 50 ms |
| Catalog storage | SQLite, read-only, rebuilt atomically | unchanged | Postgres with pgvector (backlog) | writes at runtime, or several instances |
| LLM quota | free tier: about 1,000 calls a day, so roughly 300 recommendations a day at ≤ 3 calls each | fast path and cache raise answers per call | paid opt-in (`ALLOW_PAID_PROVIDERS`, Bedrock stub-tested) | sustained cap hits in Langfuse |
| Photos | 3–4 a minute (Groq tokens a minute) and about 100 a day (Groq vision tokens a day); the ledger stops photos at 90 | separate photo budget; a friendly "describe it in words" fallback | paid vision model | photo refusals above 5% |
| Processes | one Space process; ledger in memory | ledger and checkpointer in shared SQLite on a persistent volume | Redis or Postgres (paid tier) | more than one worker |
| API | stateless apart from the ledger and checkpointer | horizontal, once those are shared | — | — |
| Evaluation | 62 text and 127 image queries; caches about 0.5 MB | caches grow linearly; bootstrap stays cheap | sample queries per nightly run | nightly over 20 min |

**Capacity statement for the demo:** the free tier, not the code, sets the ceiling of about 300 recommendations a day. P7's UI says so when quotas run out, and retrieval-only answers continue.

## 5. Reproducibility

**Already in place:**
- universal, hash-checked lock files checked against `pyproject.toml` in CI;
- Python pinned to 3.11;
- third-party actions pinned to commits;
- the embedder fingerprint and an index manifest bound to the catalog data hash;
- input-hash no-op builds;
- vision descriptions cached and tracked;
- seeded models;
- results committed with sample sizes and intervals.

| Gap | Action | Phase |
|---|---|---|
| Result files record different subsets of their inputs | One `RunRecord` model written into every committed result: git SHA, lock-file hash, catalog data hash, embedder and reranker fingerprints, model names, prompt versions, seeds, and the CPU model, cores and OS family (never hostnames, users or paths; rules §8.4) | done in P3 for retrieval results; then every runner |
| Seeds scattered | `model_seed` stays; add `eval_seed` for splits, bootstrap and image transforms; no unseeded randomness in `src/` | done in P3 |
| Live-model replies are non-deterministic | Cache raw replies with model and prompt version; a regeneration is a new run, never an edit to the old one | P3 onwards |
| No end-to-end reproduction check | README section "Reproduce from a clean checkout" (`build` → `evaluate` → `report`), run as written by the nightly | P3: the nightly runs `build` then `evaluate --check`, and the README's "Evaluate retrieval" lists the commands. The full section comes with the close-out |
| Container builds drift | Base image pinned by digest; models downloaded at build and checked against the committed fingerprint; dependencies installed from the lock | P4: done (#122). Base pinned by digest, hash-checked locked installs; the indexes are built in the image with the model they record, so a load checks the fingerprint |
| Prompts could diverge between git and Langfuse | Git is the source of truth; `sync_prompts.py` pushes one way; the version is in every trace and cache key | P5 |

## 6. Environments and staging

| Environment | Purpose | Models | Data | Tracing | Trigger |
|---|---|---|---|---|---|
| `dev` (local) | build and try | fakes, or live free tiers via `.env` | full catalog | local Langfuse (Docker), full content | manual |
| `test` (pytest) | correctness | fakes only, no network | fixtures and the tracked catalog | off | every commit |
| `ci` (PR) | gates (`secrets`, `test` and `image` required; `image` builds only when an image input changed) | fakes and committed caches | tracked | off | every PR |
| nightly | dependency audit of both locks, drift, build, evaluate | open-model downloads; never hosted LLMs | tracked | off | schedule |
| `staging` | rehearse a release | live free tiers, staging keys and caps | built in the image | Langfuse, metadata only | every merge to `main` (from P4) |
| `prod` | public demo | live free tiers | built in the image | Langfuse Cloud Hobby, metadata only | tag |

- **`ENVIRONMENT`** (`development | ci | production`) selects the defaults (trace content: metadata only in production). Tests assert that settings refuse unsafe values: tokens shorter than 32 characters or shared between scopes and paid providers without the flag everywhere, and in production the localhost `ALLOWED_HOSTS` default and full trace content (#124). A staging value comes with a staging Space.
- **Build once, promote the same image** (rules §9.6). Staging deploys the image built from `main`. A tag promotes that same image digest to production, so what was tested is what ships.
- **Smoke test after every deploy:** `/health`, `/ready`, one `/v1/search`, MCP `list_tools` with a scoped token, plus one model call in staging only. A failed smoke test blocks promotion.
- **Configuration as the release lever:** the retrieval configuration (ADR-0001), model names (ADR-0006/0007) and agent topology (ADR-0004) are settings. Rollback means redeploying the previous tag or flipping a setting.
- **Approval needed:** a second Hugging Face Space for staging is free but a new resource, so it needs the owner's go-ahead in P4 (rules §5.5). Decided 2026-10-24: no staging Space for now. The `docker` workflow smoke-tests every image built from `main`, and the production image will be pushed once to GitHub's container registry, the Space running that exact digest.

## 7. Plan execution

1. **Plan before build:** each phase opens with a plan PR containing pre-registered decision rules and a risk table. Nothing is built until it merges (Phase 3: #38).
2. **Workstream size:** about 400 changed lines of code at most, excluding data, so one review sitting covers it. Larger work splits.
3. **Work in progress:** one open PR at a time, two only when they touch different files. Each branch is cut from the latest `main`.
4. **Definition of ready**, before a workstream starts:
   - its inputs exist;
   - any owner input is scheduled, with a time estimate;
   - its live-call budget (calls and tokens) is written down.
5. **Batch owner input:** one review request per workstream, saying what to look at and how long it takes. Long live jobs are timed to overlap owner tasks (Phase 3: the photo job runs during grading).
6. **Live jobs** are paced, resumable, idempotent and budgeted in advance, never run in CI, and spread across days when two would share a quota.
7. **Timebox spikes,** and record negative results in the plan or an ADR (the Phase 1 Gemini latency, the Phase 2 vision tag reversal).
8. **Handover between sessions:** the phase status in `phases.md` and the local report say where work stopped and what is next, so work restarts from them rather than from rediscovery.
9. **Delivery metrics** in the phase report: PR cycle time, CI duration against its budget, live calls spent against plan.

## 8. Where each item lands

| Phase | Items |
|---|---|
| P3 | `RunRecord`, `eval_seed`, stage timings, reproduce-from-clean-checkout in the nightly, the 150 ms retrieval guard |
| P4 | user-photo downscaling, the image-size experiment, per-model ledger keys and photo budget, warm start and `/ready`, ONNX threads, concurrency limits, `APP_ENV`, staging Space (approval), digest-pinned image promoted from staging, deploy smoke test |
| P5 | fast path, rule-based extraction, one structured intent call, trimmed tool results, prompt size budgets, per-request token budget, exact-match response cache, call-count gate, provider prefix-caching check |
| P6 | tokens, calls and latency per path against §1 and against the original approach; cache hit rate; zero-LLM share; shadow cost |
| P7 | capacity notice and retrieval-only continuation in the UI; photo fallback message |
