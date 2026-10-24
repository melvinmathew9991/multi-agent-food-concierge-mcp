# Changelog

Notable changes, grouped by delivery phase (`docs/planning/phases.md`). Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) and are tagged at the milestones listed in the plan.

## [Unreleased]

### Phase 4: MCP server and first deploy (in progress)

#### Fixed
- Deploy safety, from the Phase 4 audit (#124). Rate limits keyed on the connecting address, and uvicorn rewrote it from `X-Forwarded-For` whenever the peer was loopback, so a client could forge a new address per request and escape its limit (reproduced against a running server); behind the Space's proxy, every visitor would instead have shared one 20-a-minute bucket. Uvicorn's proxy-header handling is now off, and `AccessGate` takes the address the outermost trusted proxy recorded, per `TRUSTED_PROXY_HOPS` (new; 0 locally, 1 on a Space), ignoring entries a client could have written. Anonymous callers together may use `RATE_LIMIT_ANONYMOUS_PER_MIN` (new, 90 of the global 120), so token holders always have capacity. With `ENVIRONMENT=production`, settings refuse the localhost `ALLOWED_HOSTS` default (MCP would answer 421 to every request) and `TRACE_CONTENT=full`, and a server without an agent token logs a warning. Settings read only the repository's `.env`, not one in the working directory, which could break the server or change its settings when a client launched it from another project's folder. The container runs need `ALLOWED_HOSTS`; the docs now pass variables one by one, since a whole `.env` would carry `ENVIRONMENT=development` into the container.

#### Added
- ADR-0002, MCP-first layering (part of #121): services know no protocol or agent framework, the MCP server is the one tool surface, and the Phase 5 agent reaches the services only as an MCP client. In-process MCP added about 2 ms per search at the median (200 constrained searches, 7.8 against 9.7 ms). `tests/unit/test_layering.py` enforces it by reading each package's imports (type-only imports excepted).
- The server image (`Dockerfile`, part of #121). The base is pinned by digest and packages come from the hash-checked locks: the offline build runs with the dev lock in a build stage, and the runtime stage holds only the server's lock, the source, the data, the embedding model and the photo thumbnails, verified against their pinned SHA-256 (the originals are dropped). It runs as uid 1000 on port 7860 with `ENVIRONMENT=production` and `HF_HUB_OFFLINE=1`, so a cold start reads only local files. `.dockerignore` is an allow-list, so `.env` never reaches a layer. `scripts/smoke_deploy.py` checks a running server without model calls: `/health`, `/ready`, a constrained `/v1/search`, the six MCP tools, `describe_image` refused for the public scope and accepted for the agent token, and a wrong token refused with 401. A new `docker` workflow builds the image, starts it and runs the smoke test, on `main`, on PRs that change the image's inputs, and by hand.
- Connecting MCP clients (`docs/mcp.md`, `examples/claude_desktop_config.json`, #117): Claude Desktop and MCP Inspector over stdio and HTTP, what each tool, resource and prompt takes and returns, the error shapes and a troubleshooting table. The client config holds only the venv's Python and the module; provider keys stay in the repository's `.env`, which the server reads wherever it is started from (checked by starting it over stdio from outside the repository). The tool, resource and prompt schemas clients see are pinned in `tests/fixtures/mcp_schema.json`; a change fails the tests until the snapshot is regenerated with `UPDATE_MCP_SCHEMA=1` and reviewed.
- Access control and the HTTP server (`food_concierge/security/`, `food_concierge/api/`, closes #112). `python -m food_concierge.api` serves `/health`, `/ready`, `POST /v1/search` (retrieval only, the `search_dishes` arguments as JSON) and MCP over Streamable HTTP at `/mcp`, stateless with JSON responses so any replica can answer. Bearer tokens map to scopes (`API_TOKEN_PUBLIC/AGENT/ADMIN`, compared in constant time): a request without a token is public, by the owner's decision, so anyone can try the demo, while a token that matches none is refused with 401 rather than treated as public. Each MCP tool call is checked against its scope inside the server: `describe_image`, which spends the vision quota, needs `agent`, and a tool not listed needs `admin`; over stdio the local user has every scope. A denied call returns `isError` naming the scope it needs and is logged as a security event, and every call is traced with its tool and scope. Token-bucket rate limits apply per caller (its token, or its address when anonymous; `RATE_LIMIT_PER_MIN`, 20) and for the whole server (`RATE_LIMIT_GLOBAL_PER_MIN`, 120, new), with `Retry-After` on a 429. MCP answers only `ALLOWED_HOSTS` (new; localhost by default, add the deployed host), and its body limit fits one photo. Errors share one shape, `{"error": {"code", "message", "request_id"}}`. Search moved into `services/search.py`, so `/v1/search` and the MCP tool share one path and its safety re-check. New settings `API_HOST` and `API_PORT`.
- The MCP server (`food_concierge/mcp_server/`, #111, closes #110), on the official `mcp` SDK 2.3 (`MCPServer`, the 1.x `FastMCP`); `python -m food_concierge.mcp_server` runs it on stdio. Six read-only tools over the services: `search_dishes`, `find_similar`, `get_dish`, `check_allergens`, `meal_totals` and `describe_image`, with typed input and output schemas (diet and cuisine as enums; allergens in plain words such as "dairy" or "nuts", an unknown word refused by name). Search filters in SQL before ranking, then re-checks every result against the same constraints and drops and logs any that breaks one, so a violating dish is never returned; `find_similar` leaves out the same dish at other restaurants. Four resources (`menu://catalog`, `menu://dishes/{id}`, its WebP photo from the thumbnail cache, `menu://policy/allergens` with the words each allergen answers to) and two prompts (`plan_meal`, `find_similar_dish`). Domain errors come back as `isError` results with their safe message; anything unexpected only as "Error executing tool <name>". Inputs are bounded: queries by `MAX_QUERY_CHARS` (now enforced), dish lists by 20, results by 10, photos by `MAX_IMAGE_MB` before decoding. An allergen disclaimer (`services.allergens.DISCLAIMER`) rides on the server's instructions, allergen checks, the catalog and the policy. Tests run in-memory MCP sessions on the real catalog, including a property test that search never returns a dish breaking random constraints. The runtime lock gains `mcp`, `mcp-types`, `jsonschema`, `pyjwt` with `cryptography`, and `sse-starlette`; pip-audit is clean on both locks.
- Nutrition and constraint services (`services/nutrition.py`, `services/constraints.py`, #109, closes #107). `NutritionService.meal_totals` sums kcal, macros, rupees and servings over the dishes as listed, a repeat counting twice. `Constraints` holds the per-dish rules search filters on plus caps on the whole meal (`meal_max_kcal`, `meal_budget_inr`). `parse_constraints` reads loose tool input ("Dairy, nuts", "veg", "eggless", "Punjabi; south indian") through the taxonomy's aliases and refuses an unknown term by name rather than dropping it. `merge` only tightens: allergens joined, caps at the lower value, the strictest diet, confirmation of unverified recipes only if every set gave it; cuisine, a preference, takes the latest. `ConstraintService.check` names every rule each dish breaks and the meal caps it exceeds; a dish with an unverified recipe breaks the rules whenever allergens matter until the diner confirms (PRD F8), and allergens only a photo suggests are warnings. On 100 random constraint sets over all 150 dishes, the check passes a dish exactly when search would return it. The Phase 5 verifier builds on it.
- Catalog and allergen services for the MCP tools (`services/catalog.py`, `services/allergens.py`, #108). `CatalogService` returns one dish in full (ingredients, nutrition, the restaurant, the photo and its credit, every allergen tag with its source and evidence) or several as result-list summaries; prices are in rupees. `AllergenService.check` reports, for each dish and the allergens asked about, what it `contains`, what only the photo suggests (`may_contain`, a warning) and whether the recipe is `unverified`, with `clear` only when nothing is found and the recipe is checked. Both read the stored tags that search filters on, and a test shows the check and the filter agree on random allergen sets. Ids are trimmed and lower-cased; a request names 1 to 20 dishes, and unknown ids are named in a `NotFoundError` (new, HTTP 404). `CatalogStore` gained `items()` and `allergen_tags()`.
- The vision service (`services/vision.py`, part of #53). An upload is validated, stripped of metadata and shrunk to `VISION_IMAGE_PX` (1,024, the catalog's size) in one decode, so a phone photo costs no more tokens than a catalog photo. It is described with the catalog's name-free prompt. The description is cached in memory by SHA-256, model, prompt version and size for `cache_ttl_hours` (at most `VISION_CACHE_ENTRIES`, 256), so a photo sent again costs nothing. When the model can't describe it (the photo cap, a quota, an outage), the user is asked to describe the dish in words. `clean_image` gained `max_px`; `describe_data_uri` is the model call shared with the catalog path.
- `scripts/idle_latency.py` measures fallback latency after idle gaps of 0 to 60 min, with one cold and two paced warm calls per gap through the production structured-output path (part of #50). The first run, on 2026-10-21 (`eval/results/idle_latency_gemini_2026-10-21T1416Z.json`), followed Google's load, not the idle gap: from 14:16 to 15:05 UTC, Gemini 3.5 Flash-Lite took 11 to 60 s per call, cold or warm, with one timeout at 60 s. At 15:35 UTC it took 7 to 9 s, and from 16:21 UTC 1.1 to 2.0 s, cold calls after 45 and 79 min of idle included. Of 27 calls, 20 were over the 5 s attempt timeout. The second run, on 2026-10-22 (`eval/results/idle_latency_gemini_2026-10-22T0833Z.json`), agreed: 1.0 to 1.9 s for most of the day, cold calls after 78 and 127 min of idle included, but 1.3 to 10 s early on (7 of 12 over 5 s) and 18 s for all three calls at 13:09 UTC; 11 of 27 were over 5 s. ADR-0006 now treats Gemini as a best-effort fallback, and the P4 demo does not rely on a forced fallback (#50).

- The model profile set grows to 32 routing and 32 tool-call items, split 16 dev / 16 test (dataset v2, #51, audit M5), with prompt injection, Hinglish, typos and mixed greeting-plus-request messages; every route and tool is in test, and two genuinely ambiguous routing items accept two routes. `scripts/model_profile.py --split dev|test` runs one split and writes one file per run (`model_profile_<UTC time>_<split>.json`). It now waits out the usage ledger's per-minute caps instead of recording `budget_exceeded`: the longer set reached Groq's 8,000 tokens a minute. The owner reviewed the labels: rt13 accepts only `out_of_scope`, and a tool item may accept more than one tool, as tl26 does. On the single test run (2026-10-23) both models scored routing 15/16 and tool calls 16/16, missing the same item (a quoted "thank you" routed to smalltalk). Groq stays primary and Gemini the best-effort fallback (ADR-0006).

#### Changed
- Tokens protected end to end (#115). API tokens must be at least 32 characters (`MIN_API_TOKEN_CHARS`) and different for each scope, or settings refuse to load, naming the variable and how to generate one, never echoing the value. Tokens are compared as SHA-256 digests in constant time, so a guess learns neither timing nor length. A wrong token now spends the caller's address allowance before its 401, so guesses past the rate limit get 429; before, refusals were not counted and tokens could be guessed without limit. `Settings` reads the repo's git-ignored `.env` wherever the process starts, with a `.env` in the working directory overriding it, so Claude Desktop, which launches the MCP server from its own directory, finds the keys without them being pasted into its plaintext config. `.gitignore` adds `*.env`. The token rule (engineering rules §7.6), data handling, README and `.env.example` say how tokens are kept, sent and generated.
- Documents brought in line with Phase 4 items 1–4: the README's status and a new "Run the MCP server" section (stdio, HTTP, tokens, limits); architecture's deployment, folder, stack and configuration entries; the least-privilege rule for an open public scope (engineering rules 6a); data handling for search queries, bearer tokens, client addresses and security-event logs, which go to stderr, not stdout; baseline defects #7, #8 and #10 marked done for their P4 part; the Phase 3 and 4 status dates.
- Photos stay at 1,024 px (#53). Sent at 768 px, the 30-photo sample cost the same Groq tokens (within 2% on every photo): the count follows the aspect ratio, about 900 for wide, 1,420 for square and 1,930 for 4:3 photos, not the pixels. `scripts/vision_measurement.py --px` runs the measurement at another size under its own files (ADR-0007 update).
- Streamed replies count toward the token caps (#52). The router asks every OpenAI-compatible endpoint for streamed usage (`stream_usage`), takes it off each chunk, keeps the last one and yields it once as a usage-only chunk, recording it in the ledger when the stream ends, also when it fails or the caller stops reading. A live probe found that providers repeat it: Gemini sends a running total on every chunk and Groq the final totals twice, so counting each chunk would have overcounted Gemini about five times, in the ledger and in telemetry's chunk sum alike. Groq had reported streamed usage unasked, so its streams were already counted.
- Pillow 12.3.0 → 13.0.0, certifi 2026.7.22 → 2026.10.19 and uv 0.12.21 → 0.13.5, for advisories published on 2026-10-21 that failed `pip-audit` in CI. Pillow 13 fixes decoder flaws reachable from an uploaded photo (among them a JPEG heap overflow, CVE-2026-112403). fastembed caps Pillow below 13 but uses it only for image embeddings, which we never call, so `[tool.uv] override-dependencies` lifts the cap until a fastembed release allows 13. pip's resolver rejects that override, so the locked install (CI, nightly, README) uses `--no-deps`; the lock already pins and hash-checks every package (#98).
- filelock 4.0.7 → 4.2.4 for CVE-2026-112907, which failed `pip-audit` in CI; it comes in through huggingface-hub, which fastembed uses to download the embedding model (#119).
- opentelemetry-api and -sdk 1.45.0 → 1.45.1 (semantic-conventions 0.66b0 → 0.66b1) for CVE-2026-114603, published on 2026-10-22.
- Vision allergen tags skip terms the description denies ("no visible fruit or nuts", "without egg", "nut-free"). A denial lasts to the end of its clause: a sentence stop, "but" or "except", a place ("on the bread") or a comma that starts a new clause. On the Groq 30-photo sample, false terms fell from 3 of 24 to 2 of 23 (`eval/results/vision_measurement_groq_2026-10-14.json`). Hedged alternatives ("cream or yogurt") are kept, since 12 of the 16 on the two samples are right. Ingredient tags, the ones search filters on, still ignore negations, so the safety gate is unchanged, and so are the catalog's tags: none of its descriptions has a denied term (#54).
- Every image in the local Langfuse `docker-compose.yml` is pinned by digest (`tag@sha256:…`), not only MinIO, so a tag moved upstream cannot change what is pulled; the header comment says how to upgrade (#59, Phase 1 audit L5).
- Images given by URL or file id are refused (`InputValidationError`, "Images must be sent inline (uploaded), not as a link."); before, they went to the provider as they were. The provider would have fetched them itself, so their metadata survived and a caller could make it fetch any address. Every chat model checks this before sending, so the MCP tools and the API inherit it (#60, Phase 1 audit L7).
- The image pixel limit is a setting (`MAX_IMAGE_MEGAPIXELS`, 40), and the router's copy of the 5 MB byte limit is gone: guarded models take both from settings (#57). Engineering rule 2.8 now separates operational limits, which live in `config.py` once, from internal bounds, which stay named constants.
- Settings with no reader yet (`MAX_QUERY_CHARS`, `MAX_HISTORY_TURNS`, the API tokens, `RATE_LIMIT_PER_MIN`, `CORS_ORIGINS`) are commented as reserved, with the phase that enforces them (#58).
- The usage ledger counts each model separately (`groq/<model>`), as Groq limits each model on its own, and caps tokens per rolling 24 hours as well as per minute (`DAILY_TOKEN_LIMIT_GROQ`, 180,000 against Groq's 200,000). A vision call on a hosted provider also counts against a daily photo budget shared across providers (`DAILY_PHOTO_LIMIT`, 90, against about 100 a day that Groq's vision quota allows), so photos cannot spend chat's quota; at the cap the user is asked to describe the dish in words. An attempt that any of its budgets refuses is counted in none (#49).

### Phase 3: Retrieval evaluation, ranking and baselines (in progress)

Planned in #38. The plan's workstreams 1–7 are merged (#62–#70), and #71–#73 followed. The image descriptions finished on 2026-10-15 (#96, closes #43). Still open: the owner's grades (#42), ADR-0001 with one final test run (#47), and the close-out (#48). Production retrieval defaults don't change until ADR-0001 is accepted.

#### Added
- `Retriever` and `RetrievalConfig` (`services/retrieval.py`, #62): BM25, dense or hybrid search over FAISS or Qdrant, reciprocal rank fusion (`k_rrf` 60, 20 results from each list), an optional cross-encoder rerank depth, `dish_key` de-duplication and per-stage timings. The reranker carries a fingerprint, as the embedder does. Evaluation runs this same class, not a copy.
- The evaluation package `food_concierge/evaluation/` (#63), typed and inside the 100% coverage gate:
  - dataset schemas, with a test split locked by SHA-256 and refused without `--final`;
  - metrics checked against hand-computed values: nDCG@3, Success@3, Recall@3/10, MRR and Violation@3, judged against the ground truth;
  - a seeded paired bootstrap (10,000 resamples);
  - a cached embedder and reranker that implement the production protocols from committed files;
  - a cache manifest and a `RunRecord` with the git SHA, lock-file hash, data hash, fingerprints, models, prompt versions, seeds and machine.
- 62 text queries in six types with gold filters (`eval/datasets/retrieval.yaml`, #64, #70), plus a labelling guide. `scripts/label_queries.py` pools each query's top 10 from every candidate (1,179 dishes) into a blind, shuffled grading page; `import` and `split` follow the owner's grading. Every query was drafted with the catalog in view, and the report says so.
- Image queries (`eval/datasets/image_queries.jsonl`, #65): seeded Pillow transforms of all 127 described photos, the altered image's SHA-256, a dev/test side split by dish, and three description rungs (L0 named and unaltered, local; L1 name-free and unaltered, Groq; L2 name-free and altered, Groq). `scripts/image_queries.py describe` is paced, saves after each photo and resumes.
- Baseline C0, the original approach re-implemented (`evaluation/original.py`, `scripts/original_baseline.py`, #66). It takes the raw query, the unfiltered dense top 5 and one Groq `openai/gpt-oss-20b` Yes/No call per dish (310 calls, about 64k tokens). The raw replies are committed and scored offline with the original's strict parser and a fixed one. On the 12 constrained queries, 21% of C0's top 3 breaks a constraint, and 6 of 62 queries get no dish.
- Committed retrieval caches (`eval/caches/retrieval/`, 479 KiB, no pickle) and `scripts/build_eval_cache.py build|check` (#67). The PR gate is `scripts/retrieval_gate.py` against `eval/gates/retrieval_baseline.json`, and it runs in CI after the tests. It fails on any violation, an nDCG@3 drop of more than 0.02, a stale cache (naming the field) or a missing entry. Tests show that a random ranker and a pipeline without SQL filters both fail it.
- `evaluate` and `report` Prefect flows (`python -m food_concierge.flows evaluate [--check] [--split test --final]`, `report`, #68). They write `eval/results/retrieval_<date>_<split>.json` with bootstrap intervals overall and per slice, the ladder comparisons and the gate outcome. `report` renders `docs/evaluation.md` deterministically. A Langfuse mirror (a dataset item per query, an experiment run per candidate) runs only when keys are configured, and its failures are swallowed.
- Rerank depth 10 (`C5_hybrid_rerank10`, #69), added alongside depth 20 before any graded number existed, because the measured rerank latency puts depth 20 at risk of the 150 ms guard.
- Nightly: the caches are checked against freshly downloaded models (cosine ≥ 0.999, rerank scores within 1e-3), and `evaluate --check` reproduces the committed results.
- A provider-notes section in `architecture.md` (§18): the measured quirks of Groq, Gemini, Langfuse 4 and Wikimedia Commons in one table, linked from the scripts that hit them.

#### Changed
- The query set is 62 drafted queries. The owner's 10 catalog-blind queries were dropped on 2026-10-11, so the authoring bias is stated rather than measured (#70).
- `Telemetry` exposes its Langfuse client for the evaluation mirror.
- `.gitattributes` marks `*.npy` as binary; cache JSON is written with LF line endings.
- `uv.lock` is git-ignored (#72).
- The README's install and check commands now match the CI `test` job: install from the hash-checked lock, the lock-file check, the retrieval gate, and `pip-audit` on both lock files (#56, Phase 1 audit L4).
- The Langfuse mirror was checked against the local Docker Langfuse 4.48.0 (#73): 62 dataset items, 1,054 experiment runs and 1,054 `violation@3` scores, matching the results file. Langfuse 4 runs in events-only mode, so its legacy read endpoints return 404; the v2 observations, v3 scores and v2 metrics endpoints serve the data.
- The Groq vision daily capacity is corrected to about 100 photos (about 1,900 tokens each), from about 140, in ADR-0007, `optimization.md` and `phases.md`.

#### Fixed
- Per-photo token counts in `scripts/image_queries.py describe` and `scripts/vision_measurement.py describe` (#74). They were the change in the ledger's rolling one-minute total, which drops earlier calls that age out mid-call, so some rows read 0 or a few hundred. They now use the ledger's new running total (`Usage.tokens_total`). Rows written before the fix keep their recorded values and can't be recovered without new calls: the low values in `image_queries.jsonl` and in `vision_descriptions_groq.jsonl` (four of its 30 rows, 885–921) are likely undercounts.

### Phase 2: Data, safety and ingestion flows (closed 2026-10-09)

The data every later phase stands on:
- An authored 150-dish catalog with owner-reviewed allergen ground truth and 131 licence-checked Commons photos.
- Three-source allergen tags with an `unverified` state, and a safety gate at 100% recall.
- A SQLite catalog, FAISS, Qdrant and BM25 indexes, and Prefect build flows whose re-run does nothing.
- Measured photo descriptions, and the runtime vision model chosen from them (ADR-0007).

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
  - the photo description → `may_contain`, shown as a warning but not filtered on (ADR-0007).

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
- Photo descriptions (`ingestion/descriptions.py`, `scripts/describe_photos.py`): a name-free and a named description of each photo from local `qwen2.5vl:7b`, sent at 1024 px without metadata, cached by image SHA-256 in the tracked `data/processed/image_descriptions.jsonl`. The job saves after each photo and resumes.
  - The build reads the cache: the name-free description is the allergen tagger's vision source (`may_contain`), both variants are stored in the catalog, and the cache is part of `data_sha256`.
  - `scripts/allergen_report.py` includes the vision source and counts its tags against the ground truth.
- The vision measurement (`scripts/vision_measurement.py`): a seeded sample of 30 photos, a local review page beside the thumbnails, and scores for dish type, ingredients visible in the photo and allergen terms the photo gives no evidence of, with Wilson 95% intervals and latency p50/p95.
  - A `describe` command sends the sampled photos to a hosted model through the production path (same prompt, 1024 px, no metadata). It paces itself under the provider's tokens-a-minute cap and records tokens, latency and failed attempts per photo.
  - Owner-reviewed results for local `qwen2.5vl:7b` and Groq `qwen/qwen3.8-27b` are in `eval/results/`, with their samples in `eval/datasets/`.
- ADR-0007: Groq `qwen/qwen3.8-27b` describes user photos (dish type 22/30 correct and 30/30 correct or partial, p95 0.9 s, against 23/30, 29/30 and p95 4.8 s locally). Gemini gets no vision model, so a photo never falls back to its free tier.
- The SQLite catalog (`storage/catalog_db.py`): restaurants, dishes, items, allergen tags with source and level, images, and descriptions. It is built into a temporary file and swapped in atomically.
  - `dish_key` groups the same dish across restaurants.
  - The document text for search is semantic only (name, description, cuisine, category, ingredients, diet); prices and calories stay in indexed columns.
  - A `meta` table records the document-text version, lexicon version and source-data hash. The hand-checked ground truth never enters the database.
  - `CatalogStore.candidate_ids(filters)` applies diet, eggless, excluded allergens (`contains` tags), maximum calories, maximum price and cuisine as SQL. When allergens matter, unverified dishes are left out unless the caller asks for them. A property test checks 400 random filter combinations against a plain-Python oracle.
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
- The lexicon accepts an "es" plural only after a vowel or s, x, z, ch or sh, so "til" (sesame) no longer matches "tiles" in a photo description (lexicon version 2).
- The description cache is written with LF line endings on every platform.

#### Changed
- Vision `may_contain` tags warn instead of filter (ADR-0007). Search hides a dish when an excluded allergen is in `contains`, or when the dish is unverified; vision tags stay in the catalog for display. On this catalog this restores 47 false exclusions on 29 dishes.
- The safety gate checks recall of `contains`, the tags search filters on. It passes on all 150 dishes.
- `GROQ_VISION_MODEL` defaults to `qwen/qwen3.8-27b`; `docs/data-handling.md` records that Groq receives user photos and Gemini never does.
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
