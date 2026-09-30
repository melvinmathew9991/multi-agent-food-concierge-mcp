# Phases — remaining sprints

> Revision 3 (2026-09-24): adds multi-agent orchestration, A2A, reranker + Qdrant, Prefect, guardrails/access control and a Responsible AI programme. Only remaining work is planned. Baseline: `prd.md` §2. Progress: [`CHANGELOG.md`](../../CHANGELOG.md) and ADRs in [`docs/adr/`](../adr/). Audit refs (A#, and "audit #N" in `architecture.md`) are summarised in the [audit reference index](#audit-reference-index) below; the full audit is kept locally because it quotes material that is not redistributed. Feature refs (F#) → `prd.md` §6.

Size: **S** small · **M** medium · **L** large (split into several PR-sized commits).

Every phase ends with something demonstrable. The walking skeleton is live at the end of Phase 4.

## Skill coverage map

| Job-market skill | Where it is demonstrated |
|---|---|
| Agent frameworks (LangGraph; why not CrewAI/AutoGen) | P5 + ADR-0003 |
| Multi-agent orchestration (supervisor, planner-executor, routing) | P5, measured in P6 |
| Agent-to-agent (A2A) | P8 |
| MCP / tool standards | P4 |
| Function calling / API orchestration | P4, P5, P7 |
| RAG: retrieval + ranking (BM25, dense, hybrid, reranker) | P3 |
| Vector databases (FAISS, Qdrant) | P2, P3 |
| Agent memory (short-term, long-term, contextual) | P5 |
| Workflow orchestration (Prefect) | P2, P3 |
| Cloud AI (AWS Bedrock, stub-tested) | P1 |
| LLMOps (tracing, prompts, datasets, experiments, gates, monitoring) | P1, P3, P6 |
| Guardrails (safety, compliance, fallback) | P5, P7 |
| Prompt-injection protection | P5, red-teamed in P6 |
| Tool access control and data governance | P4, P5, P7 |
| Responsible AI (bias, fairness, complexity, confusion matrices) | P9 |

---

## Git workflow

- `main` ← one branch per phase, cut from the latest `main`.
- Conventional Commits with scope; no AI/tool references; no co-author trailers (`engineering-rules.md` §8).
- End of phase: tests + gates green → final commit → push and open the PR → **stop for review** → the owner merges.
- Public repo, no LICENSE. Tags: `v0.1.0` (P4 live MCP), `v0.2.0` (P7 live demo), `v0.3.0` (P8 A2A), `v1.0.0` (P10).

---

## Phase 0 — Foundation ✅ `chore/phase-0-foundation`, `chore/phase-0-closeout`
Done: package skeleton, settings, error hierarchy, logging, CI with secret scanning, `main` ruleset, ignore rules for credentials and user data, defect log, audits, plan revisions 2–3. Course baseline analysed locally and **not redistributed**.
Close-out (Phase 0 audit, 2026-09-25): settings aligned with `.env.example` (paid-provider opt-in enforced, unknown `.env` keys rejected); log values escaped against injection; tests isolated from the shell environment; mypy (strict) and pip-audit in CI; weekly full-history secret scan; 5 MB file check; third-party action pinned; CHANGELOG and ADR index started. `main` was already protected by a repository ruleset.

**DoD:** lint, strict type check, offline tests (100% coverage of `src/`), dependency audit and secret scan green in CI; `.env.example` and `Settings` checked against each other by a test; `main` accepts changes only through pull requests with green checks.

## Phase 1 — Models, embeddings, telemetry · `feat/phase-1-models-telemetry` · L
**Goal:** one model layer that every later phase calls: typed, with fallbacks, bounded by a request deadline, traced and masked. Plus a measured profile of what each free model can reliably do, which P3–P6 choices depend on. Provider selection, fallback order and `ALLOW_PAID_PROVIDERS` landed in the Phase 0 close-out; A20 and A24 are closed.

Decisions (2026-09-30): Langfuse self-hosted with Docker for development (WSL2 memory capped), Langfuse Cloud Hobby from the P4 demo; `uv` generates the lock files; `qwen2.5vl:7b` for local vision, `qwen2.5vl:3b` if it does not fit in 8 GB VRAM; heavy caches (fastembed, Docker volumes, Ollama) live outside the synced project folder.

1. **Dependencies (A19):** `requirements.lock` (runtime, for Docker) and `requirements-dev.lock` (CI), universal and hash-checked; SDKs capped below their next major; CI fails when a lock no longer matches `pyproject.toml`, and audits both locks.
2. **Settings:** model names per provider and role (`chat`, `vision`, `router`, `judge`), blank until verified live; a deadline for one model call across the fallback chain (default 8 s, the PRD p95 target), per-provider attempt timeouts (Groq 3 s, Gemini 5 s, from first measurements) and SDK retries only on the last provider, since the next provider is the retry and 30 s × 3 attempts × 2 providers cannot meet it; per-minute and per-day call caps for free providers; tracing switch, sample rate and environment.
3. **`models/router.py`:** `ChatOpenAI(base_url=…)` for Groq, Gemini and Ollama, `ChatBedrockConverse` behind the paid flag, `ScriptedChatModel` (scripted messages, tool calls and exceptions) for tests. SDK retries and `.with_fallbacks` only. Fallback on 429, timeout, 5xx, connection and auth errors (auth logged at ERROR); no fallback on 400. One translation from provider exceptions into the `Provider*Error` hierarchy. A capability table per provider (tools, JSON schema, vision, streaming, usage) picks the structured-output method; one repair, then fallback. Temperature 0 and a seed for routing and extraction. Images stripped of EXIF before they leave the process. Reasoning effort is set per role (low for routing and extraction), and an empty reply counts as a failed attempt: on 2026-09-30 thinking models spent a small output budget on hidden reasoning and returned nothing visible.
4. **`models/embeddings.py`:** fastembed `BAAI/bge-small-en-v1.5` with separate query and passage embedding (BGE query prefix), L2-normalised, batched, revision pinned; an `EmbedderFingerprint` stored in every index manifest, and loading a mismatched index fails. A token-hashing fake embedder, so offline tests can assert ranking order. Cross-encoder rerank wrapper for P3. A nightly drift check against committed reference vectors (cosine ≥ 0.999).
5. **`telemetry.py`:** a thin wrapper over the Langfuse v4 (OpenTelemetry) SDK: traces, spans, LangChain callback, flush, `trace_id` in log lines; a no-op when disabled or unconfigured; failures logged and swallowed. One `TraceMeta` model (request, provider, model, role, fallback used, attempts, tokens, prompt version, degraded). A recursive, size-bounded masking function for image data, bytes, emails and phone numbers, tested in both directions: PII is masked, and prices, calories, dates and IDs are not. `docker-compose.yml` for local Langfuse with pinned images bound to 127.0.0.1.
6. **Tests:** contract tests per OpenAI-compatible provider through an injected `httpx2.MockTransport` (success, 429 → fallback, 5xx → one retry → fallback, timeout, 400 without fallback, auth, malformed output → repair → error, deadline); botocore Stubber for Bedrock and its paid-flag guard; masking, fake models, fingerprint mismatch. Coverage stays at 100% of `src/`.
7. **Model profile (live, never in CI):** candidates come from real calls, not model listings (on 2026-09-30 Gemini still listed 2.5 models that refuse new accounts with 404). Per provider and role, a fixed set of 20 prompts measures structured-output adherence (first try and after repair), tool-call accuracy, p50/p95 latency and tokens, vision on own photos, and rate-limit headers; Llama Guard-class availability recorded for P5. Results in `eval/results/` with sample sizes and intervals; model names and fallback order in **ADR-0006**.

**DoD:** offline tests green at 100% coverage with lint, strict types, both lock audits and the secret scan; CI installs from the lock; live smoke on Groq, Gemini and Ollama visible as masked traces in local Langfuse; a forced Groq failure served by Gemini; Bedrock stub contract passes; ADR-0006 accepted with measured numbers.

## Phase 2 — Data, safety and ingestion flows · `feat/phase-2-data-safety` · M
**Goal:** trustworthy catalog, allergens and indexes, built by orchestrated flows.
0. **Own catalog v2 (publishable):** ~150 dishes across ~15 restaurants authored for this project: Atwater-consistent nutrition, hand-verified allergen ground truth, and a separate "restaurant-provided labels" column with documented, deliberate gaps (synthetic by design). One openly licensed photo per dish (e.g. Wikimedia Commons CC0/CC BY/CC BY-SA) fetched by script, with author, licence and source URL in `data/raw/attributions.csv`. Image descriptions generated with local Ollama vision ($0).
1. Loader + normalisation (diet semantics + `eggless`; categories vs cuisines; serves; Atwater check) (A11, A13).
2. Allergens EU-14 ∪ Big-9 from label ∪ lexicon ∪ vision; `contains` / `may_contain` / `unverified` (A1, A2, A12); hand-checked labels for all 50 items + table-driven test.
3. SQLite catalog; `dish_key` de-duplication; semantic doc text (A14).
4. `VectorStore` protocol with **FAISS** and **Qdrant (local mode)** backends; BM25; manifest; thumbnails.
5. **Prefect flows** `ingest` and `build_index` (retries, input-hash caching); CLI wrappers.
6. Name-free descriptions via local Ollama vision (A7); `docs/data-card.md` (sources, licences, synthetic-by-design fields, known gaps).

**DoD:** one-command flow run builds everything; re-run is a no-op; allergen tests green; property test: filtered search never returns a violating item on either backend.

## Phase 3 — Retrieval evaluation, ranking and baselines · `feat/phase-3-eval-ranking` · M
**Goal:** the "before" numbers and a CI gate, before any agent logic (A5).
1. `eval/datasets/retrieval.yaml`: ~60 queries, graded human labels for semantic queries; structured queries only for violation metrics (A3); dev/test split.
2. Image queries from **altered catalog images** (decided): random crops, rotation, colour/lighting shifts, blur, JPEG re-compression and background change, with a fixed seed and the transforms recorded per query. Reported as a **lower bound** on real-photo performance, with unaltered self-retrieval shown separately as a sanity check (A4). Name-free vs name-conditioned descriptions compared (A7).
3. Baselines and candidates: **original approach re-implemented on the new catalog** (raw query, k=5, LLM Yes/No relevance check; publishable), BM25, dense (FAISS), dense (Qdrant), hybrid RRF, **+ cross-encoder reranker**. Metrics: nDCG@3, Success@3, Violation@3, Recall@k (narrow); latency; paired bootstrap CIs (A6, A8, A18).
4. **Prefect flows** `evaluate` and `report`; nightly GitHub Actions schedule; results mirrored to Langfuse.
5. Committed embedding caches + retrieval gate in PR CI.

**DoD:** reproducible results; the gate fails on a deliberately degraded config (tested); retrieval stack and vector backend chosen from dev-split evidence (ADR-0001).

## Phase 4 — MCP server, access control, first deploy · `feat/phase-4-mcp-server` · L
**Goal:** secure, reusable tools, live on HF Spaces.
1. `services/` layer (catalog, retrieval, allergens, nutrition, constraints, vision).
2. FastMCP server: 6 tools, 4 resources, 2 prompts; typed schemas; `isError`; size limits.
3. **Access control:** scoped bearer tokens (`public` / `agent` / `admin`) on the HTTP transport; per-tool scope checks; audit spans; rate limits per token + global.
4. Transports: stdio + Streamable HTTP in a minimal FastAPI app (`/health`, `/ready`, `/v1/search`, `/mcp`).
5. Tests: in-memory MCP sessions, schema snapshots, scope-denial tests, error paths. Manual: MCP Inspector, Claude Desktop (`docs/mcp.md`, `examples/claude_desktop_config.json`).
6. Dockerfile (slim, non-root, models cached at build); GitHub Action deploys to a HF Space on tag. ADR-0002 (MCP-first layering).

**DoD:** Claude Desktop uses the tools locally; the public Space serves `/v1/search` and MCP with scopes enforced; `v0.1.0`. Creating the HF Space needs approval.

## Phase 5 — Multi-agent system, memory and guardrails · `feat/phase-5-agents` · L
**Goal:** a bounded, safe, observable multi-agent system over MCP.
1. Graph per Architecture §6: `guard_input` → `load_context` → supervisor → specialists (vision, recommender, meal planner) → deterministic `verify` → `confirm` (interrupt) → `respond` → `save_memory`; plus the **single-agent baseline graph**.
2. **Per-agent tool allow-lists**; budgets (tool calls, hops, repairs); safe fallback.
3. **Memory:** SQLite checkpointer (short-term), preference store with semantic recall (long-term), summarisation + trimming (contextual).
4. **Guardrails:** local injection classifier, content-safety screen, spotlighting of untrusted text, off-topic routing, output checks; inferred constraints surfaced for confirmation (A15).
5. Versioned prompts + `scripts/sync_prompts.py`; prompt version on traces.
6. Tests with the scripted fake model for every path (routing, repair, budget exhaustion, interrupt/resume, fallback, allow-list denial, injection fixtures); zero-violation assertions. ADR-0003 (LangGraph over CrewAI/AutoGen; supervisor design).

**DoD:** meal-planning scenarios run end to end on Groq with full traces; every graph path tested offline.

## Phase 6 — Agent evaluation, red-teaming and LLMOps · `feat/phase-6-agent-evals` · M
**Goal:** measured agent quality, security and operations.
1. `eval/datasets/agent.yaml` (~30 scenarios) and `eval/datasets/redteam.yaml` (direct/indirect injection, tool escalation, allergen override).
2. Metrics: task success, violations (0), tool-selection and routing accuracy, steps, repairs, latency, fallback rate, tokens + shadow cost; **attack success rate**; injection classifier TPR/FPR.
3. **Single vs multi-agent ablation**; default topology chosen from data (ADR-0004).
4. LLM-judge on local Ollama, calibrated vs ~40 human labels (κ) (A17).
5. Langfuse experiments (prompt versions, models); dashboards; `docs/evaluation.md` draft.
6. CI gates: agent trajectory tests + red-team suite with the fake model; live evals run manually and committed.

**DoD:** reproducible eval runs; judge calibration and red-team results reported; topology decision recorded.

## Phase 7 — API, UI and live demo · `feat/phase-7-api-ui` · M
**Goal:** a polished, governed public demo.
1. FastAPI: `/v1/agent/runs` (SSE), resume, feedback → Langfuse scores; token scopes; rate limits; placeholder-key refusal; `DATA_DIR` (A9, A21, A22).
2. Streamlit per `design.md` §9: streaming chat, tool timeline, meal-plan card, interrupt buttons, constraint chips, preferences panel (view/delete), feedback, **AI-interaction disclosure**, data-use notice (A23).
3. Deploy the UI to Streamlit Community Cloud; CI smoke test against the Space.

**DoD:** public URLs work from a clean browser; quota exhaustion degrades to retrieval-only (drill documented); `v0.2.0`.

## Phase 8 — A2A interoperability · `feat/phase-8-a2a` · M
**Goal:** agent↔agent delegation next to agent↔tool (MCP).
1. A2A server: Agent Card (`/.well-known/agent-card.json`), skills `recommend_dishes` + `plan_meal`, streaming task updates, `input-required` mapped to interrupts, artifacts = verified proposal.
2. Scoped bearer auth; audit spans; rate limits.
3. `examples/a2a_client_agent.py`: a "fitness coach" LangGraph agent that discovers the card and delegates meal plans.
4. Tests: in-process A2A client/server with the fake model; card snapshot; task-state tests. `docs/a2a.md` with a sequence diagram. ADR-0005 (MCP vs A2A boundaries).

**DoD:** the demo peer agent completes a delegated meal plan against the live Space; `v0.3.0`.

## Phase 9 — Responsible AI review · `feat/phase-9-responsible-ai` · M
**Goal:** a documented, measured Responsible AI assessment.
1. **Fairness:** restaurant exposure vs relevance-deserved exposure; popularity (rating) bias controlling for relevance; mitigation if disparity is material (e.g. tie-break policy), re-measured.
2. **Counterfactual consistency:** paired identity/cultural-cue requests; safety agreement (target 100%), top-3 Jaccard.
3. **Confusion matrices:** allergen tagger (per allergen), supervisor routing, injection classifier; error analysis of misses.
4. **Complexity profile:** steps, tokens and latency per task type, single vs multi-agent.
5. **Governance review:** data card final, masking verified on exported traces, retention and deletion paths, token scopes.
6. `docs/responsible-ai/system-card.md` structured around the **NIST AI Risk Management Framework** (Govern / Map / Measure / Manage), so it reads like an AI review board submission; compliance mapping ("engineering mapping, not legal advice"); Prefect `report` flow renders the RAI tables.

**DoD:** every metric reproducible from a committed run; limitations and failure modes documented.

## Phase 10 — Hardening and portfolio release · `chore/phase-10-release` · M
1. Dependabot (pip + GitHub Actions, which also keeps the pinned action current), PR template, CHANGELOG release notes (gitleaks, pip-audit and mypy are already in CI from Phase 0).
2. Small load test against a local container; latency/throughput table.
3. `docs/runbook.md` (quota exhausted, Space asleep, Langfuse down, token rotation), ADR index.
4. README: problem, architecture diagram, results (legacy vs new, retrieval, agent, red-team, RAI), live links, demo GIF, MCP + A2A quick starts, limitations, "what didn't work".
5. Repo description with measured headline numbers; `v1.0.0`.

**DoD:** a reviewer can understand, run and verify the project in 10 minutes from the README.

---

## Audit reference index

One-line summaries of the end-to-end audit of 2026-09-24. Severity: C critical · H high · M medium · L low.

| Ref | Sev | Finding | Addressed in |
|---|---|---|---|
| A1 | C | Image descriptions mention allergens missing from the dish metadata | P2 (vision as a third allergen source) |
| A2 | C | Ingredient lists are incomplete, so an ingredient lexicon alone under-tags allergens | P2 (`unverified` state), P9 |
| A3 | C | Labels derived from metadata make structured-query evaluation circular | P3 (violation metrics only) |
| A4 | C | Evaluating image queries with catalog photos leaks the answer | P3 (altered images, lower bound) |
| A5 | H | Evaluation was planned after the design choices it should inform | P3 moved before agent work |
| A6 | H | Recall@3 is uninformative for broad queries | P3 (nDCG@3, Success@3) |
| A7 | H | Descriptions generated with the dish name differ from what a user photo produces | P2, P3 (name-free descriptions) |
| A8 | H | No lexical baseline | P3 (BM25) |
| A9 | H | Per-client rate limits are ineffective behind a shared UI server | P4, P7 (scoped tokens, global limits) |
| A10 | H | A budget counter in ephemeral storage resets on every cold start | Superseded: no paid provider by default |
| A11 | H | Diet semantics undefined (eggs tagged vegetarian) | P2 |
| A12 | M | Allergen taxonomy not chosen | P2 (EU-14 ∪ Big-9) |
| A13 | M | Synthetic data: duplicate dishes, calories inconsistent with macros | P2 (`dish_key`, Atwater check) |
| A14 | M | Numbers in embedded text; filters already cover them | P2 (semantic-only document text) |
| A15 | M | Inferred filters can wrongly narrow results | P5 (inferred constraints confirmed by the user) |
| A16 | M | Hand-tuned rerank weights overfit and add popularity bias | P3 (cross-encoder reranker instead of hand-tuned weights), P9 |
| A17 | M | Faithfulness of generated explanations unmeasured | P6 (calibrated judge) |
| A18 | M | Small evaluation sets have low statistical power | P3 (bootstrap CIs) |
| A19 | M | No lock file; SDK lower bounds only | P1 |
| A20 | M | CI pip cache had no dependency file | Closed in P0 |
| A21 | M | Data paths depend on an editable install | P4, P7 (`DATA_DIR`) |
| A22 | M | Example config shipped a known API key | Closed in P0 (blank tokens); P7 refuses placeholders |
| A23 | M | User photos are sent to third-party providers | P7 (data-use notice) |
| A24 | L | Logging setup untested | Closed in P0 |
| A25 | L | Risk of over-building for a small catalog | Plan revision 3 |
| A26 | L | FAISS is more than a 50-item catalog needs | P3 (vector backends benchmarked) |
| A27 | L | Course-material licensing unresolved | Closed in P0 (not redistributed; own catalog in P2) |

## Backlog
- Full MCP OAuth 2.1 authorisation flow for the remote transport.
- Multimodal (image) embeddings if Phase 3 shows image-query weakness.
- Postgres/pgvector (free tier) if persistence beyond demo becomes necessary.
- Admin single-item ingestion endpoint.
