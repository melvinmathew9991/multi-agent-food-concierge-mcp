# PRD — Multi-Agent Food Concierge (MCP)

> Living document. Source of truth for *what* is built and *why*. Architecture: `architecture.md`. Sequencing: `phases.md`. Revision 3 (2026-09-24): adds multi-agent orchestration, A2A, reranking, Qdrant comparison, Prefect workflows, and a Responsible AI programme (fairness, injection defence, tool access control, data governance).

Repo name: **`multi-agent-food-concierge-mcp`** (decided 2026-09-24). Python package: `food_concierge`.

---

## 1. Summary

A multimodal food concierge for a restaurant-aggregator catalog. Users describe what they want, attach a dish photo, or ask for a multi-course meal plan with calorie, budget and allergen limits. A **LangGraph supervisor** routes work to specialist agents (vision, recommender, meal planner) that call tools exposed by an **MCP server** (search, dish details, allergen checks, meal totals, image description). A **deterministic verifier** guarantees every constraint before anything reaches the user. Every run is traced, evaluated and monitored with **Langfuse**. The concierge is also reachable by other agents over **A2A**. Offline pipelines (ingestion, indexing, evaluation) are orchestrated with **Prefect**, and the system ships with a **Responsible AI review** (fairness, injection red-teaming, access control, data governance).

**Hard constraint: $0.** No paid APIs in the default configuration, no service that requires a credit card, free-tier hosting only.

## 2. As-built baseline (audited 2026-09-24)

Summary only; details in `docs/baseline-defects.md`.

- The starting point was a course exercise (not redistributed here): a Streamlit + LangChain 0.2 + Bedrock + FAISS chatbot. It makes up to 11 sequential LLM calls per query, leaves allergen safety to an LLM Yes/No check, crashes on malformed JSON, uses a retired model, and has no tests, evaluation or version control.
- Phase 0 is complete: package layout, settings, error hierarchy, logging and CI.
- **Findings on the course dataset that drive requirements** (analysed locally):
  - 20/50 items have empty allergen labels.
  - **14/50 image descriptions mention allergen-relevant ingredients absent from the metadata.**
  - Egg-containing dishes are labelled vegetarian.
  - The catalog is synthetic: one duplicate item and 2 items with inconsistent nutrition.

## 3. Problem statement

Diners choose food by craving, diet, budget and appearance, and increasingly want an assistant to do the multi-step work: "plan dinner for two under $30, one of us is vegan, nothing with nuts". Single-shot RAG can't plan across courses or do arithmetic reliably, and an LLM can't be trusted to enforce allergen rules over incomplete data. **We need** an agent that can plan, use tools, ask when unsure, and whose safety-critical decisions are made by code, with measured quality and full observability, at zero running cost.

## 4. Target users

| Persona | Need | Surface |
|---|---|---|
| **Diner** | Recommendations, similar-to-photo search, meal plans within limits | Streamlit chat UI |
| **AI-assistant user / developer** | Use the menu tools from Claude Desktop, Cursor or any MCP client | MCP server (stdio + Streamable HTTP) |
| **Integrator** | Call search / agent over HTTP | FastAPI (`/v1/*`) |
| **Other AI agents** | Delegate "recommend dishes" / "plan a meal" tasks | A2A server (Agent Card + task API) |
| **ML / LLMOps reviewer** | Inspect traces, prompts, evaluation results and quality gates | Langfuse, `docs/evaluation.md`, CI |
| **Responsible AI reviewer** | Check fairness, safety, access control and data handling | `docs/responsible-ai/system-card.md`, red-team results |

## 5. Scope

### In scope
- Text, image, and text + image requests over the menu catalog.
- **Agentic tasks:** single recommendations, "more like this", dish comparison, and **multi-course meal planning** under calorie, price, diet and allergen limits.
- **MCP server** exposing tools, resources and prompts, usable by the agent and by external MCP clients.
- **Multi-agent LangGraph:** supervisor (task routing) → specialist agents (vision, recommender, meal planner), each with least-privilege tool sets → deterministic verifier → human-in-the-loop confirmation → response; short-term, long-term and contextual memory; step and tool-call budgets. Single-agent vs multi-agent is measured, not assumed.
- **A2A interoperability:** the concierge is exposed as an A2A server (Agent Card, streaming tasks); a demo client agent delegates to it.
- **Retrieval:** dense (FAISS) + BM25 + hybrid + cross-encoder reranker; Qdrant as an alternative vector store benchmarked against FAISS.
- **Workflow orchestration:** Prefect flows for ingestion → indexing → evaluation → report, run locally and on a GitHub Actions schedule.
- **Responsible AI:** exposure fairness across restaurants, popularity bias, counterfactual consistency, confusion matrices for classifiers, injection red-teaming, content-safety screening, tool access control, trace masking, data governance, system card and compliance mapping.
- Hard safety constraints enforced in tools and the verifier, never by the LLM.
- **LLMOps:** tracing, prompt registry and versioning, datasets and experiments, retrieval + agent evaluation, calibrated LLM-as-judge, CI quality gates, online feedback scores, dashboards.
- Model layer: Groq and Gemini free tiers (hosted), Ollama (local), OpenAI-compatible interface; **AWS Bedrock implemented but tested only against stubs** (off by default, never run live).
- Zero-cost deployment: Hugging Face Spaces (API + MCP), Streamlit Community Cloud (UI), GitHub Actions (CI/CD).

### Out of scope
- Ordering, payments, user accounts. Full MCP OAuth 2.1 is backlog; the demo uses scoped bearer tokens + rate limits.
- CrewAI, AutoGen, Pinecone, Weaviate, Temporal, Airflow, and live Azure AI / Vertex AI: deliberately excluded (one framework and one orchestrator in depth; zero-cost rule). Rationale recorded in ADRs.
- Any paid service or credit-card-backed account.
- Medical-grade allergen guarantees; the app shows derived allergen information with a disclaimer.
- Fine-tuning; real-time menu sync; catalogs over ~10k items (the design allows more, but that isn't tested).

## 6. Core features

Priority: **M** must, **S** should, **C** could.

| ID | Feature | Pri |
|---|---|---|
| F1 | Recommendations from text: up to 3 dishes with image, nutrition, price, rating, diet badge, allergen chips and a reason | M |
| F2 | Image input: dish photo → description → similar dishes; combinable with text | M |
| F3 | **Meal planning agent:** multi-course plans that meet calorie, price, diet and allergen limits; totals computed by a tool, not the LLM | M |
| F4 | Hard constraints: diet (vegan / vegetarian = lacto-ovo / eggless / non-veg), allergen exclusion (EU-14 ∪ US Big-9), max calories, max price, cuisine; never violated | M |
| F5 | Allergen tags from three sources (label ∪ ingredient lexicon ∪ vision description) with `contains` / `may contain` / `unverified` levels | M |
| F6 | **MCP server:** tools `search_dishes`, `get_dish`, `find_similar`, `check_allergens`, `meal_totals`, `describe_image`; resources for catalog, dishes and images; prompts `plan_meal`, `find_similar_dish`; stdio + Streamable HTTP | M |
| F7 | **Deterministic verifier:** re-checks every proposed dish against constraints, recomputes totals, validates grounding; sends violations back to the planner (bounded retries) | M |
| F8 | **Human-in-the-loop:** agent pauses for confirmation when a dish has unverified ingredients or limits must be relaxed | M |
| F9 | **Memory:** per-conversation state (checkpointer); long-term preferences such as allergies and diet, recalled across conversations | S |
| F10 | Model fallback chain: Groq → Gemini → retrieval-only answer; budgets on steps and tool calls | M |
| F11 | **Observability:** a trace per request with nested spans for LLM calls, tool calls and verifier; token usage and a notional "shadow cost" | M |
| F12 | **Evaluation:** retrieval metrics, agent trajectory metrics, task success, constraint-violation rate, calibrated LLM-judge; CI quality gates | M |
| F13 | Prompt registry: prompts versioned in the repo, synced to Langfuse, version recorded on every trace | S |
| F14 | Input/output guardrails: size limits, prompt-injection heuristics, off-topic routing, grounding checks | M |
| F15 | Streaming UI: token streaming, live tool-call timeline, meal-plan view with totals vs limits, feedback buttons | S |
| F16 | Retrieval-only search endpoint and MCP tool (no LLM), used as the quota-exhaustion fallback | S |
| F17 | AWS Bedrock provider (mock-tested; enable with credentials) | C |
| F18 | **Supervisor + specialist agents** (vision, recommender, meal planner) with per-agent tool allow-lists; ablation vs single agent | M |
| F19 | **A2A server:** Agent Card (skills `recommend_dishes`, `plan_meal`), task lifecycle with streaming updates; demo client agent | M |
| F20 | Cross-encoder reranker (ONNX, local) as a retrieval stage, adopted only if evaluation supports it | S |
| F21 | Qdrant (local mode) vector-store backend behind the same interface; benchmark vs FAISS + SQL pre-filter | S |
| F22 | **Prefect flows:** `ingest`, `build_index`, `evaluate`, `report`; scheduled nightly run via GitHub Actions | M |
| F23 | **Contextual memory:** thread summarisation/trimming; semantic recall of stated preferences | S |
| F24 | **Prompt-injection defence:** local classifier + delimiting/spotlighting of untrusted tool and catalog text + red-team suite with attack-success-rate gate | M |
| F25 | **Content-safety screening** of user input (Llama Guard-class model; local or free tier) | S |
| F26 | **Tool access control:** scoped tokens for MCP/A2A (`public`: search/read; `agent`: + describe_image and runs; `admin`: ingestion); per-agent tool allow-lists; tool-call audit trail | M |
| F27 | **Data governance:** PII/image masking in traces, no upload retention, preference view/delete, lineage via manifests, data card | M |
| F28 | **Responsible AI review:** fairness (restaurant exposure, popularity bias), counterfactual consistency, confusion matrices (allergen tagger, router, injection classifier), complexity/cost profile, system card, compliance mapping, AI-interaction disclosure | M |

## 7. Non-functional requirements

| Category | Requirement |
|---|---|
| **Cost** | **$0.** Default config uses only free tiers and local models; no card-backed accounts; CI never calls live LLMs. |
| Latency | Single recommendation p95 ≤ 8 s; meal plan p95 ≤ 20 s on the free hosted tier. Retrieval-only ≤ 300 ms server time. |
| Agent bounds | ≤ 8 tool calls and ≤ 2 verifier repair loops per request; hard recursion limit. |
| Safety | **0 constraint violations** in evaluation and in verifier-checked output; allergen disclaimer always visible. |
| Reliability | Every external call has a timeout, bounded retries and a fallback; the demo still answers (retrieval-only) when free quotas run out. |
| Observability | 100% of API requests traced; trace ID returned to the client. |
| Security / privacy | No secrets in the repo; MCP tools are read-only; uploads are never stored; UI warns that free-tier providers may use inputs for training. |
| Reproducibility | One command rebuilds catalog + index; manifest binds index ↔ embedder ↔ data hash; evaluation reproducible from committed caches. |
| Testability | Full test suite runs offline: no network, no model downloads, no API keys. |

## 8. Success metrics (measured, not assumed)

- **Safety:** constraint-violation rate of the legacy approach vs this system (target 0) on the same queries.
- **Retrieval:** nDCG@3, Success@3 vs legacy, BM25, dense and hybrid baselines, with paired bootstrap CIs.
- **Agent:** task success on meal-planning scenarios, tool-selection accuracy, mean steps, verifier catch rate; **single vs multi-agent** on the same scenarios (success, latency, tokens).
- **Security:** prompt-injection **attack success rate** (target 0 for safety-relevant attacks); injection classifier TPR/FPR; unauthorised tool calls blocked (100%).
- **Responsible AI:** restaurant exposure ratio vs relevance-deserved exposure; rating–rank correlation beyond relevance; counterfactual safety consistency (target 100%) and result overlap; per-allergen precision/recall of the tagger.
- **Retrieval stack:** reranker and Qdrant results vs FAISS/hybrid, with latency.
- **LLM-judge:** agreement with human labels (Cohen's κ) reported before any judge score is used.
- **Ops:** p50/p95 latency, tokens per request, fallback rate. All numbers come from committed evaluation runs, including negative results.

## 9. Constraints, assumptions, risks

| Item | Mitigation |
|---|---|
| Free tiers have rate limits and may change terms | Fallback chain, retrieval-only mode, per-session + global rate limits; verify terms at each phase |
| Free-tier providers may use inputs for training | UI notice; no personal data; uploads not stored |
| Bedrock not run live | README states it plainly; stub tests prove the integration contract |
| Synthetic catalog (own, ~150 dishes) | Stated in the README and data card; metrics describe system behaviour, not real-world accuracy |
| Free hosts are ephemeral | Durable data (traces, scores, datasets) lives in Langfuse; SQLite holds rebuildable or demo-only state |
| Scope is large | Walking skeleton deployed by Phase 4; each phase produces something demonstrable |
| Dataset/image licensing | Course material kept private; the public catalog is authored for this project with openly licensed photos and per-image attribution |

## 10. Open questions

1. ~~Course material~~ Decided: kept private, not redistributed; own catalog built in Phase 2.
2. ~~Repo name~~ Decided: `multi-agent-food-concierge-mcp`.
4. ~~Hold-out images~~ Decided: altered catalog images (reported as a lower bound on real-photo performance).
6. "AIRB" interpreted as an AI review board-style assessment (bias, fairness, complexity, confusion matrices). Correct if a different framework is meant.
5. Defaults adopted pending confirmation: vegetarian = lacto-ovo plus an `eggless` filter; allergen standard = EU-14 ∪ US Big-9.
