# Multi-Agent Food Concierge (MCP)

A multimodal food concierge: a LangGraph agent plans meals and recommends dishes from text and photos using tools served over the Model Context Protocol (MCP). Dietary and allergen constraints are enforced in code by the tools and a deterministic verifier, every run is traced and evaluated with Langfuse, and the whole stack runs at zero cost on free tiers and local models.

> **Status:** under active development. Phases 0–2 are complete: the model layer, and the catalog with allergen safety, storage, search indexes, build flows and a measured vision model ([ADR-0007](docs/adr/0007-vision-model-and-vision-allergen-tags.md)). Phase 3 (retrieval evaluation) is in progress: the retrieval stages, the evaluation package, the query sets, the original approach re-implemented as a baseline, the CI retrieval gate and the `evaluate`/`report` flows are built; the owner's relevance grades and ADR-0001 (the chosen retrieval stack) are next. See [`docs/planning/phases.md`](docs/planning/phases.md) for the plan, [`CHANGELOG.md`](CHANGELOG.md) for progress and [`docs/baseline-defects.md`](docs/baseline-defects.md) for what is being fixed from the original implementation.

> **Acknowledgement:** this is an independent rebuild of a course exercise on multimodal RAG. No course code, data or images are redistributed; the catalog in this repository is authored for this project.

## Quick start (development)

```bash
python -m venv .venv            # keep it outside cloud-synced folders
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
cp .env.example .env
```

The same checks as CI:

```bash
ruff check . && ruff format --check .
mypy
pytest --cov=food_concierge --cov-fail-under=100
pip-audit --skip-editable
```

## Build the catalog and search indexes

```bash
python -m food_concierge.flows build   # validate, tag allergens, build SQLite and the indexes; a re-run with unchanged inputs does nothing
python scripts/allergen_report.py      # the allergen safety gate and per-allergen precision
```

The first build downloads the local embedding model (fastembed, cached outside the repository). The catalog database and indexes go to `data/processed/` and are git-ignored; the photo descriptions there (`image_descriptions.jsonl`) are tracked, so the build never needs a vision model.

Catalog photos are maintained by hand, never by CI:

```bash
python scripts/fetch_photos.py sync    # fetch the approved Commons photos into the local cache and verify their hashes
python scripts/describe_photos.py      # describe new photos with the local Ollama vision model (qwen2.5vl:7b)
``` The catalog is synthetic by design; see [`docs/data-card.md`](docs/data-card.md) and [`data/raw/README.md`](data/raw/README.md).

## Evaluate retrieval

Offline and deterministic: the evaluation runs from committed caches, with no model download and no network.

```bash
python scripts/retrieval_gate.py                  # the CI gate: no constraint violation, no nDCG@3 drop over 0.02
python -m food_concierge.flows evaluate           # every candidate; writes eval/results/retrieval_<date>_<split>.json
python -m food_concierge.flows evaluate --check   # reproduce the committed results instead of writing (nightly)
python -m food_concierge.flows report             # render docs/evaluation.md from the newest results
```

The latest numbers are in [`docs/evaluation.md`](docs/evaluation.md), and what each dataset, cache and result file holds is in [`eval/README.md`](eval/README.md). Rebuilding the caches (`scripts/build_eval_cache.py build`) downloads the models and is done locally, never in CI.

## Tracing (optional)

Without Langfuse keys tracing is off and everything still runs. To see traces locally (needs Docker):

```bash
cp .env.langfuse.example .env.langfuse        # fill in every value
docker compose --env-file .env.langfuse up -d  # Langfuse on http://localhost:3000
```

Then put the project keys from `.env.langfuse` into `.env` as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. Trace payloads are masked before export (image data, emails, card and phone numbers never leave the process), and with `ENVIRONMENT=production` traces carry no message content at all. See [`docs/data-handling.md`](docs/data-handling.md). With keys set, `evaluate` also mirrors its results to Langfuse as a dataset and one experiment run per candidate; the results file stays the record.

## Project documents

Planning documents live in [`docs/planning/`](docs/planning/):

| File | Contents |
|---|---|
| [`prd.md`](docs/planning/prd.md) | Problem, users, scope, features, success metrics |
| [`architecture.md`](docs/planning/architecture.md) | Pipelines, model layer, storage, API, deployment, cost controls |
| [`engineering-rules.md`](docs/planning/engineering-rules.md) | Engineering rules: libraries, error handling, testing, git |
| [`design.md`](docs/planning/design.md) | UI theme and components |
| [`phases.md`](docs/planning/phases.md) | Delivery phases, definitions of done and the audit reference index |
| [`optimization.md`](docs/planning/optimization.md) | Budgets, token and model usage, performance, scalability, reproducibility, environments and staging |

Architecture decisions are recorded in [`docs/adr/`](docs/adr/). Retrieval results are in [`docs/evaluation.md`](docs/evaluation.md), generated from `eval/results/`. The catalog is described in [`docs/data-card.md`](docs/data-card.md), and the handling of user data in [`docs/data-handling.md`](docs/data-handling.md).
