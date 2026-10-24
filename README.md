# Multi-Agent Food Concierge (MCP)

A multimodal food concierge: a LangGraph agent plans meals and recommends dishes from text and photos using tools served over the Model Context Protocol (MCP). Dietary and allergen constraints are enforced in code by the tools and a deterministic verifier, every run is traced and evaluated with Langfuse, and the whole stack runs at zero cost on free tiers and local models.

> **Status:** under active development. Phases 0–2 are complete: the model layer, and the catalog with allergen safety, storage, search indexes, build flows and a measured vision model ([ADR-0007](docs/adr/0007-vision-model-and-vision-allergen-tags.md)). Phase 3 (retrieval evaluation) is built, and waits on the owner's relevance grades for ADR-0001, the choice of retrieval stack. Phase 4 is well along: the domain services, an MCP server with 6 tools, 4 resources and 2 prompts, and an HTTP server with scoped tokens and rate limits are built; so are the Claude Desktop guide and a container image smoke-tested in CI ([ADR-0002](docs/adr/0002-mcp-first-layering.md) records the layering). Fixes from a Phase 4 audit are under way: rate limits a visitor can't dodge, constraints read from the query's words, and meal totals per diner. The public deploy to a Hugging Face Space is next. See [`docs/planning/phases.md`](docs/planning/phases.md) for the plan, [`CHANGELOG.md`](CHANGELOG.md) for progress and [`docs/baseline-defects.md`](docs/baseline-defects.md) for what is being fixed from the original implementation.

> **Acknowledgement:** this is an independent rebuild of a course exercise on multimodal RAG. No course code, data or images are redistributed; the catalog in this repository is authored for this project.

## Quick start (development)

```bash
python -m venv .venv            # keep it outside cloud-synced folders
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
python -m pip install --upgrade pip setuptools
pip install --require-hashes --no-deps -r requirements-dev.lock   # the locked, hash-checked versions CI uses
pip install --no-deps -e .
cp .env.example .env
```

The same checks as the CI `test` job (`.github/workflows/ci.yml`), in its order:

```bash
# lock files match pyproject.toml (uv is in the dev lock)
uv pip compile pyproject.toml --universal --python-version 3.11 --generate-hashes --no-header -q -o requirements.lock
uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 --generate-hashes --no-header -q -o requirements-dev.lock
git diff --exit-code -- requirements.lock requirements-dev.lock

ruff check . && ruff format --check .
mypy
pytest --cov=food_concierge --cov-report=term-missing --cov-fail-under=100
python scripts/retrieval_gate.py                # committed caches, no model download
pip-audit --require-hashes --disable-pip -r requirements.lock
pip-audit --require-hashes --disable-pip -r requirements-dev.lock
```

CI's `secrets` job also scans for secrets with gitleaks and rejects tracked files over 5 MB.

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
```

The catalog is synthetic by design; see [`docs/data-card.md`](docs/data-card.md) and [`data/raw/README.md`](data/raw/README.md).

## Evaluate retrieval

Offline and deterministic: the evaluation runs from committed caches, with no model download and no network.

```bash
python scripts/retrieval_gate.py                  # the CI gate: no constraint violation, no nDCG@3 drop over 0.02
python -m food_concierge.flows evaluate           # every candidate; writes eval/results/retrieval_<date>_<split>.json
python -m food_concierge.flows evaluate --check   # reproduce the committed results instead of writing (nightly)
python -m food_concierge.flows report             # render docs/evaluation.md from the newest results
```

The latest numbers are in [`docs/evaluation.md`](docs/evaluation.md), and what each dataset, cache and result file holds is in [`eval/README.md`](eval/README.md). Rebuilding the caches (`scripts/build_eval_cache.py build`) downloads the models and is done locally, never in CI.

## Run the MCP server

Build the catalog and indexes first. On stdio, for Claude Desktop or MCP Inspector, the server runs as your own process and every tool is available:

```bash
python -m food_concierge.mcp_server
```

Over HTTP, with health checks, a retrieval-only search endpoint and MCP at `/mcp` (Streamable HTTP, stateless, JSON responses):

```bash
python -m food_concierge.api           # http://127.0.0.1:8000 (API_HOST, API_PORT)
curl -s localhost:8000/v1/search -H 'content-type: application/json' \
  -d '{"query": "dosa", "diet": "vegan", "exclude_allergens": ["peanuts"], "k": 3}'
```

| Tools | Resources | Prompts |
|---|---|---|
| `search_dishes`, `find_similar`, `get_dish`, `check_allergens`, `meal_totals`, `describe_image` | `menu://catalog`, `menu://dishes/{dish_id}`, `menu://dishes/{dish_id}/image`, `menu://policy/allergens` | `plan_meal`, `find_similar_dish` |

Search never returns a dish that breaks the constraints given: it filters in SQL before ranking and re-checks every result, and constraints the query states in words ("peanut-free", "vegan") are applied as well. `meal_totals` takes the number of `diners` and gives the whole order and each diner's share. Over HTTP a request without a token gets the public scope (search and the read-only tools); `describe_image`, which spends the hosted vision quota, needs the `agent` token (`API_TOKEN_AGENT`), sent as `Authorization: Bearer <token>` over HTTPS. Keep tokens in the git-ignored `.env` or platform secrets; each must be at least 32 characters and different from the others, or the server won't start (`python -c "import secrets; print(secrets.token_urlsafe(32))"`). A token that matches none is refused, and wrong guesses spend the caller's rate limit. Callers are rate-limited (`RATE_LIMIT_PER_MIN` each, `RATE_LIMIT_ANONYMOUS_PER_MIN` for everyone without a token, `RATE_LIMIT_GLOBAL_PER_MIN` in all; behind a proxy set `TRUSTED_PROXY_HOPS`, 1 on a Space), and MCP answers only the hosts in `ALLOWED_HOSTS`, so add the deployed host there. Allergen information is derived and comes with a disclaimer; it is not medical advice.

In a container, with the catalog, indexes, embedding model and photo thumbnails built into the image and keys passed at run time:

```bash
docker build -t food-concierge .
docker run --rm -p 7860:7860 -e ALLOWED_HOSTS=127.0.0.1:7860,localhost:7860 -e GROQ_API_KEY food-concierge
python scripts/smoke_deploy.py http://127.0.0.1:7860    # health, search, MCP, scopes; no model calls
```

To connect Claude Desktop or MCP Inspector, see [`docs/mcp.md`](docs/mcp.md) and [`examples/claude_desktop_config.json`](examples/claude_desktop_config.json).

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

Architecture decisions are recorded in [`docs/adr/`](docs/adr/). Retrieval results are in [`docs/evaluation.md`](docs/evaluation.md), generated from `eval/results/`. The catalog is described in [`docs/data-card.md`](docs/data-card.md), and the handling of user data in [`docs/data-handling.md`](docs/data-handling.md). Connecting MCP clients is covered in [`docs/mcp.md`](docs/mcp.md).
