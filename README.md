# Multi-Agent Food Concierge (MCP)

A multimodal food concierge: a LangGraph agent plans meals and recommends dishes from text and photos using tools served over the Model Context Protocol (MCP). Dietary and allergen constraints are enforced in code by the tools and a deterministic verifier, every run is traced and evaluated with Langfuse, and the whole stack runs at zero cost on free tiers and local models.

> **Status:** under active development. See [`docs/planning/phases.md`](docs/planning/phases.md) for the plan and `docs/baseline-defects.md` for what is being fixed from the original implementation.

> **Acknowledgement:** this is an independent rebuild of a course exercise on multimodal RAG. No course code, data or images are redistributed; the catalog in this repository is authored for this project.

## Quick start (development)

```bash
python -m venv .venv            # keep it outside cloud-synced folders
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
cp .env.example .env
pytest
```

## Project documents

Planning documents live in [`docs/planning/`](docs/planning/):

| File | Contents |
|---|---|
| [`prd.md`](docs/planning/prd.md) | Problem, users, scope, features, success metrics |
| [`architecture.md`](docs/planning/architecture.md) | Pipelines, model layer, storage, API, deployment, cost controls |
| [`engineering-rules.md`](docs/planning/engineering-rules.md) | Engineering rules: libraries, error handling, testing, git |
| [`design.md`](docs/planning/design.md) | UI theme and components |
| [`phases.md`](docs/planning/phases.md) | Delivery phases and definitions of done |
