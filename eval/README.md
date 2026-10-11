# eval

Labelled datasets, evaluation runners and CI quality gates (Phases 3 and 6): retrieval metrics (nDCG@3, Success@3, Violation@3), agent metrics, red-team attack success rate and Responsible AI metrics, compared with a re-implementation of the original approach.

What is here so far:

| File | What it holds | Made by |
|---|---|---|
| `datasets/model_profile.yaml` | 20 prompts per provider and role for the model profile | `scripts/model_profile.py` |
| `results/model_profile_2026-10-01*.json`, `results/smoke_*.json` | Phase 1 model profile and live smoke run (ADR-0006) | `scripts/model_profile.py`, `scripts/smoke_live.py` |
| `datasets/vision_sample.yaml`, `datasets/vision_sample_groq.yaml` | 30 seeded catalog photos with owner-reviewed scores per description (dish type, ingredients seen, allergen terms not visible) | `scripts/vision_measurement.py sample` |
| `datasets/vision_descriptions_groq.jsonl` | Groq `qwen/qwen3.8-27b` descriptions of the same 30 photos, with latency and tokens | `scripts/vision_measurement.py describe --provider groq` |
| `results/vision_measurement*_2026-10-09.json` | Phase 2 vision measurement, with Wilson 95% intervals (ADR-0007) | `scripts/vision_measurement.py report` |
| `datasets/retrieval.yaml` | Phase 3 text queries (six types), gold filters for constrained queries, target dishes, and the owner's grades once imported; dev/test split locked by SHA-256 | drafted, then `scripts/label_queries.py import` and `split` |
| `datasets/labelling-guide.md` | How the owner grades dishes 0-3, with worked examples | — |
| `datasets/image_queries.jsonl` | Phase 3 image queries: one per described catalog photo, with its seeded transforms, the altered image's SHA-256, the dev/test side (split by dish) and three description rungs (L0 named and unaltered, local model; L1 name-free and unaltered; L2 name-free and altered, both Groq) | `scripts/image_queries.py build` and `describe` |
| `caches/original_verdicts.jsonl` | Baseline C0, the original approach re-implemented: each query's raw request with constraints as words, its unfiltered dense top 5 and Groq `openai/gpt-oss-20b`'s raw Yes/No reply per dish, scored offline with the original's strict parser and a fixed one | `scripts/original_baseline.py collect`, then `replay` |
| `caches/retrieval/` | Retrieval caches for the PR gate (under 2 MB, no pickle): document vectors, query vectors with and without the BGE prefix, the full query × document rerank score matrix, and a manifest binding them to the models, catalog data, document-text version and query texts | `scripts/build_eval_cache.py build` (local); `check` runs nightly against fresh models |
| `gates/retrieval_baseline.json` | The gated configurations and their baseline nDCG@3 (none until the owner's grades exist), with the reason it last changed; the gate fails on any Violation@3 or a drop over 0.02 | `scripts/retrieval_gate.py`, `--update --reason` |
| `results/retrieval_<date>_<split>.json` | Every candidate (C0 strict and fixed, C1-C5 with and without de-duplication, the gated configurations) with per-query rankings and metrics, bootstrap intervals overall and per slice, the ladder comparisons, the gate outcome and the run record (`all` until the queries are split) | `python -m food_concierge.flows evaluate`; `--check` reproduces it (nightly) |
