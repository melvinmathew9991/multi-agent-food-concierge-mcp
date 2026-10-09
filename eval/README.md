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
