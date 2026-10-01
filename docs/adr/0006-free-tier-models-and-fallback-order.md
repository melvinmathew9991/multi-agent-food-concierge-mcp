# 0006. Free-tier models and fallback order

- Status: accepted
- Date: 2026-10-01
- Requirements: PRD §6 (model layer, zero cost), §7 (latency p95 ≤ 8 s, reliability, privacy)

## Context

Every later phase calls one model layer: Groq first, then Gemini through their OpenAI-compatible endpoints, with Ollama locally (`models/router.py`). The router needs model names, and P3–P6 rely on those models producing valid typed output and correct tool calls within the 8 s request deadline. Model names were left blank until measured, because provider listings are not proof of access. On 2026-09-30, Gemini listed 2.5 models that return 404 for new accounts, and it did so again on 2026-10-01.

**Evidence.** `scripts/model_profile.py` ran the 20 fixed prompts in `eval/datasets/model_profile.yaml`:
- 10 constraint extractions and 5 routing decisions, as structured output with one repair;
- 5 tool calls.

It used the production code path and production settings (temperature 0 and seed 7 for the router role, `reasoning_effort` low, 400 output tokens). Attempt timeouts were relaxed to 30 s, and calls slower than the production timeout were counted instead. Requests were paced under the free-tier per-minute caps.

There were two full runs on 2026-10-01:
- `eval/results/model_profile_2026-10-01.json` is the source of all numbers below.
- `…_run1.json` is the first run. Its latency is invalid because the pacing wait was timed. It is used only to check run-to-run agreement.

Intervals are Wilson 95%. With n = 20 per run they are wide, so the decision rests on clear gaps, not small differences.

**Probe** (one real call per listed candidate):
- 9 of 11 candidates answered.
- `gemini-2.5-flash-lite` returned 404 (`provider_model_unavailable`).
- `gemma-4-31b-it` returned 400, "Thinking level is not supported for this model": it rejects `reasoning_effort`. Without it, Gemma writes its reasoning into the reply.
- `gemini-flash-lite-latest` answered but is a moving alias, so it was not profiled.
- Groq lists no vision model.

**Profile** (run 2; "agree" is per-item correctness matching run 1):

| Provider | Model | Valid typed output (1st try) | Correct, all 20 | Tool calls | p50 / p95 latency | Over production timeout | Mean tokens in / out | Agree |
|---|---|---|---|---|---|---|---|---|
| groq | openai/gpt-oss-20b | 15/15 | **20/20** [0.84, 1] | 5/5 | 544 / 731 ms | 0 (3 s) | 297 / 59 | 20/20 |
| groq | qwen/qwen3.8-27b | 15/15 | **20/20** [0.84, 1] | 5/5 | 339 / 396 ms | 0 (3 s) | 586 / 64 | 20/20 |
| groq | openai/gpt-oss-120b | 15/15 | 19/20 | 5/5 | 679 / 5433 ms | 2 (3 s) | 297 / 71 | 20/20 |
| gemini | gemini-3.5-flash-lite | 15/15 | **20/20** [0.84, 1] | 5/5 | 1089 / 1248 ms | 1 (5 s) | 339 / 34 | 19/20 |
| gemini | gemini-3.1-flash-lite | 12/15 | 16/20 | 4/5 | 4157 / 7850 ms | 2 (5 s) | 260 / 25 | 15/20 |
| gemini | gemini-3.5-flash | 0/15 | 0/20 | 0/5 | none | n/a | n/a | 4/20 |
| gemini | gemini-3.8-flash | 1/15 | 1/20 | 0/5 | none | n/a | n/a | 17/20 |
| ollama | llama3.1:8b | 14/15 | 8/20 [0.22, 0.61] | 3/5 | 1002 / 3584 ms | 0 (4 s) | 142 / 33 | 20/20 |
| ollama | llama3.2:3b | 15/15 | 8/20 [0.22, 0.61] | 4/5 | 404 / 940 ms | 0 (4 s) | 151 / 38 | 20/20 |

Observations:
- **gemini-3.1-flash-lite** returned 503 (overloaded) on 4 of 21 requests in run 2 and 3 of 21 in run 1.
- **gemini-3.5-flash and gemini-3.8-flash** returned 429 on 20 of 21 and 17 of 21 requests in run 2. Run 1 had already shown 429s and 503s for 3.8-flash at 8 requests a minute. Their free quotas run out after a few dozen calls, so they cannot serve as a fallback.
- **The only error shared across models** is ex08, "a light veg breakfast, around 300 calories". Two models in one or both runs answered `courses: 1`, inferring a course count the message does not state. The label was not changed after seeing results. A mild over-inference still counts as wrong, because invented constraints are the failure the verifier exists to catch.
- **Ollama models return valid JSON with wrong content.** They invent constraints, for example `diet: vegetarian` for "Non-veg is fine", `max_calories` copied from the budget, and every allergen listed for "lactose intolerant". They also misroute most messages and pass numbers and lists to tools as strings. Typed output enforces shape, not meaning.
- **Groq rate-limit headers** show 1,000 requests a day and **8,000 tokens a minute per model**. At about 300 input tokens a call, gpt-oss-20b fits about 25 calls a minute. qwen3.8-27b uses twice the input tokens for the same prompts, so it fits about half that. The token limit, not the request limit, is what binds.
- **Gemini** returns no rate-limit headers, so its quotas are known only from 429s.

## Options

1. **Groq gpt-oss-20b, then Gemini 3.5 Flash-Lite.** 40/40 correct across both runs for gpt-oss-20b [0.91, 1] and 39/40 for Flash-Lite [0.87, 1.00]. Both p95 latencies fit inside their attempt timeouts with room to spare, and neither had a single 429 or 503 in 42 requests each.
2. **Groq qwen3.8-27b, then Gemini 3.5 Flash-Lite.** Equally accurate and about half the latency. But it uses double the input tokens, which halves headroom under Groq's 8,000 tokens-a-minute limit, and P5 makes several model calls per user request.
3. **Groq gpt-oss-120b first.** No more accurate on this set, a p95 above the 3 s Groq timeout, and the same token cost.
4. **Gemini first.** Slower: p50 about 1.1 s against about 0.5 s. Its quotas are opaque (no headers) and its larger models ran out in minutes.
5. **Ollama as a hosted-model fallback.** Free and private, but 8/20 correct. It would turn an outage into confidently wrong constraints, which is worse than a clear error.

## Decision

Option 1:
- `GROQ_CHAT_MODEL=openai/gpt-oss-20b` and `GEMINI_CHAT_MODEL=gemini-3.5-flash-lite` are the defaults in `config.py`.
- The fallback order stays `CHAT_PROVIDER=groq`, then `FALLBACK_PROVIDERS=gemini`.
- The router and judge roles share the chat model. Judging is measured in P6 before a separate judge model is chosen.

Timeouts and caps stay as set: Groq 3 s and Gemini 5 s per attempt, 8 s across the chain; 900 Groq and 200 Gemini calls a day; 25 and 8 a minute.
- The measured p95s (0.73 s and 1.25 s) fit well inside the timeouts.
- The Groq daily cap stays under the 1,000-request limit the headers report.
- Gemini's daily quota is unknown, so its cap stays conservative.

Things this decision does not settle:
- **Vision** is not decided. Groq has none, and no photos were available for the profile. `GEMINI_VISION_MODEL` stays blank, and local `qwen2.5vl:7b` remains the development path. Gemini 3.5 Flash-Lite vision is measured on own photos before it is configured.
- **Ollama** stays a local, offline development provider and is not in the default chain.
- **Safety models for P5:** Groq offers `openai/gpt-oss-safeguard-20b` (a policy-based safety classifier) and `meta-llama/llama-prompt-guard-2-22m` and `-86m` (prompt-injection detection). It offers no Llama Guard. None was exercised here.

The capability table needs no change. Tool calling worked for Groq and Gemini (15/15 valid on the first try for the chosen models), and Groq accepted the seed. Gemini's seed support was not tested and stays off.

## Consequences

- **The router works out of the box** with only API keys set: no model names to configure, and the chain verifiably meets the deadline.
- **`reasoning_effort` support is per model, not per provider.** A model that rejects it, such as Gemma, gets a 400 on every call, and a 400 does not fall back. Any model change must pass `scripts/model_profile.py --probe` first, and a full profile run before it becomes a default.
- **Groq's tokens-a-minute limit** will bind before its request caps once agents make several calls per request. P4/P5 budget work should add a token-rate limit, not only call counts.
- **Gemini quotas are opaque,** so 429s from the fallback must be monitored (Langfuse error and fallback rate). If they appear in normal use, re-profile and reconsider.
- **Revisit** when a provider retires or renames a chosen model, when the free tiers change, when the P3/P5 prompts differ materially from this set, or when vision is measured.
