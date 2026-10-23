# 0007. Vision model for user photos, and what vision allergen tags may do

- Status: accepted
- Date: 2026-10-09
- Requirements: PRD §6 (F2 image input, F5 three-source allergen tags, F27 data governance), §7 (latency p95 ≤ 8 s, zero cost, privacy)

## Context

Two questions were left open by ADR-0006 and the Phase 2 plan (item 7):

1. **Which model describes a user's photo at runtime.** Catalog photos are described offline by local `qwen2.5vl:7b` on the owner's GPU. The deployment has no GPU, so user photos need a hosted model. ADR-0006 found no Groq vision model by name, and Gemini's free tier lets Google use prompts and replies, with human review (`docs/data-handling.md`).
2. **Whether vision `may_contain` tags should filter search.** The tagger turns allergen terms in a photo's name-free description into `may_contain` tags. Today `CatalogFilters` hides a dish when an excluded allergen is in `contains` or `may_contain`.

**Evidence.** Both models described the same seeded sample of 30 catalog photos (Wikimedia Commons, no user data), with the same name-free prompt (version 1), at 1024 px and without metadata. Each description was scored against its photo by the rules in `scripts/vision_measurement.py`: dish type, whether each named ingredient is visible, and allergen terms whose food the photo gives no evidence of. Intervals are Wilson 95%.

| | Local `qwen2.5vl:7b` (Ollama, RTX 4060 Laptop) | Groq `qwen/qwen3.8-27b` |
|---|---|---|
| Results file | `eval/results/vision_measurement_2026-10-09.json` | `eval/results/vision_measurement_groq_2026-10-09.json` |
| Scores reviewed by the owner | yes | yes |
| Dish type correct | 23/30 (77%) [59%, 88%] | 22/30 (73%) [56%, 86%] |
| Dish type correct or partial | 29/30 (97%) [83%, 99%] | 30/30 (100%) [89%, 100%] |
| Named ingredients visible in the photo | 82/98 (84%) [75%, 90%] | 103/115 (90%) [83%, 94%] |
| Descriptions naming an unseen ingredient | 13/30 (43%) | 11/30 (37%) |
| Allergen terms with no evidence in the photo | 4/25 (16%) [6%, 35%] | 3/24 (12%) [4%, 31%] |
| Descriptions with a false allergen term | 3/30 (10%) | 3/30 (10%) |
| Latency p50 / p95 | 3,835 / 4,804 ms (all 127 photos) | 712 / 911 ms (30 photos) |
| Tokens per photo | n/a (local) | p50 1,921, max 1,969 |

Observations:
- **Accuracy is level.** Every interval overlaps. Groq names more ingredients, and more of them are really there. It got right three photos where the local model was misled: the whole aubergines of h04 (local: "whole fish", a false fish tag on a vegetarian dish), the egg ribbons of w09 (local: cabbage) and the sev on g08 (local: rice).
- **Both models misname proteins.** Groq called hilsa steaks "bone-in meat" (b01) and a vegetarian undhiyu's muthia "meatballs" (g04). A description is evidence, not a diet label.
- **False allergen terms have three causes, in both models:** hedges ("fried spices or nuts"), negations ("no visible fruit or nuts" tags peanuts and tree nuts, because the lexicon has no negation handling) and a wrong name for the grain (a millet rotla called "bread" tags gluten and wheat).
- **Groq is fast, but the tokens-a-minute limit binds.** An image costs about 1,900 tokens against Groq's 8,000 tokens a minute per model, so three or four photos a minute at most. The run paced itself to 8,000 by the tokens each reply reported, and still got 12 rate-limit replies (429) in 30 photos. Groq appears to count more per call than the reply reports. The first two attempts at the run also stalled: three calls timed out at 30 s on the first photo before the run went through. None of the 30 kept calls was slower than the 3 s production timeout.
- **Gemini vision was not measured.** Its free-tier data terms rule it out for user photos whatever its accuracy (Option 3).

**Vision tags on the whole catalog** (`scripts/allergen_report.py`, lexicon version 2): 211 `may_contain` tags on 131 photographed dishes, 150 of them in the ground truth. Counted as distinct (dish, allergen) pairs not already in `contains`, which is what the filter sees, vision adds **47 tags on 29 dishes, and none is in the ground truth**: gluten 10, wheat 10, peanuts 10, milk 5, soy 4, fish 3, tree nuts 2, eggs 2, sesame 1. Label and lexicon already reach 100% recall here, so vision gains nothing, and today each of those 47 tags hides a dish from someone who excludes that allergen.

## Options

**Runtime vision model**

1. **Groq `qwen/qwen3.8-27b`.** Matches the local model on every score, p95 under 1 s, and Groq neither trains on inputs nor keeps inference data by default. Against: about 1,900 tokens a photo, so a few photos a minute on the free tier; it was found by a probe and is not listed as a vision model, so it may change without notice.
2. **Groq first, Gemini on fallback.** More availability, but a user's photo would reach Google under free-tier terms exactly when Groq is busy, which a user cannot predict or opt out of.
3. **Gemini first.** Not measured, and the data terms apply to every photo.
4. **No hosted model: text only.** Free and private, but drops F2 (image input), a must-have.

**Vision `may_contain` tags**

A. **Keep filtering on them.** The most cautious reading of "may contain", but on this catalog it costs 47 false exclusions on 29 dishes for no recall.
B. **Show them, don't filter on them.** Hard filters use `contains` and `unverified`. Vision tags stay in the catalog, appear as "may contain" chips and in `check_allergens`, and the safety gate keeps checking label and lexicon recall.
C. **Drop the vision source.** Simplest, but loses the warning on real menus with gaps, where a photo can show what an ingredient list leaves out (audit A1), and the evidence for revisiting this.

## Decision

**Option 1 and Option B.**

- `GROQ_VISION_MODEL=qwen/qwen3.8-27b` becomes the default in `config.py`. `GEMINI_VISION_MODEL` stays blank, so the vision chain is Groq alone. When Groq fails or is rate-limited, the request does not fall back to another provider. The user is asked to describe the dish in words, and text search serves it (P4/P7).
- Local `qwen2.5vl:7b` stays the model for catalog descriptions, run offline by `scripts/describe_photos.py`. Catalog descriptions are tracked, so neither the build nor CI needs a vision model.
- Vision `may_contain` tags no longer hide dishes. `CatalogFilters` excludes a dish when an excluded allergen is in `contains`, or when the dish is `unverified` and unverified dishes are not included. `may_contain` tags are still stored, returned by `check_allergens` and shown as chips. The deterministic verifier (P5) treats them as a warning to show, not a reason to refuse.
- The safety gate (`evaluate` in `ingestion/allergens.py`) now checks recall of `contains`, the tags search filters on, so a vision tag can no longer pass the gate for an allergen search would ignore. It passes on all 150 dishes.

Why:
- The hosted model matches the local one, so user photos can be described as well as catalog photos are, five times faster, without sending them to a provider that trains on them.
- Allergen safety rests on the label, the lexicon and the `unverified` state, which together pass the 100%-recall gate. Vision has not caught a single allergen they missed. A tag that is wrong 47 times out of 47 here is a warning, not a filter.

## Consequences

- **Photo throughput is low.** At about 1,900 tokens a photo, Groq's free tier serves three or four photos a minute per model. P4 rate limits must budget photos separately from chat. The usage ledger caps tokens per provider, but Groq limits per model, so the ledger should key token caps by model before vision and chat share a process. The pacing gap (429s while under the reported-token cap) means the ledger should stay below Groq's figure, not at it.
- **A daily token cap too (found 2026-10-11, Phase 3).** Groq also caps `qwen/qwen3.8-27b` at about 200,000 tokens in a rolling 24 hours. At about 1,900 tokens per name-free description, that is roughly 100 photos a day, which binds before the 900-calls-a-day cap. (Corrected 2026-10-12 from a first estimate of about 1,430 tokens and 140 photos. The recorded counts have a median of about 1,920; some rows are undercounted, #74.) The Phase 3 image job therefore spans several days. The P4 photo budget needs a daily token limit as well as a per-minute one. (Done in #49: the ledger keys caps by model, caps tokens per rolling day, and stops photos at 90 a UTC day.)
- **No vision fallback.** A Groq outage turns photo search into "describe it in words". That is a visible degradation, and better than a silent change of data terms.
- **Descriptions can misname proteins** ("meatballs" for muthia). Image queries search with the description, so a vegetarian dish can rank below meat dishes for its own photo, but diet and allergen filters come from catalog data, never from a description. P3 measures image-query retrieval with these descriptions.
- **Lexicon follow-ups:** negations ("no nuts") and hedged alternatives ("spices or nuts") produce false terms in both models. These now cost a chip rather than a hidden dish, and are worth fixing when the lexicon next changes. (Negations done in #54: descriptions skip terms they deny, so the Groq sample's false terms fell from 3 of 24 to 2 of 23; the local sample has no negated terms. Hedged alternatives stay: 12 of the 16 hedged terms on the two samples are right, "cream or yogurt" among them, so skipping them would drop more true warnings than false ones.)
- **Revisit** when the catalog includes real menus whose ingredient lists have gaps (vision may then catch allergens, and Option A becomes worth measuring again), when Groq lists, renames or retires vision models, when Gemini's free-tier terms change, or when a paid tier is allowed. Any new vision model must pass `scripts/vision_measurement.py describe` and an owner-reviewed sample first.

## Update 2026-10-23: image size (#53)

The 30-photo sample was described again at 768 px (`eval/datasets/vision_descriptions_groq_768.jsonl`, `scripts/vision_measurement.py describe --provider groq --px 768`) to test the expected saving of about 45% (optimization.md §2.4).

| Shape sent | Tokens at 1,024 px | Tokens at 768 px | Photos |
|---|---|---|---|
| Wide (about 16:9 to 2:1) | about 900 | about 900 | 4 |
| Square or near square | about 1,420 | about 1,420 | 6 |
| 4:3 or 3:2 | about 1,930 | about 1,930 | 20 |

Every photo's count stayed within 2% (median 1,922 at 1,024 px, 1,924 at 768 px). The endpoint evidently rescales each image to a fixed size for its aspect ratio, so the count follows the shape and not the pixels sent.
- **The size stays at 1,024 px** (`DESCRIBE_PX`, `VISION_IMAGE_PX`). A smaller image saves no tokens and can only lose detail, so the 768 px descriptions were not scored; the size decision does not depend on them.
- **Photo throughput** stays about 100 a day on the daily token cap, and is set by the photos' shapes, not by downscaling.
- Wide images cost about half as much as 4:3 ones, but cropping a photo to save tokens cuts out part of the dish; it would need its own accuracy test.
- Re-measure if the vision model or the endpoint changes, since the rescaling is the provider's and is not documented.
