# Baseline defects

Defects found when reviewing the original course implementation, and the phase of this project that addresses each one. The original code and data are not redistributed here, so evidence is described by behaviour. Phases refer to `docs/planning/phases.md` (revision 3).

| # | Defect | Evidence | Impact | Addressed in |
|---|---|---|---|---|
| 1 | Query rewrite (HyDE) computed but never used | The rewritten query is built, then similarity search runs on the raw input | The intended retrieval improvement never reaches users, while every query pays for the extra call | P3 (rewrite measured as an ablation), P5 |
| 2 | Up to 11 sequential LLM calls per query | Image description + rewrite + router + 5 relevance checks + 3 summaries | Slow and wasteful | P5 (bounded agent with a fast path) |
| 3 | Raw `json.loads` on model output | Router output parsed without validation or fallback | Any prose or code fence around the JSON crashes the app | P1 (validated structured output), P5 |
| 4 | Relevance parser drops valid answers | `.strip(' ') == 'yes'` rejects `"Yes."` and `"Yes\n"` | Relevant dishes silently discarded | P5 (LLM relevance check replaced by tool-side filters + deterministic verifier) |
| 5 | Empty reply when nothing passes the relevance check | No fallback branch | User sees a blank answer | P5 (safe fallback), P7 (empty state) |
| 6 | Input box defaults to a single space | The "has input" guard is always true | Empty submissions run the full pipeline | P7 (chat input + API validation) |
| 7 | Uploaded image re-described and re-appended on every send | The image stays in the uploader across sends | Duplicate vision calls; polluted queries | P2/P4 (description cache by hash), P7 (uploader cleared) |
| 8 | Images labelled `image/jpeg` regardless of format | Dataset images are PNG (RGBA) | Providers may reject mismatched media types | P4 (vision service validates and re-encodes) |
| 9 | Models and index reloaded on every UI rerun | Clients and index created at module level in the Streamlit script | Latency on every interaction | P7 (loaded once in the API lifespan) |
| 10 | Allergen safety left to an LLM Yes/No check over incomplete data | 20 of 50 items had empty allergen labels, including dishes that clearly contain dairy, gluten, shellfish, soy or tree nuts; 14 of 50 image descriptions named allergens absent from the metadata | Users excluding an allergen could be shown dishes that contain it | P2 (three-source allergen tags), P4 (constraints enforced in tools), P5 (verifier) |
| 11 | Pickled docstore loaded with `allow_dangerous_deserialization=True` | LangChain FAISS store with a pickle file | Arbitrary code execution if the file is replaced | P2 (SQLite + native FAISS files + manifest) |
| 12 | Retired model ID and deprecated LangChain classes | Claude 3 Sonnet model ID; `langchain_community.BedrockChat` | The app fails at the first model call | P1 (provider layer) |
| 13 | Inconsistent data encodings | `serves` mixes quoted ranges and integers; `cuisine` mixes cuisines and categories; warning strings vary in format | Broken filters and display | P2 (normalisation) |
| 14 | Redundant data | A 47 MB CSV of base64 images duplicated the image files, and the whole data folder was duplicated (~125 MB of 159 MB redundant) | Repository bloat | P0 (derived artefacts rebuilt, not stored) |
| 15 | README runs a file that doesn't exist | Documented start command pointed at a missing script | Setup fails as documented | P10 (README) |
| 16 | No tests, evaluation, configuration or version control | — | Quality claims unverifiable | P0–P10 |
