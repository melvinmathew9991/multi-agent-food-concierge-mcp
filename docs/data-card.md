# Data card: restaurant catalog

The catalog the concierge searches and plans meals from. It was written for this project in Phase 2, and it is **synthetic by design**: metrics computed on it describe how the system behaves, not how accurate it would be on real menus.

Source files and their column-level documentation live in `data/raw/` (see `data/raw/README.md`). This card summarises what the data is, where it came from, what is deliberately imperfect, and how a build is traced back to it.

## Contents

| | |
|---|---|
| Restaurants | 15, all fictional |
| Dishes | 150 (a dish at a restaurant); 6 dishes are served at two restaurants |
| Cuisines | 67% Indian (North and South Indian, Punjabi, Mughlai, Hyderabadi, Bengali, Gujarati, Kerala, Maharashtrian, street food, Indo-Chinese); the rest Italian, Continental, American, Mediterranean, Middle Eastern, Mexican, Thai, Japanese, Korean, Chinese |
| Categories | main 45, dessert 24, snack 17, breakfast 13, starter 10, beverage 10, bread 8, rice 8, soup 5, salad 5, side 4, thali 1 |
| Diets | vegetarian (lacto-ovo) 76, vegan 41, non-vegetarian 33; 17 dishes contain egg |
| Prices | ₹25 to ₹520 per dish |
| Calories | 50 to 830 kcal per serving |
| Allergens | 15 keys: EU-14 ∪ US Big-9 |

## Sources

- **Text** (restaurant names, dish names, descriptions, ingredients, prices, ratings, nutrition, labels): authored for this project in three batches, 2026-10-07. No text was copied from a real menu or from the course material this project rebuilds.
- **Restaurant names**: invented, and each searched on the web to make sure it doesn't match a real business. Two candidate names were changed because they were close to real restaurants.
- **Photos**: Wikimedia Commons only, under CC0, public domain, CC BY or CC BY-SA. Files with NC, ND, GFDL-only, non-free or unknown licences, or with restrictions such as trademarks or personality rights, are refused. The owner approves one photo per dish, or marks the dish as having none. Each approved file is recorded in `data/raw/attributions.csv` with its author, licence, licence URL and SHA-256. Files are not redistributed in the repository: they are fetched and checked against their hash at build time. **Status:** the owner reviewed all 150 dishes: 131 have a photo (127 distinct files, as four photos serve the same dish at two restaurants) and 19 have none. Licences: CC BY-SA 102, CC BY 13, CC0 15, public domain 1 (`attributions.csv` has each version).

## What is synthetic, and how

| Field | Nature |
|---|---|
| Restaurants, ratings, review counts | Invented |
| Prices | Plausible INR prices for the dish type, invented |
| Nutrition | Plausible per-serving values, not measurements. Calories are consistent with 4P + 4C + 9F (the loader warns beyond 25%; the catalog has no warnings) |
| Restaurant labels (`label_allergens`) | **Deliberately imperfect**: 47 of 150 dishes (31%) are unlabelled, some labels miss allergens, and 4 dishes are declared allergen-free (some wrongly). Real menus are like this, and the system must not trust labels alone |
| Ground truth (`true_allergens`) | Hand-checked allergens of the full recipe, including what is inside composite ingredients. Reviewed by the owner before each batch merged. Used **only by tests and evaluation**, never stored in the built catalog |

The recipe assumptions behind the ground truth (for example: hing and soy sauce contain wheat, kimchi contains fish and shrimp, coconut is not a tree nut) are listed in `data/raw/README.md`.

## Derived data

- **Allergen tags** come from three sources: the restaurant label and an ingredient lexicon (`data/lexicon/allergens.yaml`, version 1) give `contains`; the name-free description of the dish's photo gives `may_contain`. Dishes with an opaque ingredient (a masala mix, a chutney, an unnamed sauce) are `unverified`: 25 of 150 (17%).
- **Photo descriptions**: `qwen2.5vl:7b`, run locally through Ollama, wrote two descriptions of each of the 127 photos (prompt version 1): a name-free one and one told the dish's name. They are cached in `data/processed/image_descriptions.jsonl`, keyed by image SHA-256, so the build never needs a vision model. Median latency was 3.8 s per description on an RTX 4060 Laptop GPU (8 GB).
- **What vision adds here**: 212 `may_contain` tags on 131 dishes. On this catalog it adds no recall, because label and lexicon already flag every ground-truth allergen; the 59 tags that only vision gives are all false against the ground truth, mostly "bread" (gluten, wheat) and "nuts" (peanuts, tree nuts). A Commons photo shows a version of the dish, so some of these may be true of the photo but not of the recipe. Vision is kept as a cautious third source for menus whose ingredient lists are incomplete; its cost in excluded dishes is measured in the vision measurement (ADR-0007).
- **Safety gate**: every ground-truth allergen of every dish is tagged, or the dish is unverified (100% recall; checked by the tests on every build). Per-allergen precision is reported by `scripts/allergen_report.py`. These numbers are in-sample: the lexicon was written against this catalog.

## Lineage

1. `data/raw/*.csv`, `data/lexicon/allergens.yaml` and `data/processed/image_descriptions.jsonl` are hashed together (`data_sha256`).
2. `python -m food_concierge.flows ingest` builds `data/processed/catalog.db` and records that hash, the document-text version and the lexicon version in its `meta` table.
3. `python -m food_concierge.flows index` builds `data/processed/indexes/` and records the same hash, the embedder fingerprint and the hash of every index file in `manifest.json`. An index built from other data or another embedder refuses to load.
4. Evaluation runs (Phase 3 onwards) record the data hash and index manifest they ran against.

Re-running the build with unchanged inputs does nothing.

## Known limitations

- **Not medical-grade.** Allergen information is derived from authored recipes and a lexicon; the app shows it with a disclaimer.
- **Nutrition is approximate**, and serving sizes are nominal.
- **Indian-centric.** Two thirds of the dishes are Indian; other cuisines are thinly represented (one restaurant each).
- **Small.** 150 dishes limit the statistical power of every evaluation built on them.
- **One author.** The text and the ground truth were written by the same project they test; the owner's review is the safeguard.
- **Photos show a dish, not the dish.** A Commons photo of "masala dosa" is not a photo of this restaurant's masala dosa.
