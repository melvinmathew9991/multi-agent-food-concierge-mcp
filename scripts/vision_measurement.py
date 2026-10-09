"""Measure the vision model's name-free photo descriptions (Phase 2 plan, item 7). Offline: reads the tracked cache.

1. Draw the sample of 30 photos, seeded, and build a local review page (thumbnail beside description and scores):

       python scripts/vision_measurement.py sample

   This writes eval/datasets/vision_sample.yaml and vision_review.html in the photo cache. Re-running keeps scores
   already written and only rebuilds the page.

2. Score each entry in the YAML by looking at its photo; the owner reviews every score and sets ``reviewed: true``.

3. Report:

       python scripts/vision_measurement.py report

   This prints the summary and writes eval/results/vision_measurement_<date>.json. Every entry must be scored.

A hosted model is measured on the same 30 photos, for the runtime vision default (ADR-0007). It is live, so never
run in CI; the photos are the catalog's Commons files, never user data:

       python scripts/vision_measurement.py describe --provider groq
       python scripts/vision_measurement.py sample --provider groq
       python scripts/vision_measurement.py report --provider groq

``describe`` sends each sampled photo through the production code path (``describe`` in ingestion/descriptions.py:
the same prompt, 1024 px, no metadata) and writes eval/datasets/vision_descriptions_<provider>.jsonl, saving after
each photo. Attempt timeouts are relaxed to 30 s, and calls slower than the production timeout are counted. Calls
are paced under the provider's tokens-a-minute cap. ``sample`` and ``report`` then work as above, on
eval/datasets/vision_sample_<provider>.yaml.

Scores per description:

- ``dish_type``: ``correct`` (the kind of dish is right), ``partial`` (the right family, wrong specifics: "a soup"
  for a curry, "a paste" for a halwa) or ``wrong``.
- ``seen`` / ``not_seen``: each ingredient the description names, split by whether the photo shows it. Hedged
  guesses count ("possibly coconut milk"); "X or Y" counts as seen when the photo shows either.
- ``allergen_terms`` (filled by the script): each lexicon term the tagger finds in the description, and the
  allergens it implies. ``not_visible`` lists the terms whose allergen the photo gives no evidence of: false
  ``may_contain`` tags. A wrong name for the right allergen ("almonds" for cashews) is not listed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from food_concierge.config import REPO_ROOT, ProviderName, Settings, get_settings
from food_concierge.errors import ProviderRateLimitedError, ProviderTimeoutError
from food_concierge.ingestion.allergens import Lexicon, load_lexicon
from food_concierge.ingestion.descriptions import (
    DESCRIPTIONS_FILE,
    PROMPT_VERSION,
    DescriptionKey,
    ImageDescription,
    Variant,
    describe,
    prompt_for,
    read_descriptions,
)
from food_concierge.ingestion.loader import load_catalog
from food_concierge.ingestion.photos import CommonsClient, ensure_cached
from food_concierge.ingestion.taxonomy import with_implied
from food_concierge.models.router import build_chat_model
from food_concierge.models.usage import get_usage_ledger

DATASET = REPO_ROOT / "eval" / "datasets" / "vision_sample.yaml"
RESULTS = REPO_ROOT / "eval" / "results"
SAMPLE_SIZE = 30
SEED = 20261009
DISH_TYPES = ("correct", "partial", "wrong")
LOCAL = "ollama"  # the tracked description cache, written by scripts/describe_photos.py
# Hosted vision models to measure. Found by a probe on 2026-10-09: Groq names no vision model, but qwen3.8-27b
# accepts images.
HOSTED_MODELS: dict[str, str] = {"groq": "qwen/qwen3.8-27b"}
HOSTED_TIMEOUT_S = 30.0
RATE_LIMIT_WAIT_S = 60.0
SCORE_FIELDS = ("dish_type", "seen", "not_seen", "not_visible", "note")
HEADER = """\
# Hand-checked sample of name-free photo descriptions (Phase 2 plan, item 7), scored against each photo.
# Drawn by `python scripts/vision_measurement.py sample`; scoring rules are in that script's docstring.
# Drafted by looking at each thumbnail; `reviewed: true` records that the owner has checked every score.
"""


class HostedDescription(ImageDescription):
    """A hosted model's description, with the tokens its reply reported: images dominate the token budget."""

    tokens: int = Field(default=0, ge=0)
    failed_attempts: int = Field(default=0, ge=0)  # timeouts and 429s before the reply that was kept


def dataset_path(provider: str) -> Path:
    return DATASET if provider == LOCAL else DATASET.with_name(f"vision_sample_{provider}.yaml")


def hosted_cache_path(provider: str) -> Path:
    return DATASET.with_name(f"vision_descriptions_{provider}.jsonl")


def read_hosted(path: Path) -> dict[DescriptionKey, HostedDescription]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    found = (HostedDescription.model_validate_json(line) for line in lines)
    return {(d.sha256, d.variant): d for d in found}


def write_hosted(path: Path, cache: Mapping[DescriptionKey, HostedDescription]) -> None:
    """Sorted by hash, LF line ends, so the file diffs cleanly."""
    lines = [json.dumps(cache[key].model_dump(mode="json"), ensure_ascii=False) for key in sorted(cache)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")


def draw(shas: Sequence[str], size: int = SAMPLE_SIZE, seed: int = SEED) -> list[str]:
    """A seeded sample of image hashes, in hash order so the file diffs cleanly."""
    return sorted(
        random.Random(seed).sample(  # noqa: S311 - a reproducible sample, not a secret
            sorted(shas), min(size, len(shas))
        )
    )


def allergen_terms(text: str, lexicon: Lexicon) -> dict[str, list[str]]:
    """Each lexicon term in ``text`` and the allergens it implies: the evidence behind the vision tags."""
    terms: dict[str, list[str]] = {}
    for hit in lexicon.match(text):
        if hit.allergens:
            terms[hit.term] = sorted(a.value for a in with_implied(hit.allergens))
    return terms


def build_entries(
    cache: Mapping[DescriptionKey, ImageDescription],
    dishes: Mapping[str, tuple[str, str]],
    lexicon: Lexicon,
    previous: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Sample entries for ``dishes`` (sha256 → (item_id, name)), keeping any scores already in ``previous``."""
    described = [sha for sha in dishes if (sha, Variant.NAME_FREE) in cache]
    entries = []
    for sha in draw(described):
        text = cache[(sha, Variant.NAME_FREE)].text
        item_id, name = dishes[sha]
        entry: dict[str, Any] = {
            "sha256": sha,
            "item_id": item_id,
            "dish": name,
            "description": text,
            "allergen_terms": allergen_terms(text, lexicon),
            "dish_type": None,
            "seen": [],
            "not_seen": [],
            "not_visible": [],
            "note": "",
        }
        old = previous.get(sha)
        if old is not None and old.get("description") == text:
            entry.update({field: old[field] for field in SCORE_FIELDS if field in old})
        entries.append(entry)
    return entries


def problems(entries: Sequence[Mapping[str, Any]], cache: Mapping[DescriptionKey, ImageDescription]) -> list[str]:
    """Everything that stops a report: unscored entries, bad values, stale descriptions."""
    found = []
    for entry in entries:
        key = entry["item_id"]
        current = cache.get((entry["sha256"], Variant.NAME_FREE))
        if current is None or current.text != entry["description"]:
            found.append(f"{key}: the description changed since sampling; run `sample` again")
        if entry["dish_type"] not in DISH_TYPES:
            found.append(f"{key}: dish_type must be one of {', '.join(DISH_TYPES)}")
        unknown = set(entry["not_visible"]) - set(entry["allergen_terms"])
        if unknown:
            found.append(f"{key}: not_visible names terms not in allergen_terms: {', '.join(sorted(unknown))}")
    return found


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; honest at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def rate(k: int, n: int) -> dict[str, Any]:
    return {"rate": round(k / n, 3) if n else None, "k": k, "n": n, "ci95": wilson(k, n)}


def percentile(values: Sequence[int], q: float) -> int | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def latency(cache: Mapping[DescriptionKey, ImageDescription]) -> dict[str, Any]:
    """p50/p95 per variant over every cached description, not just the sample."""
    out: dict[str, Any] = {}
    for variant in Variant:
        values = [d.latency_ms for (_, v), d in cache.items() if v is variant]
        if not values:  # hosted runs describe the name-free variant only
            continue
        out[variant.value] = {"n": len(values), "p50_ms": percentile(values, 0.5), "p95_ms": percentile(values, 0.95)}
    return out


def summarise(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(entries)
    types = Counter(entry["dish_type"] for entry in entries)
    seen = sum(len(entry["seen"]) for entry in entries)
    not_seen = sum(len(entry["not_seen"]) for entry in entries)
    terms = sum(len(entry["allergen_terms"]) for entry in entries)
    false_terms = sum(len(entry["not_visible"]) for entry in entries)
    return {
        "descriptions": n,
        "dish_type": {
            "counts": {kind: types[kind] for kind in DISH_TYPES},
            "correct": rate(types["correct"], n),
            "correct_or_partial": rate(types["correct"] + types["partial"], n),
        },
        "ingredients": {
            "named": seen + not_seen,
            "precision": rate(seen, seen + not_seen),
            "descriptions_with_an_unseen_ingredient": rate(sum(bool(e["not_seen"]) for e in entries), n),
        },
        "allergen_terms": {
            "found": terms,
            "not_visible": rate(false_terms, terms),
            "descriptions_with_a_false_term": rate(sum(bool(e["not_visible"]) for e in entries), n),
        },
    }


def _pct(stat: Mapping[str, Any]) -> str:
    if stat["rate"] is None:
        return "n/a"
    low, high = stat["ci95"]
    return f"{stat['k']}/{stat['n']} ({stat['rate']:.0%}, 95% CI {low:.0%} to {high:.0%})"


def render(summary: Mapping[str, Any], timing: Mapping[str, Any], reviewed: bool) -> str:
    dish, ingredients, terms = summary["dish_type"], summary["ingredients"], summary["allergen_terms"]
    lines = [
        f"{summary['descriptions']} name-free descriptions, "
        + ("reviewed by the owner." if reviewed else "NOT YET REVIEWED by the owner."),
        f"Dish type correct: {_pct(dish['correct'])}; correct or partial: {_pct(dish['correct_or_partial'])}.",
        f"Ingredients visible in the photo: {_pct(ingredients['precision'])}; "
        f"descriptions naming an unseen one: {_pct(ingredients['descriptions_with_an_unseen_ingredient'])}.",
        f"Allergen terms whose food is not in the photo: {_pct(terms['not_visible'])}; "
        f"descriptions with one: {_pct(terms['descriptions_with_a_false_term'])}.",
    ]
    lines += [
        f"Latency ({variant}, all {t['n']} photos): p50 {t['p50_ms']} ms, p95 {t['p95_ms']} ms."
        for variant, t in timing.items()
    ]
    return "\n".join(lines)


def review_page(entries: Sequence[Mapping[str, Any]]) -> str:
    """A static page for checking scores against photos; thumbnails are read from ``thumbs/`` beside it."""
    cards = []
    for entry in entries:
        esc = {key: html.escape(str(value)) for key, value in entry.items()}
        terms = ", ".join(f"{t} → {'/'.join(a)}" for t, a in entry["allergen_terms"].items()) or "none"
        cards.append(
            f'<section><img src="thumbs/{esc["sha256"]}.webp" alt="{esc["dish"]}" loading="lazy">'
            f"<div><h2>{esc['item_id']} · {esc['dish']}</h2><p>{esc['description']}</p>"
            f"<dl><dt>dish type</dt><dd>{esc['dish_type']}</dd>"
            f"<dt>seen</dt><dd>{html.escape(', '.join(entry['seen']))}</dd>"
            f"<dt>not seen</dt><dd>{html.escape(', '.join(entry['not_seen']))}</dd>"
            f"<dt>allergen terms</dt><dd>{html.escape(terms)}</dd>"
            f"<dt>not visible</dt><dd>{html.escape(', '.join(entry['not_visible']))}</dd>"
            f"<dt>note</dt><dd>{esc['note']}</dd></dl></div></section>"
        )
    style = (
        "body{font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}"
        "section{display:flex;gap:16px;border-bottom:1px solid #ccc;padding:12px 0}"
        "img{width:360px;max-width:45%;height:auto;object-fit:contain;align-self:flex-start}"
        "h2{font-size:16px;margin:0 0 6px}dl{display:grid;grid-template-columns:9em 1fr;gap:2px 8px;margin:0}"
        "dt{color:#666}dd{margin:0}"
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Vision sample review</title>'
        f"<style>{style}</style><h1>Vision sample: {len(entries)} name-free descriptions</h1>{''.join(cards)}</html>\n"
    )


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)  # noqa: S607
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def read_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "reviewed": False, "entries": []}
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def write_dataset(path: Path, dataset: Mapping[str, Any]) -> None:
    body = yaml.safe_dump(dict(dataset), sort_keys=False, allow_unicode=True, width=110)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + body, encoding="utf-8", newline="\n")


def hosted_settings(settings: Settings, provider: ProviderName) -> Settings:
    """Production settings with the measured model and a relaxed attempt timeout, so slow calls are counted."""
    return settings.model_copy(
        update={
            f"{provider}_vision_model": HOSTED_MODELS[provider],
            f"{provider}_timeout_s": HOSTED_TIMEOUT_S,
            "request_deadline_s": HOSTED_TIMEOUT_S * 2,
        }
    )


def wait_for_token_budget(settings: Settings, provider: ProviderName, expected: int) -> None:
    """Sleep until a call of about ``expected`` tokens fits under the provider's tokens-a-minute cap."""
    cap = settings.minute_token_limit(provider)
    ledger = get_usage_ledger()
    while cap is not None and ledger.usage(provider).tokens_last_minute + expected > cap:
        time.sleep(2)


def describe_sample(settings: Settings, provider: ProviderName) -> list[HostedDescription]:
    """Describe the local sample's photos with ``provider``'s model (name-free), skipping those already described."""
    sample = read_dataset(DATASET)["entries"]
    if not sample:
        raise SystemExit("no local sample yet; run `sample` first")
    photos = {photo.sha256: photo for photo in load_catalog(settings.raw_dir).catalog.photos}
    path = hosted_cache_path(provider)
    cache = read_hosted(path)
    measured = hosted_settings(settings, provider)
    model = build_chat_model(provider, "vision", measured)
    client = CommonsClient()
    ledger = get_usage_ledger()
    new: list[HostedDescription] = []
    expected = 0  # the most tokens one photo has cost so far
    for entry in sample:
        sha = entry["sha256"]
        found = cache.get((sha, Variant.NAME_FREE))
        if found is not None and found.prompt_version == PROMPT_VERSION:
            continue
        image = ensure_cached(photos[sha], client, settings.photo_cache_dir).read_bytes()
        wait_for_token_budget(measured, provider, expected)
        before = ledger.usage(provider).tokens_last_minute
        failed = 0
        try:
            text, latency_ms = describe(model, image, prompt_for(Variant.NAME_FREE, ""))
        except (ProviderRateLimitedError, ProviderTimeoutError) as error:  # recorded, then one retry after a wait
            print(f"{entry['item_id']}: {error.code}; retrying in {RATE_LIMIT_WAIT_S:g} s")
            failed = 1
            time.sleep(RATE_LIMIT_WAIT_S)
            before = ledger.usage(provider).tokens_last_minute
            text, latency_ms = describe(model, image, prompt_for(Variant.NAME_FREE, ""))
        tokens = max(0, ledger.usage(provider).tokens_last_minute - before)
        expected = max(expected, tokens)
        cache[(sha, Variant.NAME_FREE)] = description = HostedDescription(
            sha256=sha,
            variant=Variant.NAME_FREE,
            model=HOSTED_MODELS[provider],
            prompt_version=PROMPT_VERSION,
            text=text,
            latency_ms=latency_ms,
            tokens=tokens,
            failed_attempts=failed,
        )
        new.append(description)
        write_hosted(path, cache)
        print(f"{entry['item_id']}: {latency_ms} ms, {tokens} tokens")
    return new


def hosted_usage(cache: Mapping[DescriptionKey, HostedDescription], timeout_s: float) -> dict[str, Any]:
    """Tokens per photo, and the calls that would have missed the production attempt timeout."""
    tokens = [d.tokens for d in cache.values()]
    return {
        "tokens_per_photo": {"n": len(tokens), "p50": percentile(tokens, 0.5), "max": max(tokens, default=None)},
        "over_production_timeout": {
            "timeout_s": timeout_s,
            "k": sum(d.latency_ms > timeout_s * 1000 for d in cache.values()),
            "n": len(cache),
        },
        "failed_attempts": sum(d.failed_attempts for d in cache.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("describe", "sample", "report"))
    parser.add_argument("--provider", choices=(LOCAL, *HOSTED_MODELS), default=LOCAL)
    parser.add_argument("--out", type=Path, help="results file (default eval/results/vision_measurement_<date>.json)")
    args = parser.parse_args(argv)

    settings = get_settings()
    provider = args.provider
    if args.command == "describe":
        if provider == LOCAL:
            raise SystemExit("local descriptions come from scripts/describe_photos.py")
        new = describe_sample(settings, provider)
        print(f"{len(new)} descriptions written to {hosted_cache_path(provider).relative_to(REPO_ROOT)}")
        return 0

    cache: Mapping[DescriptionKey, ImageDescription] = (
        read_descriptions(settings.processed_dir / DESCRIPTIONS_FILE)
        if provider == LOCAL
        else read_hosted(hosted_cache_path(provider))
    )
    path = dataset_path(provider)
    dataset = read_dataset(path)

    if args.command == "sample":
        catalog = load_catalog(settings.raw_dir).catalog
        names = {item.item_id: item.name for item in catalog.items}
        dishes: dict[str, tuple[str, str]] = {}
        for photo in catalog.photos:  # a photo shared by two dishes is described under the first
            dishes.setdefault(photo.sha256, (photo.item_id, names[photo.item_id]))
        if provider != LOCAL:  # the same photos as the local sample
            dishes = {entry["sha256"]: dishes[entry["sha256"]] for entry in read_dataset(DATASET)["entries"]}
        previous = {entry["sha256"]: entry for entry in dataset["entries"]}
        dataset["entries"] = build_entries(cache, dishes, load_lexicon(), previous)
        dataset.update(seed=SEED, prompt_version=PROMPT_VERSION)
        write_dataset(path, {key: dataset[key] for key in ("version", "seed", "prompt_version", "reviewed", "entries")})
        suffix = "" if provider == LOCAL else f"_{provider}"
        page = settings.photo_cache_dir / f"vision_review{suffix}.html"
        page.write_text(review_page(dataset["entries"]), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(dataset['entries'])} entries) and {page}")
        return 0

    found = problems(dataset["entries"], cache)
    if not dataset["entries"] or found:
        print("\n".join(found or ["no sample yet; run `sample` first"]))
        return 1
    summary = summarise(dataset["entries"])
    timing = latency(cache)
    print(render(summary, timing, bool(dataset["reviewed"])))
    models = sorted({d.model for d in cache.values()})
    results: dict[str, Any] = {
        "run": {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "commit": _git_commit(),
            "provider": provider,
            "dataset": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reviewed": bool(dataset["reviewed"]),
            "models": models,
            "prompt_version": dataset["prompt_version"],
        },
        "summary": summary,
        "latency": timing,
    }
    if provider != LOCAL:
        usage = hosted_usage(read_hosted(hosted_cache_path(provider)), settings.timeout_for(provider))
        results["usage"] = usage
        tokens, slow = usage["tokens_per_photo"], usage["over_production_timeout"]
        print(
            f"Tokens per photo: p50 {tokens['p50']}, max {tokens['max']}; "
            f"over the {slow['timeout_s']:g} s production timeout: {slow['k']}/{slow['n']}; "
            f"failed attempts (timeouts, 429s): {usage['failed_attempts']}."
        )
    name = "vision_measurement" if provider == LOCAL else f"vision_measurement_{provider}"
    out = args.out or RESULTS / f"{name}_{datetime.now(UTC):%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
