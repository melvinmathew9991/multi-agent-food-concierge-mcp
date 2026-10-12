"""Build the Phase 3 image queries from altered catalog photos (plan §2). Needs the local photo cache.

1. Draw the seeded transforms, compute each altered image's SHA-256, and fill L0 from the local named descriptions
   (and L1 from the Phase 2 Groq measurement where it exists). Offline:

       python scripts/image_queries.py build

2. Describe the rest with the production vision model (Groq; live, never in CI). The catalog's Commons photos are
   sent, never user data. Paced under Groq's tokens-a-minute cap, saved after each photo, resumable:

       python scripts/image_queries.py describe [--limit N]

Both write eval/datasets/image_queries.jsonl.

Groq's free tier allows this model 200,000 tokens a day (measured 2026-10-11; the 429 body says so, the rate-limit
headers do not), so at about 2,000 tokens a photo one day covers about 95 descriptions. The full set takes two to
three days: run ``describe`` once a day until nothing is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence

from food_concierge.config import REPO_ROOT, ProviderName, Settings, get_settings
from food_concierge.errors import ProviderError, ProviderRateLimitedError, ProviderTimeoutError
from food_concierge.evaluation.image_queries import (
    Described,
    RungDescription,
    build_queries,
    describe_queries,
    read_queries,
    write_queries,
)
from food_concierge.ingestion.descriptions import (
    DESCRIPTIONS_FILE,
    PROMPT_VERSION,
    Variant,
    describe,
    prompt_for,
    read_descriptions,
)
from food_concierge.ingestion.loader import load_catalog
from food_concierge.ingestion.photos import CommonsClient, ensure_cached
from food_concierge.models.router import build_chat_model
from food_concierge.models.usage import get_usage_ledger
from food_concierge.storage.catalog_db import dish_key

DATASET = REPO_ROOT / "eval" / "datasets" / "image_queries.jsonl"
HOSTED_L1 = REPO_ROOT / "eval" / "datasets" / "vision_descriptions_groq.jsonl"
PROVIDER: ProviderName = "groq"
ATTEMPT_TIMEOUT_S = 30.0  # relaxed from production: these are offline descriptions, not a user waiting
RATE_LIMIT_WAIT_S = 60.0
# Groq counts more per photo than its replies report (ADR-0007): pacing by reported tokens still drew 429s on
# half the calls on 2026-10-11, so calls are also spaced to about three photos a minute.
MIN_INTERVAL_S = 20.0


def photo_loader(settings: Settings) -> Callable[[str], bytes]:
    photos = {photo.sha256: photo for photo in load_catalog(settings.raw_dir).catalog.photos}
    client = CommonsClient()
    return lambda sha: ensure_cached(photos[sha], client, settings.photo_cache_dir).read_bytes()


def build(settings: Settings) -> None:
    catalog = load_catalog(settings.raw_dir).catalog
    names = {item.item_id: item.name for item in catalog.items}
    photos: dict[str, list[str]] = defaultdict(list)
    for photo in catalog.photos:
        key = dish_key(names[photo.item_id])
        if key not in photos[photo.sha256]:
            photos[photo.sha256].append(key)
    named = read_descriptions(settings.processed_dir / DESCRIPTIONS_FILE)
    measured = [json.loads(line) for line in HOSTED_L1.read_text(encoding="utf-8").splitlines() if line.strip()]
    hosted = {
        row["sha256"]: RungDescription(
            text=row["text"],
            model=row["model"],
            prompt_version=row["prompt_version"],
            latency_ms=row["latency_ms"],
            tokens=row["tokens"],
        )
        for row in measured
        if row["variant"] == Variant.NAME_FREE and row["prompt_version"] == PROMPT_VERSION
    }
    queries = build_queries(photos, photo_loader(settings), named, settings.eval_seed, read_queries(DATASET), hosted)
    write_queries(DATASET, queries)
    missing = sum(2 - len({"L1", "L2"} & q.rungs.keys()) for q in queries)
    print(f"{len(queries)} image queries; {missing} descriptions still to make with {PROVIDER}")


def describe_all(settings: Settings, limit: int | None) -> None:
    measured = settings.model_copy(
        update={"groq_timeout_s": ATTEMPT_TIMEOUT_S, "request_deadline_s": ATTEMPT_TIMEOUT_S * 2}
    )
    model = build_chat_model(PROVIDER, "vision", measured)
    ledger = get_usage_ledger()
    cap = measured.minute_token_limit(PROVIDER) or 0
    expected = 0  # the most tokens one photo has cost so far
    last_call = 0.0
    prompt = prompt_for(Variant.NAME_FREE, "")

    def describe_one(image: bytes) -> Described:
        nonlocal expected, last_call
        while ledger.usage(PROVIDER).tokens_last_minute + expected > cap:
            time.sleep(2)
        time.sleep(max(0.0, last_call + MIN_INTERVAL_S - time.monotonic()))
        last_call = time.monotonic()
        # The running total, not the rolling minute: calls that age out of the minute mid-call would be subtracted.
        before = ledger.usage(PROVIDER).tokens_total
        try:
            text, latency_ms = describe(model, image, prompt)
        except (ProviderRateLimitedError, ProviderTimeoutError) as error:
            print(f"{error.code}; retrying in {RATE_LIMIT_WAIT_S:g} s")
            time.sleep(RATE_LIMIT_WAIT_S)
            text, latency_ms = describe(model, image, prompt)
        tokens = ledger.usage(PROVIDER).tokens_total - before
        expected = max(expected, tokens)
        print(f"{latency_ms} ms, {tokens} tokens")
        return Described(text, latency_ms, tokens)

    queries = sorted(read_queries(DATASET).values(), key=lambda q: q.id)
    if not queries:
        raise SystemExit("no image queries yet; run `build` first")
    model_name = measured.groq_vision_model
    try:
        done = describe_queries(
            queries, photo_loader(settings), describe_one, model_name, PROMPT_VERSION, DATASET, limit
        )
    except ProviderError as error:  # every finished photo is already saved; a re-run resumes from the next one
        raise SystemExit(f"stopped: {error.code} ({error.message}); run `describe` again to resume") from error
    left = sum(not {"L1", "L2"} <= q.rungs.keys() for q in done)
    print(f"done; {left} queries still missing a description")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="draw transforms and fill what the caches already hold")
    describer = commands.add_parser("describe", help="describe the rest with Groq (live)")
    describer.add_argument("--limit", type=int, default=None, help="photos to describe in this run")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.command == "build":
        build(settings)
    else:
        describe_all(settings, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
