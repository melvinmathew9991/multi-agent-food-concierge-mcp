"""Describe every approved catalog photo with the local vision model (Phase 2 plan, item 7). Never run in CI.

    python scripts/describe_photos.py [--limit N]

Needs Ollama with the vision model (``OLLAMA_VISION_MODEL``, default ``qwen2.5vl:7b``) and the photo cache
(``python scripts/fetch_photos.py sync``). Writes data/processed/image_descriptions.jsonl, saving after each photo,
so an interrupted run resumes. Photos that already have current descriptions are skipped.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence

from food_concierge.config import get_settings
from food_concierge.ingestion.descriptions import DESCRIPTIONS_FILE, describe_missing
from food_concierge.ingestion.loader import load_catalog
from food_concierge.ingestion.photos import CommonsClient, ensure_cached
from food_concierge.models.router import build_chat_model

OLLAMA_TIMEOUT_S = 180.0  # the first call loads the model into memory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="describe at most this many photos in this run")
    args = parser.parse_args(argv)

    settings = get_settings()
    local = settings.model_copy(
        update={"provider_timeout_s": OLLAMA_TIMEOUT_S, "request_deadline_s": OLLAMA_TIMEOUT_S * 2}
    )
    catalog = load_catalog(settings.raw_dir).catalog
    names = {item.item_id: item.name for item in catalog.items}
    photos = {photo.sha256: photo for photo in catalog.photos}
    client = CommonsClient()
    model = build_chat_model("ollama", "vision", local)

    def load_image(sha256: str) -> bytes:
        return ensure_cached(photos[sha256], client, settings.photo_cache_dir).read_bytes()

    new = describe_missing(
        [(photo.sha256, names[photo.item_id]) for photo in catalog.photos],
        load_image,
        model,
        settings.ollama_vision_model,
        settings.processed_dir / DESCRIPTIONS_FILE,
        limit=args.limit,
    )
    latencies = sorted(d.latency_ms for d in new)
    if latencies:
        p95 = latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
        print(f"{len(new)} descriptions written; latency p50 {statistics.median(latencies):.0f} ms, p95 {p95} ms")
    else:
        print("All photos already have current descriptions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
