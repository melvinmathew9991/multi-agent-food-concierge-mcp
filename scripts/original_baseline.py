"""The original approach re-implemented as baseline C0 (plan §3). Needs the built catalog and index:
python -m food_concierge.flows build.

1. Retrieve each query's dense top 5 without filters and ask Groq one Yes/No question per dish (live, never in
   CI). Paced, saved after each query, resumable; a changed query is asked again:

       python scripts/original_baseline.py collect [--limit N]

   About 5 calls and 1,000 tokens a query. The chat model's quota is separate from the vision model's (Groq's
   free tier: 1,000 requests and 200,000 tokens a day), so the full set fits in one day.

2. Score C0 offline from the cache, with the original's strict parser and a fixed one:

       python scripts/original_baseline.py replay

Both read eval/datasets/retrieval.yaml; collect writes eval/caches/original_verdicts.jsonl.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from food_concierge.config import REPO_ROOT, ModelRole, ProviderName, Settings, get_settings
from food_concierge.errors import ProviderError, ProviderRateLimitedError, ProviderTimeoutError
from food_concierge.evaluation.dataset import RetrievalDataset, TextQuery, load_dataset, select
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.original import (
    SEARCH,
    Judged,
    collect,
    missing,
    read_runs,
    relevance_prompt,
    replay,
)
from food_concierge.evaluation.runner import summarize
from food_concierge.ingestion.loader import load_catalog
from food_concierge.models.embeddings import get_embedder
from food_concierge.models.router import build_chat_model
from food_concierge.services.retrieval import Retriever
from food_concierge.storage.catalog_db import CatalogStore, Filters
from food_concierge.storage.indexes import INDEX_DIR, load_indexes

DATASET = REPO_ROOT / "eval" / "datasets" / "retrieval.yaml"
CACHE = REPO_ROOT / "eval" / "caches" / "original_verdicts.jsonl"
PROVIDER: ProviderName = "groq"
ROLE: ModelRole = "router"  # temperature 0, the seed and low reasoning effort: a classification, like routing
RATE_LIMIT_WAIT_S = 60.0
MIN_INTERVAL_S = 3.0  # 20 calls a minute: under Groq's 30 requests and, at about 200 tokens each, its 8,000 tokens


def collect_all(settings: Settings, limit: int | None) -> None:
    dataset = load_dataset(DATASET)
    store = CatalogStore(settings.processed_dir / "catalog.db")
    embedder = get_embedder(settings)
    retriever = Retriever(store, load_indexes(store, embedder, settings.processed_dir / INDEX_DIR), embedder)
    items = {item.item_id: item for item in load_catalog(settings.raw_dir).catalog.items}
    model = build_chat_model(PROVIDER, ROLE, settings)
    last_call = 0.0

    def search(request: str) -> list[str]:
        return [hit.item_id for hit in retriever.search(request, Filters(), SEARCH).hits]

    def ask(prompt: str) -> Judged:
        nonlocal last_call
        time.sleep(max(0.0, last_call + MIN_INTERVAL_S - time.monotonic()))
        last_call = time.monotonic()
        reply = model.invoke(prompt)
        usage = getattr(reply, "usage_metadata", None)
        return Judged(str(reply.content), int(usage["total_tokens"]) if usage else 0)

    def judge(request: str, item_id: str) -> Judged:
        prompt = relevance_prompt(request, items[item_id])
        try:
            return ask(prompt)
        except (ProviderRateLimitedError, ProviderTimeoutError) as error:
            print(f"{error.code}; retrying in {RATE_LIMIT_WAIT_S:g} s")
            time.sleep(RATE_LIMIT_WAIT_S)
            return ask(prompt)

    model_name = settings.require_model_id(PROVIDER, ROLE)
    try:
        runs = collect(dataset.queries, search, judge, model_name, CACHE, limit)
    except ProviderError as error:  # every finished query is already saved; a re-run resumes from the next one
        raise SystemExit(f"stopped: {error.code} ({error.message}); run `collect` again to resume") from error
    tokens = sum(run.tokens for run in runs.values())
    left = missing(dataset.queries, runs, model_name)
    print(f"{len(runs)} queries cached ({tokens} tokens in all); {len(left)} still to ask")


def scored_queries(dataset: RetrievalDataset) -> tuple[TextQuery, ...]:
    """The dev split once queries are split; before that, every query (nothing is decided on C0 before the split)."""
    if any(query.split is not None for query in dataset.queries):
        return select(dataset, "dev")
    print("queries are not split yet: scoring every query")
    return dataset.queries


def replay_all(settings: Settings) -> None:
    dataset = load_dataset(DATASET)
    queries = scored_queries(dataset)
    runs = read_runs(CACHE)
    left = missing(queries, runs, settings.require_model_id(PROVIDER, ROLE))
    if left:
        raise SystemExit(f"{len(left)} queries are not cached ({', '.join(left[:5])}…); run `collect` first")
    truth = GroundTruth.from_load(load_catalog(settings.raw_dir))
    for name, rows in replay(queries, runs, truth).items():
        summary = summarize(rows, settings.eval_seed)
        print(name)
        for metric, interval in summary.items():
            print(f"  {metric:<12} {interval.mean:.3f}  [{interval.low:.3f}, {interval.high:.3f}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect", help="retrieve and ask Groq for the missing queries (live)")
    collector.add_argument("--limit", type=int, default=None, help="queries to ask in this run")
    commands.add_parser("replay", help="score C0 offline from the cache")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.command == "collect":
        collect_all(settings, args.limit)
    else:
        replay_all(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
