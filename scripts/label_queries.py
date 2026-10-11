"""Grade the Phase 3 text queries (plan §1). Local only: pooling runs the real models; nothing here calls a hosted LLM.

1. Pool candidates and build the grading page (needs the built catalog and index: python -m food_concierge.flows build):

       python scripts/label_queries.py pool

   This writes pool.json and grading.html to a review folder in the local cache (outside the repository) and prints
   where. Open the page, grade every dish 0-3 using eval/datasets/labelling-guide.md, and download grades.csv.

2. Import the grades into eval/datasets/retrieval.yaml (every pooled dish needs a grade):

       python scripts/label_queries.py import path/to/grades.csv

3. Once every query is written and graded, split dev/test (seeded by EVAL_SEED) and lock the test split:

       python scripts/label_queries.py split
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from food_concierge.config import REPO_ROOT, Settings, get_settings
from food_concierge.evaluation.dataset import RetrievalDataset, assign_splits, load_dataset, lock
from food_concierge.evaluation.labelling import (
    PoolEntry,
    apply_grades,
    build_pool,
    read_grades,
    render_page,
    write_dataset,
)
from food_concierge.ingestion.loader import load_catalog
from food_concierge.models.embeddings import FastEmbedder, get_reranker
from food_concierge.services.retrieval import Retriever
from food_concierge.storage.catalog_db import CatalogStore
from food_concierge.storage.indexes import INDEX_DIR, build_indexes, load_indexes

DATASET = REPO_ROOT / "eval" / "datasets" / "retrieval.yaml"
POOL_FILE = "pool.json"
PAGE_FILE = "grading.html"


def review_dir(settings: Settings) -> Path:
    return settings.model_cache_dir.parent / "review"


def pool(settings: Settings) -> Path:
    dataset = load_dataset(DATASET)
    store = CatalogStore(settings.processed_dir / "catalog.db")
    out = review_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    reranker = get_reranker(settings)
    prefix = FastEmbedder(settings)
    no_prefix = FastEmbedder(settings.model_copy(update={"embed_query_prefix": ""}))
    build_indexes(store, no_prefix, out / "index_no_prefix")
    retrievers = {
        "prefix": Retriever(store, load_indexes(store, prefix, settings.processed_dir / INDEX_DIR), prefix, reranker),
        "no_prefix": Retriever(store, load_indexes(store, no_prefix, out / "index_no_prefix"), no_prefix, reranker),
    }
    entries = build_pool(dataset.queries, retrievers, store.dish_keys(), settings.eval_seed)
    (out / POOL_FILE).write_text(json.dumps([entry.__dict__ for entry in entries], indent=1), encoding="utf-8")
    items = {item.item_id: item for item in load_catalog().catalog.items}
    page = render_page(entries, {query.id: query for query in dataset.queries}, items)
    (out / PAGE_FILE).write_text(page, encoding="utf-8")
    total = sum(len(entry.to_grade) for entry in entries)
    print(f"{total} dishes to grade over {len(entries)} queries: {out / PAGE_FILE}")
    return out / PAGE_FILE


def read_pool(settings: Settings) -> list[PoolEntry]:
    raw = json.loads((review_dir(settings) / POOL_FILE).read_text(encoding="utf-8"))
    return [PoolEntry(row["query_id"], tuple(row["to_grade"]), tuple(row["targets"])) for row in raw]


def import_grades(settings: Settings, grades_csv: Path) -> RetrievalDataset:
    dataset = apply_grades(load_dataset(DATASET), read_pool(settings), read_grades(grades_csv))
    write_dataset(dataset, DATASET)
    print(f"grades imported into {DATASET.name}")
    return dataset


def split(settings: Settings) -> RetrievalDataset:
    dataset = load_dataset(DATASET)
    locked = lock(dataset.model_copy(update={"queries": assign_splits(dataset.queries, settings.eval_seed)}))
    write_dataset(locked, DATASET)
    print(f"split with seed {settings.eval_seed}; test split locked ({locked.test_sha256})")
    return locked


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pool", help="pool candidates and build the grading page")
    importer = commands.add_parser("import", help="import downloaded grades")
    importer.add_argument("grades", type=Path)
    commands.add_parser("split", help="assign dev/test and lock the test split")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.command == "pool":
        pool(settings)
    elif args.command == "import":
        import_grades(settings, args.grades)
    else:
        split(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
