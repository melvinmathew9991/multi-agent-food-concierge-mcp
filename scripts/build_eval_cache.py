"""Build or check the committed retrieval caches (plan §6). Local and nightly only: it loads the real models.

1. Build eval/caches/retrieval/ from the catalog and every query in eval/datasets/retrieval.yaml: document vectors,
   query vectors with and without the BGE prefix, and the full query x document rerank score matrix:

       python scripts/build_eval_cache.py build

   Rebuild in the same PR as any change to the catalog, the document text, the query texts or a model; the PR
   gate names the stale field.

2. Check the caches against freshly downloaded models (the nightly drift check; exits 1 on drift):

       python scripts/build_eval_cache.py check
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from food_concierge.config import REPO_ROOT, Settings, get_settings
from food_concierge.evaluation.caches import build_caches, drift, load_caches, texts_sha256
from food_concierge.evaluation.dataset import load_dataset
from food_concierge.ingestion.build import build_catalog
from food_concierge.models.embeddings import Embedder, FastEmbedder, get_reranker
from food_concierge.storage.catalog_db import CatalogStore

DATASET = REPO_ROOT / "eval" / "datasets" / "retrieval.yaml"
CACHE_DIR = REPO_ROOT / "eval" / "caches" / "retrieval"
SIZE_LIMIT = 2 * 1024 * 1024


def embedders(settings: Settings) -> dict[str, Embedder]:
    return {
        "prefix": FastEmbedder(settings),
        "no_prefix": FastEmbedder(settings.model_copy(update={"embed_query_prefix": ""})),
    }


def build(settings: Settings) -> None:
    store = CatalogStore(build_catalog(settings).output)
    meta = store.meta()
    texts = [query.text for query in load_dataset(DATASET).queries]
    manifest = build_caches(
        CACHE_DIR,
        embedders(settings),
        get_reranker(settings),
        list(store.doc_texts().values()),
        texts,
        meta["data_sha256"],
        int(meta["doc_text_version"]),
        texts_sha256(texts),
    )
    size = sum(path.stat().st_size for path in CACHE_DIR.iterdir())
    print(f"{len(manifest.files)} cache files, {size / 1024:.0f} KiB: {CACHE_DIR}")
    if size > SIZE_LIMIT:
        raise SystemExit(f"the caches are over {SIZE_LIMIT // 1024 // 1024} MB; trim them before committing")


def check(settings: Settings) -> None:
    problems = drift(load_caches(CACHE_DIR), embedders(settings), get_reranker(settings))
    for problem in problems:
        print(f"drift: {problem}")
    if problems:
        raise SystemExit(1)
    print("the caches match freshly loaded models")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="embed and score everything with the real models")
    commands.add_parser("check", help="compare the caches with freshly loaded models")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.command == "build":
        build(settings)
    else:
        check(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
