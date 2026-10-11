"""Retriever: the Phase 3 stages (BM25, dense, hybrid RRF, rerank, de-duplication) on the real catalog, offline."""

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from food_concierge.config import REPO_ROOT
from food_concierge.errors import ConfigError, NotReadyError
from food_concierge.ingestion.allergens import load_lexicon, tag_item
from food_concierge.ingestion.loader import load_catalog
from food_concierge.ingestion.taxonomy import Allergen, Cuisine, Diet
from food_concierge.models.embeddings import HashingEmbedder, OverlapReranker
from food_concierge.services.retrieval import Hit, RetrievalConfig, Retriever, filtered_search, rrf_fuse
from food_concierge.storage.catalog_db import CatalogStore, Filters, build_catalog_db
from food_concierge.storage.indexes import SearchIndexes, build_indexes, load_indexes

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
NO_FILTERS = Filters()


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CatalogStore]:
    loaded = load_catalog(REPO_ROOT / "data" / "raw")
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    tags = {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}
    path = tmp_path_factory.mktemp("db") / "catalog.db"
    build_catalog_db(loaded.catalog, tags, path, lexicon_version=1, source_sha256="abc")
    opened = CatalogStore(path)
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def indexes(store: CatalogStore, tmp_path_factory: pytest.TempPathFactory) -> SearchIndexes:
    path: Path = tmp_path_factory.mktemp("processed") / "indexes"
    build_indexes(store, HashingEmbedder(), path, now=lambda: NOW)
    return load_indexes(store, HashingEmbedder(), path)


@pytest.fixture(scope="module")
def retriever(store: CatalogStore, indexes: SearchIndexes) -> Retriever:
    return Retriever(store, indexes, HashingEmbedder(), OverlapReranker())


def ids(hits: list[Hit]) -> list[str]:
    return [hit.item_id for hit in hits]


def test_rrf_fuse_sums_reciprocal_ranks_and_breaks_ties_by_id() -> None:
    fused = rrf_fuse([[Hit("a", 9.0), Hit("b", 5.0)], [Hit("b", 0.9), Hit("c", 0.8)]], rrf_k=60)

    assert ids(fused) == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert ids(rrf_fuse([[Hit("z", 1.0)], [Hit("y", 1.0)]], rrf_k=60)) == ["y", "z"]  # equal scores: by id


def test_dense_mode_matches_filtered_search(retriever: Retriever, store: CatalogStore, indexes: SearchIndexes) -> None:
    filters = Filters(diet=Diet.VEGETARIAN)
    config = RetrievalConfig(mode="dense", k=5)

    expected = filtered_search(store, indexes.faiss, HashingEmbedder().embed_query("paneer curry"), filters, 5)

    assert retriever.search("paneer curry", filters, config).hits == expected


def test_faiss_and_qdrant_rank_alike(retriever: Retriever) -> None:
    # Both are exact search over the same vectors, so only latency can tell them apart (Phase 3, C2 vs C3).
    for query in ("spicy chicken", "sweet milk dessert", "crispy rice crepe"):
        faiss_ids = ids(retriever.search(query, NO_FILTERS, RetrievalConfig(backend="faiss", k=10)).hits)
        qdrant_ids = ids(retriever.search(query, NO_FILTERS, RetrievalConfig(backend="qdrant", k=10)).hits)
        assert faiss_ids == qdrant_ids


def test_bm25_mode_returns_lexical_order(retriever: Retriever, indexes: SearchIndexes) -> None:
    result = retriever.search("butter naan", NO_FILTERS, RetrievalConfig(mode="bm25", k=4))

    assert result.hits == indexes.bm25.search("butter naan", 4)


def test_hybrid_fuses_both_lists_at_fuse_depth(retriever: Retriever, indexes: SearchIndexes) -> None:
    query = "coconut fish curry"
    config = RetrievalConfig(mode="hybrid", k=5, fuse_depth=10)
    lexical = indexes.bm25.search(query, 10)
    dense = indexes.faiss.search(HashingEmbedder().embed_query(query), 10)

    assert retriever.search(query, NO_FILTERS, config).hits == rrf_fuse([lexical, dense], 60)[:5]


def test_rerank_reorders_only_the_top_n(retriever: Retriever, store: CatalogStore) -> None:
    query = "masala dosa with potato"
    base = retriever.search(query, NO_FILTERS, RetrievalConfig(k=10)).hits
    reranked = retriever.search(query, NO_FILTERS, RetrievalConfig(k=10, rerank_depth=5)).hits
    texts = store.doc_texts()
    scores = OverlapReranker().score(query, [texts[hit.item_id] for hit in base[:5]])

    head = sorted(zip(ids(base[:5]), scores, strict=True), key=lambda pair: (-pair[1], pair[0]))
    assert ids(reranked) == [item_id for item_id, _ in head] + ids(base[5:])
    assert [hit.score for hit in reranked[:5]] == [score for _, score in head]


def test_dedupe_keeps_one_item_per_dish_and_still_fills_k(retriever: Retriever, store: CatalogStore) -> None:
    keys = store.dish_keys()
    query = "paneer butter masala"
    plain = retriever.search(query, NO_FILTERS, RetrievalConfig(mode="bm25", k=10)).hits
    assert len({keys[i] for i in ids(plain)}) < len(plain)  # the dish is at two restaurants

    deduped = retriever.search(query, NO_FILTERS, RetrievalConfig(mode="bm25", k=10, dedupe=True)).hits
    assert len(deduped) == 10
    assert len({keys[i] for i in ids(deduped)}) == 10
    assert deduped[0] == plain[0]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_every_mode_returns_only_admitted_items(retriever: Retriever, store: CatalogStore, mode: str) -> None:
    filter_sets = [
        Filters(diet=Diet.VEGAN),
        Filters(exclude_allergens=frozenset({Allergen.MILK, Allergen.TREE_NUTS})),
        Filters(max_kcal=350, max_price_inr=250),
        Filters(cuisines=frozenset({Cuisine.SOUTH_INDIAN}), eggless=True),
    ]
    for filters, rerank, dedupe in itertools.product(filter_sets, (None, 5), (False, True)):
        config = RetrievalConfig.model_validate({"mode": mode, "k": 5, "rerank_depth": rerank, "dedupe": dedupe})
        hits = retriever.search("rich spicy curry with rice", filters, config).hits
        assert set(ids(hits)) <= set(store.candidate_ids(filters))


def test_no_admitted_items_returns_nothing(retriever: Retriever) -> None:
    result = retriever.search("anything", Filters(max_kcal=1), RetrievalConfig(mode="hybrid"))

    assert result.hits == []
    assert set(result.timings_ms) == {"filter", "total"}


def test_timings_cover_each_stage(store: CatalogStore, indexes: SearchIndexes) -> None:
    ticks = itertools.count()
    retriever = Retriever(store, indexes, HashingEmbedder(), OverlapReranker(), clock=lambda: next(ticks) / 1000)
    config = RetrievalConfig(mode="hybrid", rerank_depth=5, dedupe=True)

    timings = retriever.search("dal", NO_FILTERS, config).timings_ms

    assert list(timings) == ["filter", "embed", "bm25", "dense", "fuse", "dedupe", "rerank", "total"]
    assert all(value == pytest.approx(1.0) for name, value in timings.items() if name != "total")
    assert timings["total"] > timings["filter"]


def test_rerank_without_a_reranker_is_a_config_error(store: CatalogStore, indexes: SearchIndexes) -> None:
    retriever = Retriever(store, indexes, HashingEmbedder())

    with pytest.raises(ConfigError, match="reranker"):
        retriever.search("dal", NO_FILTERS, RetrievalConfig(rerank_depth=5))
    assert retriever.search("dal", NO_FILTERS, RetrievalConfig()).hits


def test_an_embedder_the_index_was_not_built_with_is_refused(store: CatalogStore, indexes: SearchIndexes) -> None:
    with pytest.raises(NotReadyError, match="query_prefix"):
        Retriever(store, indexes, HashingEmbedder(query_prefix="query: "))


def test_config_rejects_unknown_fields_and_empty_depths() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig.model_validate({"mode": "dense", "top": 3})
    with pytest.raises(ValidationError):
        RetrievalConfig(k=0)
    with pytest.raises(ValidationError):
        RetrievalConfig(rerank_depth=0)
