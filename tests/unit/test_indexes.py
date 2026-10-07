"""Search indexes: build, manifest binding, the three backends, and filtered search never returning a violation."""

import json
import random
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import pytest

from food_concierge.config import REPO_ROOT
from food_concierge.errors import NotReadyError
from food_concierge.ingestion.allergens import DishAllergens, load_lexicon, tag_item
from food_concierge.ingestion.loader import CatalogLoad, load_catalog
from food_concierge.ingestion.taxonomy import Allergen, Cuisine, Diet
from food_concierge.models.embeddings import HashingEmbedder
from food_concierge.services.retrieval import Bm25Index, FaissStore, QdrantStore, filtered_search, tokenize
from food_concierge.storage.catalog_db import CatalogStore, Filters, build_catalog_db
from food_concierge.storage.indexes import (
    FAISS_INDEX,
    ITEM_IDS,
    MANIFEST,
    VECTORS,
    SearchIndexes,
    build_indexes,
    load_indexes,
)

RAW = REPO_ROOT / "data" / "raw"
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def loaded() -> CatalogLoad:
    return load_catalog(RAW)


@pytest.fixture(scope="module")
def tags(loaded: CatalogLoad) -> dict[str, DishAllergens]:
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    return {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}


def make_store(
    loaded: CatalogLoad, tags: dict[str, DishAllergens], path: Path, data_sha256: str = "abc"
) -> CatalogStore:
    build_catalog_db(loaded.catalog, tags, path, lexicon_version=1, source_sha256=data_sha256)
    return CatalogStore(path)


@pytest.fixture(scope="module")
def store(
    loaded: CatalogLoad, tags: dict[str, DishAllergens], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[CatalogStore]:
    opened = make_store(loaded, tags, tmp_path_factory.mktemp("db") / "catalog.db")
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture(scope="module")
def index_dir(store: CatalogStore, embedder: HashingEmbedder, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("processed") / "indexes"
    build_indexes(store, embedder, path, now=lambda: NOW)
    return path


@pytest.fixture(scope="module")
def indexes(store: CatalogStore, embedder: HashingEmbedder, index_dir: Path) -> SearchIndexes:
    return load_indexes(store, embedder, index_dir)


def test_manifest_records_what_the_index_was_built_from(indexes: SearchIndexes, embedder: HashingEmbedder) -> None:
    manifest = indexes.manifest

    assert manifest.embedder == embedder.fingerprint
    assert (manifest.dim, manifest.n_items, manifest.data_sha256, manifest.doc_text_version) == (384, 150, "abc", 1)
    assert manifest.created_at == NOW
    assert set(manifest.files) == {VECTORS, ITEM_IDS, FAISS_INDEX}


def test_backends_find_the_obvious_dish(indexes: SearchIndexes, embedder: HashingEmbedder) -> None:
    query = embedder.embed_query("masala dosa potato crepe")

    assert indexes.faiss.search(query, 3)[0].item_id in {"t01", "e09"}
    assert indexes.qdrant.search(query, 3)[0].item_id in {"t01", "e09"}
    assert indexes.bm25.search("masala dosa", 3)[0].item_id in {"t01", "e09"}


def test_faiss_and_qdrant_agree(indexes: SearchIndexes, embedder: HashingEmbedder) -> None:
    for text in ["spicy chicken", "sweet milk dessert", "vegan breakfast", "noodles with soy sauce"]:
        query = embedder.embed_query(text)
        faiss_hits = indexes.faiss.search(query, 10)
        qdrant_hits = indexes.qdrant.search(query, 10)
        assert [h.score for h in faiss_hits] == pytest.approx([h.score for h in qdrant_hits], abs=1e-5)


def random_filters(rng: random.Random) -> Filters:
    return Filters(
        diet=rng.choice([None, *Diet]),
        eggless=rng.random() < 0.3,
        exclude_allergens=frozenset(rng.sample(list(Allergen), rng.choice([0, 1, 2, 3]))),
        max_kcal=rng.choice([None, 250, 400, 600]),
        max_price_inr=rng.choice([None, 100, 250, 450]),
        cuisines=frozenset(rng.sample(list(Cuisine), rng.choice([0, 0, 1, 3]))),
        include_unverified=rng.random() < 0.3,
    )


def test_filtered_search_never_returns_a_violating_dish(
    store: CatalogStore,
    indexes: SearchIndexes,
    embedder: HashingEmbedder,
    loaded: CatalogLoad,
    tags: dict[str, DishAllergens],
) -> None:
    # Phase 2 DoD, on both dense backends and BM25: every hit satisfies every hard filter, and a backend returns as
    # many hits as the filters allow (up to k), so restricting to the allowed ids loses nothing.
    items = {item.item_id: item for item in loaded.catalog.items}
    texts = list(store.doc_texts().values())
    rng = random.Random(7)
    for _ in range(300):
        filters = random_filters(rng)
        query_text = rng.choice(texts)[: rng.randint(5, 60)]
        k = rng.choice([1, 3, 10, 200])
        allowed = store.candidate_ids(filters)
        for backend in (indexes.faiss, indexes.qdrant):
            hits = filtered_search(store, backend, embedder.embed_query(query_text), filters, k)
            assert all(filters.admits(items[h.item_id], tags[h.item_id]) for h in hits), (backend.name, filters)
            assert len(hits) == min(k, len(allowed)), (backend.name, filters)
        lexical = indexes.bm25.search(query_text, k, allowed=allowed)
        assert all(filters.admits(items[h.item_id], tags[h.item_id]) for h in lexical), filters


def test_empty_candidates_and_zero_k(indexes: SearchIndexes, embedder: HashingEmbedder) -> None:
    query = embedder.embed_query("anything")

    for backend in (indexes.faiss, indexes.qdrant):
        assert backend.search(query, 5, allowed=[]) == []
        assert backend.search(query, 5, allowed=["no-such-item"]) == []
        assert backend.search(query, 0) == []
    assert indexes.bm25.search("anything", 0) == []
    assert indexes.bm25.search("zzzzqx", 5) == []  # no shared term, no score


def test_bm25_ranks_and_tokenizes() -> None:
    bm25 = Bm25Index(["a", "b", "c"], ["paneer tikka", "paneer butter masala paneer", "dal makhani"])

    assert tokenize("Paneer-Tikka, 2 plates!") == ["paneer", "tikka", "2", "plates"]
    assert [h.item_id for h in bm25.search("paneer", 5)] == ["b", "a"]
    assert [h.item_id for h in bm25.search("paneer", 5, allowed={"a"})] == ["a"]
    with pytest.raises(ValueError, match="one text per item"):
        Bm25Index(["a"], [])


def test_stores_reject_mismatched_ids() -> None:
    vectors = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="holds 3 vectors for 2 ids"):
        FaissStore(FaissStore.build(vectors, ["a", "b", "c"]).index, ["a", "b"])
    assert [h.item_id for h in QdrantStore(vectors, ["a", "b", "c"]).search(vectors[1], 1)] == ["b"]


def rebuild(store: CatalogStore, embedder: HashingEmbedder, tmp_path: Path) -> Path:
    path = tmp_path / "indexes"
    build_indexes(store, embedder, path, now=lambda: NOW)
    return path


def test_rebuild_replaces_the_directory(store: CatalogStore, embedder: HashingEmbedder, tmp_path: Path) -> None:
    path = rebuild(store, embedder, tmp_path)
    (path / "stale.txt").write_text("from an older build", encoding="utf-8")

    rebuild(store, embedder, tmp_path)

    assert not (path / "stale.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["indexes"]  # no .part or .old left behind
    assert faiss.read_index(str(path / FAISS_INDEX)).ntotal == 150


def test_refuses_another_embedder(store: CatalogStore, embedder: HashingEmbedder, tmp_path: Path) -> None:
    path = rebuild(store, embedder, tmp_path)

    with pytest.raises(NotReadyError, match="different embedder \\(query_prefix changed\\)"):
        load_indexes(store, HashingEmbedder(query_prefix="query: "), path)


def test_refuses_other_catalog_data(
    loaded: CatalogLoad, tags: dict[str, DishAllergens], embedder: HashingEmbedder, tmp_path: Path
) -> None:
    built_from = make_store(loaded, tags, tmp_path / "a.db", data_sha256="old")
    path = rebuild(built_from, embedder, tmp_path)
    current = make_store(loaded, tags, tmp_path / "b.db", data_sha256="new")

    with pytest.raises(NotReadyError, match="different catalog data"):
        load_indexes(current, embedder, path)
    built_from.close()
    current.close()


def test_refuses_changed_or_missing_files(store: CatalogStore, embedder: HashingEmbedder, tmp_path: Path) -> None:
    path = rebuild(store, embedder, tmp_path)
    ids = json.loads((path / ITEM_IDS).read_text(encoding="utf-8"))
    (path / ITEM_IDS).write_text(json.dumps(ids[::-1]), encoding="utf-8")

    with pytest.raises(NotReadyError, match=r"item_ids\.json changed"):
        load_indexes(store, embedder, path)
    (path / VECTORS).unlink()
    (path / ITEM_IDS).write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(NotReadyError, match=r"vectors\.npy changed"):
        load_indexes(store, embedder, path)
    (path / MANIFEST).unlink()
    with pytest.raises(NotReadyError, match="missing or unreadable"):
        load_indexes(store, embedder, path)


def test_refuses_an_index_that_does_not_cover_the_catalog(
    store: CatalogStore, embedder: HashingEmbedder, tmp_path: Path
) -> None:
    path = rebuild(store, embedder, tmp_path)
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    manifest["n_items"] = 149
    (path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NotReadyError, match="does not cover"):
        load_indexes(store, embedder, path)
