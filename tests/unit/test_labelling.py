"""Pooling, the grading page and grade import for the text queries; real catalog, fake models, offline."""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from food_concierge.config import REPO_ROOT, Settings
from food_concierge.errors import EvaluationError
from food_concierge.evaluation.dataset import RetrievalDataset, TextQuery, load_dataset
from food_concierge.evaluation.labelling import (
    POOL_STACKS,
    PoolEntry,
    PoolStack,
    apply_grades,
    build_pool,
    read_grades,
    render_page,
    write_dataset,
)
from food_concierge.ingestion.allergens import load_lexicon, tag_item
from food_concierge.ingestion.loader import CatalogLoad, load_catalog
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.ingestion.taxonomy import Allergen
from food_concierge.models.embeddings import HashingEmbedder, OverlapReranker
from food_concierge.services.retrieval import RetrievalConfig, Retriever
from food_concierge.storage.catalog_db import CatalogStore, Filters, build_catalog_db
from food_concierge.storage.indexes import build_indexes, load_indexes


@pytest.fixture(scope="module")
def loaded() -> CatalogLoad:
    return load_catalog(REPO_ROOT / "data" / "raw")


@pytest.fixture(scope="module")
def items(loaded: CatalogLoad) -> dict[str, MenuItem]:
    return {item.item_id: item for item in loaded.catalog.items}


@pytest.fixture(scope="module")
def store(loaded: CatalogLoad, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CatalogStore]:
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    tags = {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}
    path = tmp_path_factory.mktemp("db") / "catalog.db"
    build_catalog_db(loaded.catalog, tags, path, lexicon_version=1, source_sha256="abc")
    opened = CatalogStore(path)
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def retrievers(store: CatalogStore, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Retriever]:
    built = {}
    for variant, embedder in {"prefix": HashingEmbedder(), "no_prefix": HashingEmbedder(query_prefix="q: ")}.items():
        path = tmp_path_factory.mktemp(variant) / "indexes"
        build_indexes(store, embedder, path)
        built[variant] = Retriever(store, load_indexes(store, embedder, path), embedder, OverlapReranker())
    return built


def query(query_id: str, text: str, **fields: object) -> TextQuery:
    base: dict[str, object] = {"id": query_id, "text": text, "type": "vague", "author": "drafted"}
    return TextQuery.model_validate(base | fields)


def test_pool_unions_every_stack_and_leaves_out_targets_and_graded_items(
    retrievers: dict[str, Retriever], store: CatalogStore
) -> None:
    keys = store.dish_keys()
    naan = query("t001", "butter naan", type="known_item", target="butter-naan", grades={"k01": 1})
    pool = build_pool([naan], retrievers, keys, seed=7)[0]

    union: set[str] = set()
    for stack in POOL_STACKS.values():
        union |= {hit.item_id for hit in retrievers[stack.variant].search(naan.text, Filters(), stack.config).hits}
    assert pool.targets == ("k03", "s08")
    assert set(pool.to_grade) == union - {"k03", "s08", "k01"}
    assert pool == build_pool([naan], retrievers, keys, seed=7)[0]
    assert pool.to_grade != build_pool([naan], retrievers, keys, seed=8)[0].to_grade  # shuffled by seed


def test_the_unfiltered_stack_pools_items_the_filters_exclude(
    retrievers: dict[str, Retriever], store: CatalogStore
) -> None:
    no_milk = query("t001", "creamy paneer curry", filters={"exclude_allergens": ["milk"]})
    stacks = {
        "filtered": PoolStack(RetrievalConfig(k=5)),
        "unfiltered": PoolStack(RetrievalConfig(k=5), filtered=False),
    }

    pooled = set(build_pool([no_milk], retrievers, store.dish_keys(), 7, stacks)[0].to_grade)

    allowed = set(store.candidate_ids(Filters(exclude_allergens=frozenset({Allergen.MILK}))))
    assert pooled - allowed  # the original approach (C0) can return dishes with milk, so they get judged too


def test_apply_grades_merges_targets_and_requires_every_pooled_item() -> None:
    dataset = RetrievalDataset(queries=(query("t001", "a", type="known_item"), query("t002", "b")))
    pool = [PoolEntry("t001", ("a01", "b02"), ("c03",))]

    with pytest.raises(EvaluationError, match="1 pooled item"):
        apply_grades(dataset, pool, [{"query_id": "t001", "item_id": "a01", "grade": "2"}])
    graded = apply_grades(
        dataset,
        pool,
        [
            {"query_id": "t001", "item_id": "a01", "grade": "2"},
            {"query_id": "t001", "item_id": "b02", "grade": "0"},
            {"query_id": "t001", "item_id": "zz9", "grade": ""},
        ],
    )

    assert graded.queries[0].grades == {"a01": 2, "b02": 0, "c03": 3}
    assert graded.queries[1].grades == {}


def test_page_shows_each_pool_blind_and_notes_filters(items: dict[str, MenuItem]) -> None:
    queries = {
        "t001": query("t001", "dal <tadka>"),
        "t002": query("t002", "light lunch", filters={"diet": "vegan"}),
        "t003": query("t003", "nothing left to grade"),
    }
    pool = [PoolEntry("t001", ("k02", "s07"), ()), PoolEntry("t002", ("t01",), ()), PoolEntry("t003", (), ("k01",))]

    page = render_page(pool, queries, items)

    assert "dal &lt;tadka&gt;" in page
    assert page.count("name='t001:k02'") == 4  # one radio per grade
    assert page.count("filters are checked separately") == 1
    assert "t003" not in page
    assert "bm25" not in page  # the stack that found a dish is never shown
    assert "hybrid" not in page


def test_dataset_round_trips_through_yaml_and_grades_csv(tmp_path: Path) -> None:
    dataset = RetrievalDataset(queries=(query("t001", "dal", grades={"k02": 3}, filters={"max_kcal": 300}),))
    path = tmp_path / "retrieval.yaml"
    write_dataset(dataset, path)
    grades = tmp_path / "grades.csv"
    grades.write_text("query_id,item_id,grade\nt001,k02,3\n", encoding="utf-8")

    assert load_dataset(path) == dataset
    assert "split" not in path.read_text(encoding="utf-8")  # defaults left out
    assert read_grades(grades) == [{"query_id": "t001", "item_id": "k02", "grade": "3"}]


def test_the_tracked_dataset_names_real_target_dishes(store: CatalogStore) -> None:
    dataset = load_dataset(REPO_ROOT / "eval" / "datasets" / "retrieval.yaml")
    keys = set(store.dish_keys().values())

    assert all(q.target in keys for q in dataset.queries if q.target is not None)
    assert all(q.target is not None for q in dataset.queries if q.type == "known_item")


@pytest.fixture
def script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("label_queries", REPO_ROOT / "scripts" / "label_queries.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DATASET", tmp_path / "retrieval.yaml")
    return module


def test_script_imports_grades_then_splits_and_locks(script: ModuleType, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, model_cache_dir=tmp_path / "models")
    queries = tuple(query(f"t{n:03d}", f"q{n}") for n in range(1, 5))
    write_dataset(RetrievalDataset(queries=queries), script.DATASET)
    review = script.review_dir(settings)
    review.mkdir(parents=True)
    pool = [{"query_id": "t001", "to_grade": ["k02"], "targets": []}]
    (review / script.POOL_FILE).write_text(json.dumps(pool), encoding="utf-8")
    grades = tmp_path / "grades.csv"
    grades.write_text("query_id,item_id,grade\nt001,k02,2\n", encoding="utf-8")

    assert script.import_grades(settings, grades).queries[0].grades == {"k02": 2}
    locked = script.split(settings)
    assert locked.test_sha256 is not None
    assert {q.split for q in load_dataset(script.DATASET).queries} == {"dev", "test"}
