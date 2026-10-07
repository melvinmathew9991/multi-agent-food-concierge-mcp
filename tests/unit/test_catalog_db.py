"""The SQLite catalog: build, dish grouping, document text, and hard filters checked against a Python oracle."""

import random
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from food_concierge.config import REPO_ROOT
from food_concierge.errors import NotReadyError
from food_concierge.ingestion.allergens import DishAllergens, load_lexicon, tag_item
from food_concierge.ingestion.loader import CatalogLoad, load_catalog
from food_concierge.ingestion.taxonomy import Allergen, Cuisine, Diet
from food_concierge.storage.catalog_db import (
    DOC_TEXT_VERSION,
    CatalogStore,
    Filters,
    build_catalog_db,
    data_sha256,
    dish_key,
    document_text,
)

RAW = REPO_ROOT / "data" / "raw"
A = Allergen


@pytest.fixture(scope="module")
def loaded() -> CatalogLoad:
    return load_catalog(RAW)


@pytest.fixture(scope="module")
def tags(loaded: CatalogLoad) -> dict[str, DishAllergens]:
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    return {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}


@pytest.fixture(scope="module")
def store(
    loaded: CatalogLoad, tags: dict[str, DishAllergens], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[CatalogStore]:
    path = tmp_path_factory.mktemp("db") / "catalog.db"
    build_catalog_db(loaded.catalog, tags, path, lexicon_version=1, source_sha256="abc")
    opened = CatalogStore(path)
    yield opened
    opened.close()


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Paneer Butter Masala", "paneer-butter-masala"),
        ("  Chicken 65 ", "chicken-65"),
        ("Crème brûlée!", "creme-brulee"),
        ("Sev Tameta nu Shaak", "sev-tameta-nu-shaak"),
    ],
)
def test_dish_key(name: str, key: str) -> None:
    assert dish_key(name) == key


def test_document_text_is_semantic_only(loaded: CatalogLoad) -> None:
    item = next(i for i in loaded.catalog.items if i.item_id == "s05")  # Butter Chicken

    text = document_text(item)

    assert text.startswith("Butter Chicken. Tandoori chicken in a velvety")
    assert "Punjabi main." in text
    assert "Ingredients: tandoori chicken, tomato, butter" in text
    assert text.endswith("Non-vegetarian.")
    for number in (item.price_inr, item.kcal, item.protein_g):
        assert str(number) not in text


def test_build_writes_every_table(store: CatalogStore, loaded: CatalogLoad) -> None:
    assert store.meta() == {"doc_text_version": str(DOC_TEXT_VERSION), "lexicon_version": "1", "data_sha256": "abc"}
    assert len(store.doc_texts()) == len(loaded.catalog.items) == 150

    grouped = Counter(store.dish_keys().values())
    assert grouped["paneer-butter-masala"] == 2
    assert grouped["margherita-pizza"] == 2
    assert sum(1 for count in grouped.values() if count > 1) == 6


def test_ground_truth_never_enters_the_database(store: CatalogStore) -> None:
    sql = "SELECT group_concat(sql, ' ') FROM sqlite_master"
    schema = store._connection.execute(sql).fetchone()[0]
    assert "true" not in schema.lower()


def test_store_is_read_only(store: CatalogStore) -> None:
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        store._connection.execute("DELETE FROM items")


def test_missing_database_is_not_ready(tmp_path: Path) -> None:
    with pytest.raises(NotReadyError):
        CatalogStore(tmp_path / "catalog.db")


def test_rebuild_replaces_atomically_and_stores_tags_and_images(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "catalog", raw)
    (raw / "attributions.csv").write_text(
        "item_id,file_page_url,file_url,author,licence,licence_url,sha256,width,height,bytes\n"
        "fx005,https://commons.wikimedia.org/wiki/File:Dosa.jpg,https://upload.wikimedia.org/a/Dosa.jpg,"
        f"Cook,CC0 1.0,https://creativecommons.org/publicdomain/zero/1.0/,{'a' * 64},800,600,1000\n",
        encoding="utf-8",
    )
    fixture = load_catalog(raw)
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    fixture_tags = {item.item_id: tag_item(item, lexicon) for item in fixture.catalog.items}
    path = tmp_path / "processed" / "catalog.db"
    path.parent.mkdir()
    path.write_bytes(b"an older build")

    build_catalog_db(fixture.catalog, fixture_tags, path, lexicon_version=1, source_sha256="x")

    assert not path.with_suffix(".db.part").exists()
    connection = sqlite3.connect(path)
    rows = connection.execute("SELECT allergen, source, level, evidence FROM allergen_tags WHERE item_id = 'fx012'")
    assert ("crustaceans", "lexicon", "contains", "prawns") in rows.fetchall()
    assert connection.execute("SELECT unverified FROM items WHERE item_id = 'fx006'").fetchone() == (1,)
    assert connection.execute("SELECT label_allergens FROM items WHERE item_id = 'fx002'").fetchone() == (None,)
    assert connection.execute("SELECT item_id, licence FROM images").fetchall() == [("fx005", "CC0 1.0")]
    connection.close()


def test_data_hash_binds_names_and_bytes(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("1", encoding="utf-8")
    b.write_text("2", encoding="utf-8")
    first = data_sha256([a, b])

    assert data_sha256([b, a]) == first  # order of the arguments doesn't matter
    b.write_text("3", encoding="utf-8")
    assert data_sha256([a, b]) != first


def random_filters(rng: random.Random) -> Filters:
    return Filters(
        diet=rng.choice([None, *Diet]),
        eggless=rng.random() < 0.3,
        exclude_allergens=frozenset(rng.sample(list(Allergen), rng.choice([0, 0, 1, 2, 3]))),
        max_kcal=rng.choice([None, 250, 400, 600]),
        max_price_inr=rng.choice([None, 100, 250, 450]),
        cuisines=frozenset(rng.sample(list(Cuisine), rng.choice([0, 0, 1, 3]))),
        include_unverified=rng.random() < 0.3,
    )


def test_candidates_match_the_oracle(store: CatalogStore, loaded: CatalogLoad, tags: dict[str, DishAllergens]) -> None:
    # Property test: for many random filter combinations, SQL returns exactly the items Filters.admits accepts,
    # the plain-Python statement of the same rules.
    rng = random.Random(20261007)
    for _ in range(400):
        filters = random_filters(rng)
        expected = sorted(i.item_id for i in loaded.catalog.items if filters.admits(i, tags[i.item_id]))
        assert store.candidate_ids(filters) == expected, filters


def test_specific_filters(store: CatalogStore) -> None:
    vegetarian = set(store.candidate_ids(Filters(diet=Diet.VEGETARIAN)))
    assert {"t01", "k01"} <= vegetarian  # vegan and vegetarian dishes
    assert "k05" not in vegetarian  # chicken tikka masala

    no_gluten = set(store.candidate_ids(Filters(exclude_allergens=frozenset({A.GLUTEN}))))
    assert "s02" in no_gluten  # makki di roti
    assert "a10" in no_gluten  # mango sticky rice: glutinous rice has no gluten
    assert "k03" not in no_gluten  # butter naan
    assert "h02" not in no_gluten  # haleem: barley and wheat
    assert "c03" not in no_gluten  # pani puri is unverified (chutneys), so it is left out ...
    with_unverified = store.candidate_ids(Filters(exclude_allergens=frozenset({A.GLUTEN}), include_unverified=True))
    assert "c05" in with_unverified  # ... while bhel puri, unverified but not tagged gluten, can be offered

    eggless = set(store.candidate_ids(Filters(eggless=True, include_unverified=True)))
    assert "w07" not in eggless  # egg fried rice
    assert "f03" not in eggless  # caesar salad: egg in the dressing

    assert store.candidate_ids(Filters()) == sorted(store.doc_texts())
