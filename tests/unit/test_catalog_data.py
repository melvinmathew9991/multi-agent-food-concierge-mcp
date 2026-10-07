"""Invariants of the authored catalog in data/raw (it grows in reviewed batches)."""

from collections import Counter

import pytest

from food_concierge.config import REPO_ROOT
from food_concierge.ingestion.loader import CatalogLoad, load_catalog

RAW = REPO_ROOT / "data" / "raw"
INDIAN = {
    "north_indian", "south_indian", "punjabi", "mughlai", "hyderabadi", "bengali", "gujarati", "kerala",
    "maharashtrian", "indian_street_food", "indo_chinese",
}  # fmt: skip


@pytest.fixture(scope="module")
def loaded() -> CatalogLoad:
    return load_catalog(RAW)


def test_catalog_loads_without_warnings(loaded: CatalogLoad) -> None:
    # Every dish's calories agree with its macros (4P + 4C + 9F within 25%).
    assert loaded.warnings == ()


def test_every_restaurant_serves_dishes_and_names_are_unique(loaded: CatalogLoad) -> None:
    served = Counter(item.restaurant_id for item in loaded.catalog.items)
    names = [r.name.casefold() for r in loaded.catalog.restaurants]

    assert {r.restaurant_id for r in loaded.catalog.restaurants} == set(served)
    assert len(names) == len(set(names))


def test_label_gaps_are_deliberate_and_bounded(loaded: CatalogLoad) -> None:
    # Real menus are often unlabelled; the catalog keeps about 30% of dishes that way (docs: data/raw/README.md).
    unlabelled = sum(item.label_allergens is None for item in loaded.catalog.items)

    assert 0.2 <= unlabelled / len(loaded.catalog.items) <= 0.4


def test_catalog_shape_matches_the_plan(loaded: CatalogLoad) -> None:
    # Phase 2 plan: ~150 dishes at ~15 restaurants, mainly Indian, with some dishes served at several restaurants.
    items = loaded.catalog.items
    indian = sum(item.cuisine.value in INDIAN for item in items)
    repeated = [name for name, count in Counter(item.name.casefold() for item in items).items() if count > 1]

    assert len(loaded.catalog.restaurants) == 15
    assert len(items) == 150
    assert 0.6 <= indian / len(items) <= 0.75
    assert len(repeated) >= 5


def test_every_dish_has_reviewed_allergens(loaded: CatalogLoad) -> None:
    assert set(loaded.true_allergens) == {item.item_id for item in loaded.catalog.items}
