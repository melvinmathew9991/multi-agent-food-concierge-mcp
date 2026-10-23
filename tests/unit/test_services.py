"""Catalog and allergen services: dish lookups and allergen checks on the real catalog, offline."""

import random
from collections.abc import Iterator

import pytest

from food_concierge.config import REPO_ROOT, get_settings
from food_concierge.errors import InputValidationError, NotFoundError
from food_concierge.ingestion.allergens import DishAllergens, Level, Source, load_lexicon, tag_item
from food_concierge.ingestion.descriptions import DESCRIPTIONS_FILE, read_descriptions, vision_texts
from food_concierge.ingestion.loader import load_catalog
from food_concierge.ingestion.taxonomy import Allergen, Diet
from food_concierge.services.allergens import AllergenService
from food_concierge.services.catalog import MAX_DISHES, CatalogService, DishDetail, DishSummary
from food_concierge.storage.catalog_db import CatalogStore, Filters, build_catalog_db

A = Allergen


@pytest.fixture(scope="module")
def tags() -> dict[str, DishAllergens]:
    # With the tracked photo descriptions, so vision's may_contain tags are there as the build stores them.
    loaded = load_catalog(REPO_ROOT / "data" / "raw")
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    vision = vision_texts(loaded.catalog.photos, read_descriptions(get_settings().processed_dir / DESCRIPTIONS_FILE))
    return {item.item_id: tag_item(item, lexicon, vision.get(item.item_id)) for item in loaded.catalog.items}


@pytest.fixture(scope="module")
def store(tags: dict[str, DishAllergens], tmp_path_factory: pytest.TempPathFactory) -> Iterator[CatalogStore]:
    path = tmp_path_factory.mktemp("db") / "catalog.db"
    build_catalog_db(load_catalog(REPO_ROOT / "data" / "raw").catalog, tags, path, lexicon_version=1, source_sha256="x")
    opened = CatalogStore(path)
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def catalog(store: CatalogStore) -> CatalogService:
    return CatalogService(store)


@pytest.fixture(scope="module")
def allergens(store: CatalogStore) -> AllergenService:
    return AllergenService(store)


def test_a_dish_in_full(catalog: CatalogService) -> None:
    dish = catalog.get_dish("s05")

    assert (dish.name, dish.restaurant, dish.restaurant_id, dish.restaurant_area) == (
        "Butter Chicken",
        "Sarson Milestone Dhaba",
        "r_sarson",
        "Highway Junction",
    )
    assert (dish.diet, dish.eggless, dish.price_inr, dish.kcal, dish.serves_min, dish.serves_max) == (
        Diet.NON_VEGETARIAN,
        True,
        440,
        510,
        2,
        2,
    )
    assert dish.ingredients[:3] == ["tandoori chicken", "tomato", "butter"]
    assert dish.label_allergens == [A.MILK, A.TREE_NUTS]
    assert dish.contains == [A.MILK, A.TREE_NUTS]
    assert dish.may_contain == [A.GLUTEN, A.WHEAT]  # "naan" and "bread" in the photo's description
    assert {(t.source, t.evidence) for t in dish.tags if t.allergen is A.TREE_NUTS} == {
        (Source.LABEL, "label"),
        (Source.LEXICON, "cashew paste"),
    }
    assert dish.photo is not None
    assert dish.photo.licence == "CC BY-SA 4.0"
    assert dish.photo_url == dish.photo.file_url


def test_a_dish_without_a_photo_or_a_checkable_recipe(catalog: CatalogService) -> None:
    dish = catalog.get_dish("a01")  # Thai Green Curry with Tofu

    assert dish.photo is None
    assert dish.photo_url is None
    assert dish.unverified


def test_every_dish_reports_the_tags_search_filters_on(catalog: CatalogService, tags: dict[str, DishAllergens]) -> None:
    ids = sorted(tags)
    dishes = [
        dish for start in range(0, len(ids), MAX_DISHES) for dish in catalog.get_dishes(ids[start : start + MAX_DISHES])
    ]

    assert [dish.item_id for dish in dishes] == ids
    for dish in dishes:
        expected = tags[dish.item_id]
        assert dish.contains == sorted(expected.contains)
        assert dish.may_contain == sorted(expected.may_contain)
        assert dish.unverified is expected.unverified
    # A blank label (not declared) stays apart from a declared "none".
    assert sum(dish.label_allergens is None for dish in dishes) == 47
    assert [dish.item_id for dish in dishes if dish.label_allergens == []] == ["e01", "g08", "t03", "v05"]


def test_ids_are_trimmed_lower_cased_and_de_duplicated_in_order(catalog: CatalogService) -> None:
    assert [dish.item_id for dish in catalog.get_dishes([" S05 ", "a01", "s05"])] == ["s05", "a01"]


def test_summaries_are_the_short_form(catalog: CatalogService) -> None:
    [summary] = catalog.summaries(["s05"])

    assert type(summary) is DishSummary
    assert not isinstance(summary, DishDetail)
    assert (summary.name, summary.price_inr, summary.contains) == ("Butter Chicken", 440, [A.MILK, A.TREE_NUTS])


def test_unknown_ids_are_named(catalog: CatalogService, allergens: AllergenService) -> None:
    with pytest.raises(NotFoundError, match="zz99, zz98"):
        catalog.get_dishes(["s05", "zz99", "zz98"])
    with pytest.raises(NotFoundError, match="zz99"):
        catalog.get_dish("zz99")
    with pytest.raises(NotFoundError, match="zz99"):
        allergens.check(["zz99"])


@pytest.mark.parametrize(
    ("ids", "message"),
    [([], "at least one"), ([" "], "at least one"), ([f"x{n}" for n in range(MAX_DISHES + 1)], "at most 20")],
    ids=["none", "blank", "too-many"],
)
def test_requests_without_ids_or_with_too_many_are_refused(
    catalog: CatalogService, ids: list[str], message: str
) -> None:
    with pytest.raises(InputValidationError, match=message):
        catalog.get_dishes(ids)


def test_check_reports_only_the_allergens_asked_about(allergens: AllergenService) -> None:
    [check] = allergens.check(["s05"], [A.MILK, A.PEANUTS])

    assert (check.contains, check.may_contain, check.unverified, check.clear) == ([A.MILK], [], False, False)
    assert {tag.allergen for tag in check.evidence} == {A.MILK}
    assert {(tag.level, tag.evidence) for tag in check.evidence} >= {(Level.CONTAINS, "butter")}


def test_check_with_no_allergens_asks_about_all_of_them(allergens: AllergenService) -> None:
    [check] = allergens.check(["s05"])

    assert (check.contains, check.may_contain) == ([A.MILK, A.TREE_NUTS], [A.GLUTEN, A.WHEAT])


@pytest.mark.parametrize(
    ("item_id", "asked", "contains", "may_contain", "unverified", "clear"),
    [
        ("a04", [A.SESAME], [], [A.SESAME], False, False),  # only the photo suggests it: a warning
        ("a01", [A.SESAME], [], [], True, False),  # nothing found, but the recipe can't be checked
        ("t04", [A.PEANUTS, A.MILK], [], [], False, True),  # Medu Vada: no tags, recipe checked
    ],
    ids=["may-contain", "unverified", "clear"],
)
def test_check_is_clear_only_when_nothing_is_found_and_the_recipe_is_checked(
    allergens: AllergenService,
    item_id: str,
    asked: list[Allergen],
    contains: list[Allergen],
    may_contain: list[Allergen],
    unverified: bool,
    clear: bool,
) -> None:
    [check] = allergens.check([item_id], asked)

    assert (check.contains, check.may_contain, check.unverified, check.clear) == (
        contains,
        may_contain,
        unverified,
        clear,
    )


def test_check_agrees_with_search(
    store: CatalogStore, allergens: AllergenService, tags: dict[str, DishAllergens]
) -> None:
    # A dish search returns to someone avoiding these allergens is exactly one with no "contains" and a checked
    # recipe: the check and the filter read the same tags.
    rng = random.Random(53)
    ids = sorted(tags)
    for _ in range(30):
        asked = rng.sample(list(Allergen), rng.randint(1, 4))
        allowed = set(store.candidate_ids(Filters(exclude_allergens=frozenset(asked))))
        sample = rng.sample(ids, MAX_DISHES)
        for check in allergens.check(sample, asked):
            assert (check.item_id in allowed) is (not check.contains and not check.unverified)
