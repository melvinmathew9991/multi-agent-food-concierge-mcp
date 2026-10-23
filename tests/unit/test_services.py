"""Domain services on the real catalog, offline: dish lookups, allergen checks, meal totals and constraints."""

import random
from collections.abc import Iterator

import pytest

from food_concierge.config import REPO_ROOT, get_settings
from food_concierge.errors import InputValidationError, NotFoundError
from food_concierge.ingestion.allergens import DishAllergens, Level, Source, load_lexicon, tag_item
from food_concierge.ingestion.descriptions import DESCRIPTIONS_FILE, read_descriptions, vision_texts
from food_concierge.ingestion.loader import load_catalog
from food_concierge.ingestion.taxonomy import Allergen, Cuisine, Diet
from food_concierge.services.allergens import AllergenService
from food_concierge.services.catalog import MAX_DISHES, CatalogService, DishDetail, DishSummary
from food_concierge.services.constraints import (
    Constraints,
    ConstraintService,
    Rule,
    dish_violations,
    merge,
    parse_constraints,
)
from food_concierge.services.nutrition import NutritionService
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


# Meal totals


def test_meal_totals_sum_the_dishes_as_served_and_count_repeats(catalog: CatalogService, store: CatalogStore) -> None:
    nutrition = NutritionService(store)
    s05, a01 = catalog.get_dishes(["s05", "a01"])

    totals = nutrition.meal_totals(["S05", "a01", "s05"])

    assert totals.dish_ids == ["s05", "a01", "s05"]
    assert totals.kcal == 2 * s05.kcal + a01.kcal
    assert totals.price_inr == 2 * s05.price_inr + a01.price_inr
    assert totals.protein_g == round(2 * s05.protein_g + a01.protein_g, 1)
    assert (totals.serves_min, totals.serves_max) == (
        2 * s05.serves_min + a01.serves_min,
        2 * s05.serves_max + a01.serves_max,
    )
    with pytest.raises(NotFoundError, match="zz99"):
        nutrition.meal_totals(["zz99", "zz99"])


# Constraints


def test_constraints_are_parsed_from_loose_input() -> None:
    parsed = parse_constraints(
        {"diet": "Veg", "exclude_allergens": "Dairy, nuts", "cuisines": "Punjabi; south indian", "max_price_inr": 300}
    )

    assert parsed.diet is Diet.VEGETARIAN
    assert parsed.exclude_allergens == {A.MILK, A.PEANUTS, A.TREE_NUTS}
    assert parsed.cuisines == {Cuisine.PUNJABI, Cuisine.SOUTH_INDIAN}
    assert parsed.max_price_inr == 300
    eggless = parse_constraints({"diet": "Eggless", "exclude_allergens": ["wheat"], "cuisines": ["Thai", " "]})
    assert (eggless.diet, eggless.eggless, eggless.cuisines) == (Diet.VEGETARIAN, True, frozenset({Cuisine.THAI}))
    assert eggless.exclude_allergens == {A.WHEAT, A.GLUTEN}  # implied, as the tags are
    assert eggless.excluded == {A.WHEAT, A.GLUTEN, A.EGGS}
    assert parse_constraints({"diet": "", "exclude_allergens": [A.SOY]}) == parse_constraints(
        {"exclude_allergens": "soya"}
    )


@pytest.mark.parametrize(
    ("data", "named"),
    [
        ({"exclude_allergens": "milk, kiwi"}, "exclude_allergens: Value error, unknown allergen term(s): kiwi"),
        ({"exclude_allergens": 5}, "exclude_allergens"),
        ({"cuisines": "martian"}, "cuisines"),
        ({"cuisines": 5}, "cuisines"),
        ({"diet": "pescatarian"}, "diet"),
        ({"max_kcal": 0}, "max_kcal"),
        ({"spicy": True}, "spicy"),
    ],
    ids=["unknown-allergen", "allergens-not-text", "unknown-cuisine", "cuisines-not-text", "diet", "cap", "extra"],
)
def test_constraints_that_cannot_be_read_are_refused_by_name(data: dict[str, object], named: str) -> None:
    with pytest.raises(InputValidationError, match="could not be read") as raised:
        parse_constraints(data)
    assert named in raised.value.message


def test_merged_constraints_only_tighten() -> None:
    first = Constraints(
        diet=Diet.NON_VEGETARIAN,
        exclude_allergens=frozenset({A.MILK}),
        max_kcal=800,
        cuisines=frozenset({Cuisine.PUNJABI}),
        include_unverified=True,
        meal_budget_inr=1000,
    )
    second = Constraints(
        diet=Diet.VEGETARIAN,
        eggless=True,
        exclude_allergens=frozenset({A.PEANUTS}),
        max_kcal=600,
        max_price_inr=250,
        cuisines=frozenset({Cuisine.ITALIAN}),
        meal_max_kcal=1200,
        meal_budget_inr=1500,
    )

    assert merge(first, second) == Constraints(
        diet=Diet.VEGETARIAN,
        eggless=True,
        exclude_allergens=frozenset({A.MILK, A.PEANUTS}),
        max_kcal=600,
        max_price_inr=250,
        cuisines=frozenset({Cuisine.ITALIAN}),  # a preference: the latest wins
        include_unverified=False,  # confirmed by one diner only
        meal_max_kcal=1200,
        meal_budget_inr=1000,
    )
    assert merge(second, Constraints()).cuisines == {Cuisine.ITALIAN}  # no cuisine given keeps the earlier one
    assert merge(Constraints(diet=Diet.VEGAN), first).diet is Diet.VEGAN
    assert merge() == Constraints()


@pytest.fixture(scope="module")
def constraints_service(store: CatalogStore) -> ConstraintService:
    return ConstraintService(store)


def test_check_names_every_rule_a_dish_breaks(constraints_service: ConstraintService) -> None:
    asked = Constraints(
        diet=Diet.VEGETARIAN,
        exclude_allergens=frozenset({A.MILK}),
        max_kcal=400,
        max_price_inr=300,
        cuisines=frozenset({Cuisine.ITALIAN}),
    )

    check = constraints_service.check(["s05"], asked)

    assert not check.ok
    assert [v.rule for v in check.violations] == [Rule.DIET, Rule.ALLERGEN, Rule.MAX_KCAL, Rule.MAX_PRICE, Rule.CUISINE]
    assert [v.message for v in check.violations] == [
        "Butter Chicken is non-vegetarian, not vegetarian.",
        "Butter Chicken contains milk.",
        "Butter Chicken has 510 kcal, over 400.",
        "Butter Chicken costs ₹440, over ₹300.",
        "Butter Chicken is punjabi, not a cuisine asked for.",
    ]
    assert {v.item_id for v in check.violations} == {"s05"}


def test_an_unverified_recipe_needs_the_diners_confirmation(constraints_service: ConstraintService) -> None:
    eggless = Constraints(eggless=True)

    before = constraints_service.check(["b10"], eggless)  # Egg Chicken Roll: egg, and an opaque ingredient
    after = constraints_service.check(["b10"], eggless.model_copy(update={"include_unverified": True}))

    assert [v.rule for v in before.violations] == [Rule.EGGLESS, Rule.ALLERGEN, Rule.UNVERIFIED]
    assert [v.rule for v in after.violations] == [Rule.EGGLESS, Rule.ALLERGEN]


def test_check_holds_the_meal_to_its_caps_and_warns_on_photo_only_allergens(
    constraints_service: ConstraintService,
) -> None:
    check = constraints_service.check(
        ["s05", "s05"], Constraints(exclude_allergens=frozenset({A.GLUTEN}), meal_max_kcal=900, meal_budget_inr=800)
    )

    assert [(v.rule, v.item_id, v.message) for v in check.violations] == [
        (Rule.MEAL_KCAL, None, "The meal has 1020 kcal, over 900."),
        (Rule.MEAL_BUDGET, None, "The meal costs ₹880, over the budget of ₹800."),
    ]
    assert check.warnings == ["Butter Chicken may contain gluten (seen in its photo only)."]
    within = constraints_service.check(
        ["s05"], Constraints(exclude_allergens=frozenset({A.GLUTEN}), meal_budget_inr=440)
    )
    assert within.ok
    assert within.warnings == check.warnings  # a warning doesn't block


def _random_constraints(rng: random.Random) -> Constraints:
    return Constraints(
        diet=rng.choice([None, *Diet]),
        eggless=rng.random() < 0.3,
        exclude_allergens=frozenset(rng.sample(list(Allergen), rng.randint(0, 3))),
        max_kcal=rng.choice([None, 300, 500, 700]),
        max_price_inr=rng.choice([None, 150, 300, 450]),
        cuisines=frozenset(rng.sample(list(Cuisine), rng.choice([0, 0, 1, 3]))),
        include_unverified=rng.random() < 0.3,
    )


def test_check_agrees_with_search_on_every_dish(
    catalog: CatalogService, store: CatalogStore, tags: dict[str, DishAllergens]
) -> None:
    # The same per-dish rules, in SQL for search and in Python for the check: they must give the same verdict.
    ids = sorted(tags)
    dishes = [
        dish for start in range(0, len(ids), MAX_DISHES) for dish in catalog.get_dishes(ids[start : start + MAX_DISHES])
    ]
    rng = random.Random(107)
    for _ in range(100):
        asked = _random_constraints(rng)
        allowed = set(store.candidate_ids(asked.filters))
        for dish in dishes:
            assert (dish.item_id in allowed) is (not dish_violations(dish, asked)), (dish.item_id, asked)
