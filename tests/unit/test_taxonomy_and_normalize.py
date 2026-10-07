"""Allergen taxonomy, user terms, diet checks and nutrition consistency."""

import pytest

from food_concierge.errors import CatalogIssue, CatalogValidationError
from food_concierge.ingestion.normalize import atwater_deviation, diet_problems, parse_serves
from food_concierge.ingestion.taxonomy import (
    EU_14,
    US_BIG_9,
    Allergen,
    Diet,
    parse_allergen_keys,
    parse_allergen_terms,
    slug,
)

A = Allergen


def test_taxonomy_is_eu14_union_big9() -> None:
    assert len(EU_14) == 14
    assert len(US_BIG_9) == 9
    assert set(Allergen) == EU_14 | US_BIG_9
    assert {A.WHEAT} == US_BIG_9 - EU_14


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("milk", {A.MILK}),
        ("Dairy; LACTOSE", {A.MILK}),
        ("shellfish", {A.CRUSTACEANS, A.MOLLUSCS}),
        ("nuts", {A.PEANUTS, A.TREE_NUTS}),
        ("tree nuts, groundnut", {A.TREE_NUTS, A.PEANUTS}),
        ("tree_nuts", {A.TREE_NUTS}),
        ("coeliac", {A.GLUTEN}),
        ("wheat", {A.WHEAT, A.GLUTEN}),  # wheat implies gluten
        ("soya; sesame seeds; sulfites", {A.SOY, A.SESAME, A.SULPHITES}),
        ("  ;  ", set()),
    ],
)
def test_user_terms_map_onto_every_key_they_may_mean(text: str, expected: set[Allergen]) -> None:
    assert parse_allergen_terms(text) == expected


def test_unknown_allergen_terms_are_named_not_dropped() -> None:
    with pytest.raises(ValueError, match="kiwi, chocolate"):
        parse_allergen_terms("milk; kiwi; chocolate")


def test_ground_truth_accepts_canonical_keys_only() -> None:
    assert parse_allergen_keys("milk;; tree_nuts") == {A.MILK, A.TREE_NUTS}
    with pytest.raises(ValueError, match="not an allergen key: dairy"):
        parse_allergen_keys("dairy")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Indo-Chinese", "indo_chinese"), ("  North   Indian ", "north_indian"), ("non-veg", "non_veg")],
)
def test_slug(text: str, expected: str) -> None:
    assert slug(text) == expected


@pytest.mark.parametrize(("text", "expected"), [("2", (2, 2)), (" 2 - 3 ", (2, 3)), ("1-1", (1, 1))])
def test_parse_serves(text: str, expected: tuple[int, int]) -> None:
    assert parse_serves(text) == expected


@pytest.mark.parametrize("text", ["", "two", "2-", "2 to 3", "1.5"])
def test_parse_serves_rejects_other_shapes(text: str) -> None:
    with pytest.raises(ValueError, match="number of people"):
        parse_serves(text)


@pytest.mark.parametrize(
    ("diet", "egg", "ingredients", "problem"),
    [
        (Diet.VEGAN, False, ("rice", "ghee"), "vegan dish lists animal products: ghee"),
        (Diet.VEGAN, False, ("bread", "honey"), "vegan dish lists animal products: honey"),
        (Diet.VEGAN, True, ("rice",), "vegan dish is marked contains_egg"),
        (Diet.VEGETARIAN, False, ("rice", "Chicken stock"), "vegetarian dish lists meat or fish: Chicken stock"),
        (Diet.VEGETARIAN, False, ("pasta", "pork sausages"), "vegetarian dish lists meat or fish: pork sausages"),
        (Diet.VEGAN, False, ("noodles", "fish sauce"), "vegan dish lists meat or fish: fish sauce"),
        (Diet.VEGETARIAN, False, ("bread", "eggs"), "lists egg but contains_egg is no: eggs"),
        (Diet.NON_VEGETARIAN, False, ("mutton", "mayonnaise"), "lists egg but contains_egg is no: mayonnaise"),
    ],
)
def test_diet_contradictions_are_found(diet: Diet, egg: bool, ingredients: tuple[str, ...], problem: str) -> None:
    assert problem in diet_problems(diet, egg, ingredients)


@pytest.mark.parametrize(
    ("diet", "egg", "ingredients"),
    [
        (Diet.VEGAN, False, ("coconut milk", "peanut butter", "cocoa butter", "eggplant", "soy yogurt")),
        (Diet.VEGAN, False, ("vegan mayonnaise", "dairy-free cheese", "butternut squash", "chickpeas")),
        (Diet.VEGETARIAN, False, ("eggless sponge", "goat cheese", "honeydew melon", "graham crackers")),
        (Diet.VEGETARIAN, True, ("paneer", "egg")),
        (Diet.NON_VEGETARIAN, True, ("chicken", "egg", "ghee")),
    ],
)
def test_consistent_diets_and_plant_based_names_pass(diet: Diet, egg: bool, ingredients: tuple[str, ...]) -> None:
    assert diet_problems(diet, egg, ingredients) == []


def test_atwater_deviation() -> None:
    assert atwater_deviation(400, 10, 50, 20) == pytest.approx(0.05)  # 4*10 + 4*50 + 9*20 = 420
    assert atwater_deviation(200, 10, 50, 20) == pytest.approx(1.1)


def test_catalog_issue_text_and_error() -> None:
    row_issue = CatalogIssue("menu.csv", 4, "kcal", "too high")
    file_issue = CatalogIssue("menu.csv", None, None, "file not found")
    assert str(row_issue) == "menu.csv:4 kcal: too high"
    assert str(file_issue) == "menu.csv: file not found"

    error = CatalogValidationError([row_issue, file_issue])
    assert error.issues == (row_issue, file_issue)
    assert error.code == "catalog_invalid"
    assert error.http_status == 500
    assert error.message == "The catalog files have 2 problem(s)."
