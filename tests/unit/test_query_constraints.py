"""Constraints a request states in words (audit H1): avoidance is read, a wish never is."""

import pytest

from food_concierge.ingestion.taxonomy import Allergen, Diet, parse_allergen_terms
from food_concierge.services.constraints import Constraints, constraints_in_text, describe

A = Allergen
NUTS = frozenset({A.PEANUTS, A.TREE_NUTS})


@pytest.mark.parametrize(
    ("text", "diet", "allergens"),
    [
        ("peanut-free snack", None, {A.PEANUTS}),
        ("snack without peanuts", None, {A.PEANUTS}),
        ("dairy free curry", None, {A.MILK}),
        ("vegan dessert", Diet.VEGAN, set()),
        ("plant-based bowl", Diet.VEGAN, set()),
        ("no egg cake", None, {A.EGGS}),
        ("eggless cake", None, {A.EGGS}),
        ("nut allergy friendly sweets", None, NUTS),
        ("no dairy or nuts please", None, {A.MILK} | NUTS),
        ("gluten-free and vegan", Diet.VEGAN, {A.GLUTEN}),
        ("I can't eat shellfish", None, {A.CRUSTACEANS, A.MOLLUSCS}),
        ("allergic to cashews", None, {A.TREE_NUTS}),
        ("free of soy, sesame and mustard", None, {A.SOY, A.SESAME, A.MUSTARD}),
        ("no prawns", None, {A.CRUSTACEANS}),
        ("lactose intolerant", None, {A.MILK}),
        ("Non-dairy latte", None, {A.MILK}),
        ("wheat-free roti", None, {A.WHEAT, A.GLUTEN}),  # wheat implies gluten, as in the tags
        ("nut-free, egg-free brownie", None, NUTS | {A.EGGS}),
        ("veg biryani", Diet.VEGETARIAN, set()),
        ("PURE VEG thali", Diet.VEGETARIAN, set()),
    ],
)
def test_avoided_allergens_and_diets_are_read(text: str, diet: Diet | None, allergens: set[Allergen]) -> None:
    stated = constraints_in_text(text)

    assert stated.diet == diet
    assert stated.exclude_allergens == allergens
    assert stated.eggless == (A.EGGS in allergens)


@pytest.mark.parametrize(
    "text",
    [
        "peanut chikki",
        "prawn curry",
        "egg curry",
        "coconut curry",
        "doughnut",
        "eggplant bharta",
        "non veg thali",
        "non-vegetarian platter",
        "Nonveg biryani",
        "vegetable soup",
        "sugar free dessert",
        "hold the onions",
        "something sweet with milk and nuts",
    ],
)
def test_wishes_and_unrelated_words_add_nothing(text: str) -> None:
    assert constraints_in_text(text) == Constraints()


def test_a_confirmation_the_diner_gave_is_carried_through() -> None:
    assert constraints_in_text("peanut-free", include_unverified=True).include_unverified
    assert not constraints_in_text("peanut-free").include_unverified


def test_constraints_are_described_in_words() -> None:
    assert describe(Constraints(diet=Diet.VEGAN, exclude_allergens=NUTS)) == "vegan, no peanuts or tree nuts"
    assert describe(Constraints(eggless=True)) == "no eggs"
    assert describe(Constraints()) == ""


@pytest.mark.parametrize(
    ("word", "allergen"),
    [("cashews", A.TREE_NUTS), ("Brazil nuts", A.TREE_NUTS), ("shrimp", A.CRUSTACEANS), ("squid", A.MOLLUSCS)],
)
def test_the_words_for_one_member_of_a_group_name_the_group(word: str, allergen: Allergen) -> None:
    assert parse_allergen_terms(word) == {allergen}
