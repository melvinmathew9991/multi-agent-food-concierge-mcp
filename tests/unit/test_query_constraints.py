"""Constraints a request states in words (audit H1): avoidance is read, a wish never is."""

import pytest

from food_concierge.ingestion.taxonomy import Allergen, Diet, parse_allergen_terms
from food_concierge.services.constraints import Constraints, constraints_in_text, describe, read_query

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
        # #142: the audit's misses
        ("nothing with nuts please, a sweet", None, NUTS),
        ("hold the peanuts, something crunchy", None, {A.PEANUTS}),
        ("skip the peanuts", None, {A.PEANUTS}),
        ("anything except peanuts", None, {A.PEANUTS}),
        ("keep it away from sesame", None, {A.SESAME}),
        ("celiac friendly bread", None, {A.GLUTEN}),
        ("no meat please", Diet.VEGETARIAN, set()),
        ("I don't eat meat", Diet.VEGETARIAN, set()),
        ("meatless monday", Diet.VEGETARIAN, set()),
        ("jain food", Diet.VEGETARIAN, {A.EGGS}),
        ("any curry except ones with cashews", None, {A.TREE_NUTS}),
        ("allergies: peanuts, sesame", None, {A.PEANUTS, A.SESAME}),
        ("lactose-intolerant friendly", None, {A.MILK}),
        ("nuts are a no for me", None, NUTS),
        ("eggs make me sick", None, {A.EGGS}),
        ("can't tolerate lactose", None, {A.MILK}),
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
        "not vegan, just want a curry",
        "doesn't have to be vegetarian",
        "no need for veg, I eat anything",
        "not allergic to nuts, any dessert",
        "anything but boring",
        "skip the starters, mains only",
        "except for dessert, anything goes",
    ],
)
def test_wishes_and_unrelated_words_add_nothing(text: str) -> None:
    assert constraints_in_text(text) == Constraints()


@pytest.mark.parametrize(
    ("text", "untracked"),
    [
        ("no onion no garlic", ("onion", "garlic")),
        ("nothing with pork or beef", ("pork", "beef")),
        ("onion-free gravy", ("onion",)),
        ("halal chicken", ("halal",)),
        ("Kosher meal, no bacon", ("pork", "kosher")),
        ("jain thali", ("onion", "garlic", "root vegetables")),
        ("onion bhaji", ()),
    ],
)
def test_what_the_menu_does_not_record_is_named_not_applied(text: str, untracked: tuple[str, ...]) -> None:
    reading = read_query(text)

    assert reading.untracked == untracked
    assert reading.constraints.exclude_allergens == ({A.EGGS} if "jain" in text else set())


@pytest.mark.parametrize(
    ("text", "unread"),
    [
        ("no, I want prawns", {A.CRUSTACEANS}),
        ("no rush, something with peanuts", {A.PEANUTS}),
        ("a cake, not with egg", set()),  # read, so nothing is left unread
        ("peanut chikki", set()),  # no negation anywhere: a wish
    ],
)
def test_an_allergen_mentioned_near_a_negation_but_not_read_is_reported(text: str, unread: set[Allergen]) -> None:
    assert read_query(text).unread == unread


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
