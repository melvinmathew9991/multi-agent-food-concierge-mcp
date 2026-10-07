"""Controlled vocabularies for the catalog: allergens, diets, cuisines and categories.

Allergens follow the union of EU-14 and US Big-9 (audit A12): 15 keys, because the Big-9 adds wheat to the EU list.
Diet semantics follow audit A11: vegetarian is lacto-ovo, and eggs are tracked separately so the
``eggless`` filter does not depend on how a restaurant labels its dishes.

Free-text allergen terms (restaurant labels now, user requests from Phase 5) map onto the keys through
one table. A term maps to every key it may mean ("nuts" → peanuts and tree nuts, "shellfish" →
crustaceans and molluscs), because for exclusion a wider match is the safe one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum


class Allergen(StrEnum):
    GLUTEN = "gluten"
    WHEAT = "wheat"
    CRUSTACEANS = "crustaceans"
    MOLLUSCS = "molluscs"
    FISH = "fish"
    EGGS = "eggs"
    MILK = "milk"
    PEANUTS = "peanuts"
    TREE_NUTS = "tree_nuts"
    SOY = "soy"
    SESAME = "sesame"
    MUSTARD = "mustard"
    CELERY = "celery"
    LUPIN = "lupin"
    SULPHITES = "sulphites"


A = Allergen

EU_14: frozenset[Allergen] = frozenset(
    {A.GLUTEN, A.CRUSTACEANS, A.MOLLUSCS, A.FISH, A.EGGS, A.MILK, A.PEANUTS, A.TREE_NUTS}
    | {A.SOY, A.SESAME, A.MUSTARD, A.CELERY, A.LUPIN, A.SULPHITES}
)
US_BIG_9: frozenset[Allergen] = frozenset(
    {A.MILK, A.EGGS, A.FISH, A.CRUSTACEANS, A.TREE_NUTS, A.PEANUTS, A.WHEAT, A.SOY, A.SESAME}
)

# A key that implies others: wheat always contains gluten. Ground truth must state both; parsed terms get both.
IMPLIES: dict[Allergen, frozenset[Allergen]] = {A.WHEAT: frozenset({A.GLUTEN})}

USER_TERMS: dict[str, frozenset[Allergen]] = {
    **{a.value.replace("_", " "): frozenset({a}) for a in Allergen},
    "cereals containing gluten": frozenset({A.GLUTEN}),
    "coeliac": frozenset({A.GLUTEN}),
    "celiac": frozenset({A.GLUTEN}),
    "crustacean": frozenset({A.CRUSTACEANS}),
    "mollusc": frozenset({A.MOLLUSCS}),
    "mollusk": frozenset({A.MOLLUSCS}),
    "molluscs and shellfish": frozenset({A.CRUSTACEANS, A.MOLLUSCS}),
    "shellfish": frozenset({A.CRUSTACEANS, A.MOLLUSCS}),
    "seafood": frozenset({A.FISH, A.CRUSTACEANS, A.MOLLUSCS}),
    "egg": frozenset({A.EGGS}),
    "dairy": frozenset({A.MILK}),
    "lactose": frozenset({A.MILK}),
    "peanut": frozenset({A.PEANUTS}),
    "groundnut": frozenset({A.PEANUTS}),
    "groundnuts": frozenset({A.PEANUTS}),
    "tree nut": frozenset({A.TREE_NUTS}),
    "nut": frozenset({A.PEANUTS, A.TREE_NUTS}),
    "nuts": frozenset({A.PEANUTS, A.TREE_NUTS}),
    "soya": frozenset({A.SOY}),
    "soybean": frozenset({A.SOY}),
    "soybeans": frozenset({A.SOY}),
    "sesame seeds": frozenset({A.SESAME}),
    "lupine": frozenset({A.LUPIN}),
    "sulfites": frozenset({A.SULPHITES}),
    "sulphur dioxide": frozenset({A.SULPHITES}),
    "sulfur dioxide": frozenset({A.SULPHITES}),
}


class Diet(StrEnum):
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"  # lacto-ovo: may contain milk and eggs; `contains_egg` drives `eggless`
    NON_VEGETARIAN = "non_vegetarian"


DIET_ALIASES: dict[str, Diet] = {"veg": Diet.VEGETARIAN, "non_veg": Diet.NON_VEGETARIAN, "nonveg": Diet.NON_VEGETARIAN}


class Cuisine(StrEnum):
    NORTH_INDIAN = "north_indian"
    SOUTH_INDIAN = "south_indian"
    PUNJABI = "punjabi"
    MUGHLAI = "mughlai"
    HYDERABADI = "hyderabadi"
    BENGALI = "bengali"
    GUJARATI = "gujarati"
    KERALA = "kerala"
    MAHARASHTRIAN = "maharashtrian"
    INDIAN_STREET_FOOD = "indian_street_food"
    INDO_CHINESE = "indo_chinese"
    CONTINENTAL = "continental"
    ITALIAN = "italian"
    AMERICAN = "american"
    MEDITERRANEAN = "mediterranean"
    MIDDLE_EASTERN = "middle_eastern"
    MEXICAN = "mexican"
    CHINESE = "chinese"
    THAI = "thai"
    JAPANESE = "japanese"
    KOREAN = "korean"


class Category(StrEnum):
    # What the dish is on a menu, kept apart from where it comes from (the course data mixed the two: audit #13).
    BREAKFAST = "breakfast"
    STARTER = "starter"
    SOUP = "soup"
    SALAD = "salad"
    MAIN = "main"
    BREAD = "bread"
    RICE = "rice"
    SIDE = "side"
    SNACK = "snack"
    THALI = "thali"
    DESSERT = "dessert"
    BEVERAGE = "beverage"


_SEPARATORS = re.compile(r"[;,]")


def slug(text: str) -> str:
    """Lower-case with runs of spaces and hyphens as one underscore: "Indo-Chinese" → "indo_chinese"."""
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def with_implied(allergens: Iterable[Allergen]) -> frozenset[Allergen]:
    found = set(allergens)
    for key in list(found):
        found |= IMPLIES.get(key, frozenset())
    return frozenset(found)


def missing_implied(allergens: frozenset[Allergen]) -> frozenset[Allergen]:
    return with_implied(allergens) - allergens


def parse_allergen_terms(text: str) -> frozenset[Allergen]:
    """Map free-text terms ("Dairy; nuts, shellfish") onto allergen keys, adding implied ones.

    Raises ``ValueError`` naming every term it doesn't know: an unknown term must never be dropped silently.
    """
    found: set[Allergen] = set()
    unknown: list[str] = []
    for raw in _SEPARATORS.split(text):
        term = " ".join(raw.lower().replace("_", " ").split())
        if not term:
            continue
        keys = USER_TERMS.get(term)
        if keys is None:
            unknown.append(raw.strip())
        else:
            found |= keys
    if unknown:
        raise ValueError(f"unknown allergen term(s): {', '.join(unknown)}")
    return with_implied(found)


def parse_allergen_keys(text: str) -> frozenset[Allergen]:
    """Parse canonical keys only ("milk; tree_nuts"), for hand-checked ground truth where aliases hide mistakes."""
    keys: set[Allergen] = set()
    unknown: list[str] = []
    for raw in _SEPARATORS.split(text):
        term = raw.strip()
        if not term:
            continue
        try:
            keys.add(Allergen(term))
        except ValueError:
            unknown.append(term)
    if unknown:
        raise ValueError(f"not an allergen key: {', '.join(unknown)} (use {', '.join(a.value for a in Allergen)})")
    return frozenset(keys)
