"""Meal totals for the ``meal_totals`` tool: exact sums over the catalog's numbers (architecture §4).

The menu gives calories and macros per serving and the price per dish, which serves a number or a range of people
(``serves``). A dish's whole energy is its per-serving value times its servings, so the order's totals are a range
when a dish serves a range; each diner's share is the order divided by the diners. A dish listed twice counts twice.
The nutrition values are the menu's estimates, not measurements (``data/raw/README.md``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from food_concierge.errors import InputValidationError
from food_concierge.services.catalog import known_items
from food_concierge.storage.catalog_db import CatalogStore

MAX_DINERS = 20


class Nutrients(BaseModel):
    model_config = ConfigDict(frozen=True)

    kcal: int
    protein_g: float
    carbs_g: float
    fat_g: float


class MealTotals(BaseModel):
    """An order shared by its diners: the whole order and each diner's equal share, as min-max ranges."""

    model_config = ConfigDict(frozen=True)

    dish_ids: list[str]  # as counted, repeats included
    diners: int
    serves_min: int  # servings the order makes
    serves_max: int
    order_min: Nutrients  # every dish at its fewest servings
    order_max: Nutrients  # every dish at its most servings
    per_person_min: Nutrients
    per_person_max: Nutrients
    price_inr: int  # the whole order
    price_inr_per_person: int
    notes: list[str]


def _nutrients(dishes: Sequence[Mapping[str, Any]], serves: str, diners: int) -> Nutrients:
    def total(field: str) -> float:
        return float(sum(dish[field] * dish[serves] for dish in dishes)) / diners

    return Nutrients(
        kcal=round(total("kcal")),
        protein_g=round(total("protein_g"), 1),
        carbs_g=round(total("carbs_g"), 1),
        fat_g=round(total("fat_g"), 1),
    )


class NutritionService:
    """Totals over dishes from one built catalog."""

    def __init__(self, store: CatalogStore) -> None:
        self._store = store

    def meal_totals(self, values: Iterable[str], diners: int = 1) -> MealTotals:
        """The totals of these dishes shared by ``diners``; raises ``NotFoundError`` naming any unknown id."""
        if not 1 <= diners <= MAX_DINERS:
            raise InputValidationError(f"Give between 1 and {MAX_DINERS} diners.")
        ids, rows = known_items(self._store, values, distinct=False)
        dishes = [rows[item_id] for item_id in ids]
        serves_min = sum(dish["serves_min"] for dish in dishes)
        serves_max = sum(dish["serves_max"] for dish in dishes)
        price = sum(dish["price_inr"] for dish in dishes)
        people = f"{diners} diner" + ("s" if diners > 1 else "")
        notes = []
        if serves_max < diners:
            notes.append(f"The order makes at most {serves_max} servings for {people}.")
        elif serves_min > diners:
            notes.append(f"The order makes at least {serves_min} servings for {people}; each share is more than one.")
        return MealTotals(
            dish_ids=ids,
            diners=diners,
            serves_min=serves_min,
            serves_max=serves_max,
            order_min=_nutrients(dishes, "serves_min", 1),
            order_max=_nutrients(dishes, "serves_max", 1),
            per_person_min=_nutrients(dishes, "serves_min", diners),
            per_person_max=_nutrients(dishes, "serves_max", diners),
            price_inr=price,
            price_inr_per_person=round(price / diners),
            notes=notes,
        )
