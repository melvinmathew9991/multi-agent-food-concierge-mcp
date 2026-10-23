"""Meal totals for the ``meal_totals`` tool: exact sums over the catalog's numbers (architecture §4).

Each dish counts as it is served, for its ``serves`` range of people, and a dish listed twice counts twice. Totals
are for the whole order; dividing by diners is the caller's choice, since a dish's serves is a range.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from food_concierge.services.catalog import known_items
from food_concierge.storage.catalog_db import CatalogStore


class MealTotals(BaseModel):
    model_config = ConfigDict(frozen=True)

    dish_ids: list[str]  # as counted, repeats included
    kcal: int
    protein_g: float
    carbs_g: float
    fat_g: float
    price_inr: int
    serves_min: int
    serves_max: int


class NutritionService:
    """Totals over dishes from one built catalog."""

    def __init__(self, store: CatalogStore) -> None:
        self._store = store

    def meal_totals(self, values: Iterable[str]) -> MealTotals:
        """The totals of these dishes; raises ``NotFoundError`` naming any unknown id."""
        ids, rows = known_items(self._store, values, distinct=False)
        dishes = [rows[item_id] for item_id in ids]
        return MealTotals(
            dish_ids=ids,
            kcal=sum(dish["kcal"] for dish in dishes),
            protein_g=round(sum(dish["protein_g"] for dish in dishes), 1),
            carbs_g=round(sum(dish["carbs_g"] for dish in dishes), 1),
            fat_g=round(sum(dish["fat_g"] for dish in dishes), 1),
            price_inr=sum(dish["price_inr"] for dish in dishes),
            serves_min=sum(dish["serves_min"] for dish in dishes),
            serves_max=sum(dish["serves_max"] for dish in dishes),
        )
