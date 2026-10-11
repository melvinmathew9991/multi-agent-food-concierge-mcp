"""Ranking metrics per query, and the ground truth that Violation@k is judged against.

Violations are checked against the owner-reviewed ``true_allergens`` and the catalog's own diet, calories,
price and cuisine, never against the system's allergen tags: scoring the system with its own tags would be
circular (audit A3).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from food_concierge.ingestion.allergens import AllergenTag, DishAllergens, Level, Source
from food_concierge.ingestion.loader import CatalogLoad
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.ingestion.taxonomy import Allergen
from food_concierge.storage.catalog_db import Filters


def _gain(grade: int) -> float:
    return float(2**grade - 1)


def _discounted(grades: Sequence[int]) -> float:
    return sum(_gain(grade) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 3) -> float:
    """nDCG@k with gain 2^grade - 1; the ideal ordering comes from every judged item; unjudged items count 0."""
    ideal = _discounted(sorted(grades.values(), reverse=True)[:k])
    if ideal == 0:
        raise ValueError("nDCG is undefined for a query with no relevant item")
    return _discounted([grades.get(item, 0) for item in ranked[:k]]) / ideal


def success_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 3, threshold: int = 2) -> float:
    return float(any(grades.get(item, 0) >= threshold for item in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without a relevant item")
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked, start=1) if item in relevant), 0.0)


def judged_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 3) -> float:
    """Share of the top k that has a grade; below 95% the configuration needs its new pairs graded first."""
    top = ranked[:k]
    return sum(item in grades for item in top) / len(top) if top else 1.0


@dataclass(frozen=True)
class GroundTruth:
    items: Mapping[str, MenuItem]
    allergens: Mapping[str, frozenset[Allergen]]

    @classmethod
    def from_load(cls, load: CatalogLoad) -> GroundTruth:
        return cls({item.item_id: item for item in load.catalog.items}, load.true_allergens)

    def admits(self, filters: Filters, item_id: str) -> bool:
        # The reviewed allergens stand in for the tags, so the filter rules are the ones search uses.
        truth = DishAllergens(
            item_id,
            tuple(AllergenTag(a, Source.LABEL, Level.CONTAINS, "ground truth") for a in self.allergens[item_id]),
            (),
        )
        return filters.admits(self.items[item_id], truth)


def violation_at_k(ranked: Sequence[str], filters: Filters, truth: GroundTruth, k: int = 3) -> float:
    """Share of the returned top k that breaks the query's filters by the ground truth."""
    top = ranked[:k]
    return sum(not truth.admits(filters, item) for item in top) / len(top) if top else 0.0
