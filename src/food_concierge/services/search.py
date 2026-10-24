"""Dish search for the MCP tools and the HTTP API: one path, so both enforce the constraints the same way.

Search filters in SQL before ranking (``candidate_ids``), then re-checks every result against the same constraints
in Python (``dish_violations``) and drops any that breaks one. The two agree by test; if they ever didn't, the
dish is dropped and the disagreement logged, so a dish that violates the constraints is never returned.

Constraints the query states in words ("peanut-free", "vegan") are added to the ones given, since a client may
write them only in the query; the note says which were read, so the client can confirm them with the diner.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from food_concierge.config import Settings
from food_concierge.errors import InputValidationError
from food_concierge.services.catalog import CatalogService, DishSummary, summary_of
from food_concierge.services.constraints import Constraints, constraints_in_text, describe, dish_violations, merge
from food_concierge.services.retrieval import RetrievalConfig, SearchResult
from food_concierge.storage.catalog_db import Filters

logger = logging.getLogger(__name__)

MAX_RESULTS = 10
NO_MATCH = "No dish meets all of these constraints; relaxing one may help."


class Searcher(Protocol):
    def search(self, query: str, filters: Filters, config: RetrievalConfig) -> SearchResult: ...


class Dishes(BaseModel):
    model_config = ConfigDict(frozen=True)

    dishes: list[DishSummary]
    note: str | None = None  # constraints read from the query, and why the list is empty when that can be said


class SearchService:
    """Ranked, constraint-checked dish search over one retrieval stack."""

    def __init__(self, catalog: CatalogService, searcher: Searcher, retrieval: RetrievalConfig, settings: Settings):
        self._catalog = catalog
        self._searcher = searcher
        self._retrieval = retrieval
        self._max_query_chars = settings.max_query_chars

    def search(
        self, query: str, constraints: Constraints, k: int, skip: Collection[str] = (), *, read_query: bool = True
    ) -> Dishes:
        """Up to ``k`` dishes for ``query`` that meet ``constraints``, leaving out the ids in ``skip``.

        With ``read_query``, constraints the query states in words are added; turn it off when the query isn't a
        diner's words (``find_similar`` searches with a dish's own description).
        """
        if not query.strip():
            raise InputValidationError("The query is empty.")
        if len(query) > self._max_query_chars:
            raise InputValidationError(f"The query is longer than {self._max_query_chars} characters.")
        if not 1 <= k <= MAX_RESULTS:
            raise InputValidationError(f"Ask for 1 to {MAX_RESULTS} dishes.")
        notes: list[str] = []
        if read_query:
            stated = constraints_in_text(query, constraints.include_unverified)
            tightened = merge(constraints, stated)
            added = Constraints(
                diet=tightened.diet if tightened.diet != constraints.diet else None,
                exclude_allergens=tightened.excluded - constraints.excluded,
            )
            if describe(added):
                notes.append(f"Also applied from the query: {describe(added)}.")
            constraints = tightened
        config = self._retrieval.model_copy(update={"k": k + len(skip)})
        hits = [hit.item_id for hit in self._searcher.search(query, constraints.filters, config).hits]
        ids = [item_id for item_id in hits if item_id not in skip][:k]
        if not ids:
            return Dishes(dishes=[], note=" ".join([*notes, NO_MATCH]))
        details = self._catalog.get_dishes(ids)
        allowed = [dish for dish in details if not dish_violations(dish, constraints)]
        if len(allowed) != len(details):
            dropped = sorted({dish.item_id for dish in details} - {dish.item_id for dish in allowed})
            logger.error("search returned dishes that break the constraints", extra={"item_ids": dropped})
        return Dishes(dishes=[summary_of(dish) for dish in allowed], note=" ".join(notes) or None)
