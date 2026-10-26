"""Dish search for the MCP tools and the HTTP API: one path, so both enforce the constraints the same way.

Search filters in SQL before ranking (``candidate_ids``), then re-checks every result against the same constraints
in Python (``dish_violations``) and drops any that breaks one. The two agree by test; if they ever didn't, the
dish is dropped and the disagreement logged, so a dish that violates the constraints is never returned.

Constraints the query states in words ("peanut-free", "vegan") are added to the ones given, since a client may
write them only in the query; the note says which were read, so the client can confirm them with the diner.

Each dish carries its score, and ``weak_match`` says when even the best one is a weak match (``services/relevance``),
so a client can say the menu has nothing like the request instead of recommending the nearest dish regardless.
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
from food_concierge.services.relevance import WEAK_MATCH, MenuWords, is_weak
from food_concierge.services.retrieval import RetrievalConfig, SearchResult
from food_concierge.storage.catalog_db import Filters

logger = logging.getLogger(__name__)

MAX_RESULTS = 10
NO_MATCH = "No dish meets all of these constraints; relaxing one may help."


class Searcher(Protocol):
    def search(self, query: str, filters: Filters, config: RetrievalConfig) -> SearchResult: ...


class Match(DishSummary):
    """A dish found by search."""

    score: float  # similarity to the query, higher is closer; compare within one result list only


class Dishes(BaseModel):
    model_config = ConfigDict(frozen=True)

    dishes: list[Match]
    weak_match: bool = False  # even the best dish is a weak match: the menu may have nothing like the query
    note: str | None = None  # constraints read from the query, a weak match, and why the list is empty when it can


class SearchService:
    """Ranked, constraint-checked dish search over one retrieval stack."""

    def __init__(
        self,
        catalog: CatalogService,
        searcher: Searcher,
        retrieval: RetrievalConfig,
        settings: Settings,
        words: MenuWords,
        weak_score: float | None = None,  # the embedding model's cut-off (``WEAK_SCORES``); None never says weak
    ) -> None:
        self._catalog = catalog
        self._searcher = searcher
        self._retrieval = retrieval
        self._words = words
        self._weak_score = weak_score if retrieval.mode == "dense" else None
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
        found = self._searcher.search(query, constraints.filters, config).hits
        scores = {hit.item_id: hit.score for hit in found if hit.item_id not in skip}
        ids = list(scores)[:k]
        if not ids:
            return Dishes(dishes=[], note=" ".join([*notes, NO_MATCH]))
        # Cosine similarity has a fixed scale; BM25 and fused scores don't, so only dense search can say "weak".
        cutoff = self._weak_score
        weak = cutoff is not None and is_weak(query, scores[ids[0]], cutoff, self._words)
        if weak:
            notes.append(WEAK_MATCH)
        details = self._catalog.get_dishes(ids)
        allowed = [dish for dish in details if not dish_violations(dish, constraints)]
        if len(allowed) != len(details):
            dropped = sorted({dish.item_id for dish in details} - {dish.item_id for dish in allowed})
            logger.error("search returned dishes that break the constraints", extra={"item_ids": dropped})
        dishes = [Match(**summary_of(dish).model_dump(), score=round(scores[dish.item_id], 4)) for dish in allowed]
        return Dishes(dishes=dishes, weak_match=weak, note=" ".join(notes) or None)
