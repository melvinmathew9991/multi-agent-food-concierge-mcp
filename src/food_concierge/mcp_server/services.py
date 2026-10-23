"""The services one MCP server runs on, built from settings or handed in by tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from food_concierge.config import Settings
from food_concierge.errors import ConfigError
from food_concierge.models.embeddings import get_embedder
from food_concierge.services.allergens import AllergenService
from food_concierge.services.catalog import CatalogService
from food_concierge.services.constraints import ConstraintService
from food_concierge.services.nutrition import NutritionService
from food_concierge.services.retrieval import RetrievalConfig, Retriever, SearchResult
from food_concierge.services.vision import VisionService, build_vision_service
from food_concierge.storage.catalog_db import CATALOG_DB, CatalogStore, Filters
from food_concierge.storage.indexes import INDEX_DIR, load_indexes

logger = logging.getLogger(__name__)


class Searcher(Protocol):
    def search(self, query: str, filters: Filters, config: RetrievalConfig) -> SearchResult: ...


@dataclass(frozen=True)
class Services:
    store: CatalogStore
    catalog: CatalogService
    allergens: AllergenService
    nutrition: NutritionService
    constraints: ConstraintService
    searcher: Searcher
    retrieval: RetrievalConfig  # the production stack; its k is replaced per request
    vision: VisionService | None  # None when no provider has a vision model: photos are then refused
    settings: Settings

    @classmethod
    def over(
        cls,
        store: CatalogStore,
        searcher: Searcher,
        settings: Settings,
        *,
        retrieval: RetrievalConfig | None = None,
        vision: VisionService | None = None,
    ) -> Services:
        return cls(
            store=store,
            catalog=CatalogService(store),
            allergens=AllergenService(store),
            nutrition=NutritionService(store),
            constraints=ConstraintService(store),
            searcher=searcher,
            retrieval=retrieval or RetrievalConfig(),
            vision=vision,
            settings=settings,
        )


def build_services(settings: Settings) -> Services:
    """The built catalog and indexes under ``processed_dir``, the configured embedder, and vision if configured.

    Raises ``NotReadyError`` when the catalog or indexes haven't been built (``python -m food_concierge.flows build``).
    """
    store = CatalogStore(settings.processed_dir / CATALOG_DB)
    embedder = get_embedder(settings)
    retriever = Retriever(store, load_indexes(store, embedder, settings.processed_dir / INDEX_DIR), embedder)
    try:
        vision = build_vision_service(settings)
    except ConfigError:
        logger.warning("no vision model configured; describe_image will ask for words instead")
        vision = None
    # RetrievalConfig() is the Phase 2 stack until ADR-0001 (#47) chooses the production one.
    return Services.over(store, retriever, settings, vision=vision)
