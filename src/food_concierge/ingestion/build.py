"""The two offline build steps, in plain Python: catalog files → SQLite catalog → search indexes.

Each step is skipped when its output already matches its inputs, so re-running the build with unchanged inputs
does nothing (Phase 2 DoD):

- the catalog is current when its recorded ``data_sha256`` (a hash of the raw files and the allergen lexicon) and
  document-text version match;
- the indexes are current when they load: same embedder, same catalog data, files unchanged.

The Prefect flows in ``food_concierge.flows`` orchestrate these functions; tests can call them directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from food_concierge.config import Settings
from food_concierge.errors import NotReadyError
from food_concierge.ingestion.allergens import LEXICON_PATH, load_lexicon, tag_item
from food_concierge.ingestion.loader import ATTRIBUTIONS_FILE, MENU_FILE, RESTAURANTS_FILE, load_catalog
from food_concierge.models.embeddings import Embedder
from food_concierge.storage.catalog_db import (
    CATALOG_DB,
    DOC_TEXT_VERSION,
    CatalogStore,
    build_catalog_db,
    data_sha256,
)
from food_concierge.storage.indexes import INDEX_DIR, build_indexes, load_indexes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    step: str
    rebuilt: bool
    output: Path
    data_sha256: str


def source_files(settings: Settings) -> list[Path]:
    """The inputs the catalog is built from; attributions are optional."""
    raw = settings.raw_dir
    files = [raw / RESTAURANTS_FILE, raw / MENU_FILE, settings.data_dir / LEXICON_PATH]
    return files + ([raw / ATTRIBUTIONS_FILE] if (raw / ATTRIBUTIONS_FILE).exists() else [])


def _catalog_is_current(path: Path, source_sha256: str) -> bool:
    if not path.exists():
        return False
    store = CatalogStore(path)
    try:
        meta = store.meta()
    finally:
        store.close()
    return meta.get("data_sha256") == source_sha256 and meta.get("doc_text_version") == str(DOC_TEXT_VERSION)


def build_catalog(settings: Settings, *, force: bool = False) -> StepResult:
    """Validate the catalog files, tag allergens and write the SQLite catalog, unless it is already current."""
    output = settings.processed_dir / CATALOG_DB
    source_sha256 = data_sha256(source_files(settings))
    if not force and _catalog_is_current(output, source_sha256):
        logger.info("catalog up to date", extra={"data_sha256": source_sha256[:12]})
        return StepResult("catalog", False, output, source_sha256)

    loaded = load_catalog(settings.raw_dir)
    for warning in loaded.warnings:
        logger.warning("catalog warning", extra={"issue": str(warning)})
    lexicon = load_lexicon(settings.data_dir / LEXICON_PATH)
    tags = {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}
    build_catalog_db(loaded.catalog, tags, output, lexicon_version=lexicon.version, source_sha256=source_sha256)
    logger.info("catalog built", extra={"items": len(loaded.catalog.items), "data_sha256": source_sha256[:12]})
    return StepResult("catalog", True, output, source_sha256)


def build_search_index(settings: Settings, embedder: Embedder, *, force: bool = False) -> StepResult:
    """Embed the catalog and write the search indexes, unless the existing ones still load for this catalog."""
    output = settings.processed_dir / INDEX_DIR
    store = CatalogStore(settings.processed_dir / CATALOG_DB)
    try:
        source_sha256 = store.meta()["data_sha256"]
        if not force:
            try:
                load_indexes(store, embedder, output)
            except NotReadyError as reason:
                logger.info("index rebuild needed", extra={"reason": reason.message})
            else:
                logger.info("index up to date", extra={"data_sha256": source_sha256[:12]})
                return StepResult("index", False, output, source_sha256)
        manifest = build_indexes(store, embedder, output)
    finally:
        store.close()
    logger.info("index built", extra={"items": manifest.n_items, "dim": manifest.dim})
    return StepResult("index", True, output, source_sha256)
