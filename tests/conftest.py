from collections.abc import Callable
from pathlib import Path

import pytest

from food_concierge.config import REPO_ROOT, Settings, get_settings
from food_concierge.ingestion.allergens import load_lexicon, tag_item
from food_concierge.ingestion.loader import load_catalog
from food_concierge.models.embeddings import HashingEmbedder
from food_concierge.models.usage import get_usage_ledger
from food_concierge.storage.catalog_db import CATALOG_DB, CatalogStore, build_catalog_db
from food_concierge.storage.indexes import INDEX_DIR, build_indexes


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests must not depend on the developer's shell: clear every variable Settings reads.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_usage_ledger() -> None:
    # Call caps are per process; each test starts with none of its providers' quota used.
    get_usage_ledger().reset()


BuildData = Callable[..., None]


def _build_data(data_dir: Path, query_prefix: str = "") -> None:
    """The real catalog and its indexes under ``data_dir/processed``, as ``flows build`` makes them, embedded with
    the hashing embedder so no model is downloaded."""
    loaded = load_catalog(REPO_ROOT / "data" / "raw")
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    tags = {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}
    processed = data_dir / "processed"
    build_catalog_db(loaded.catalog, tags, processed / CATALOG_DB, lexicon_version=1, source_sha256="x")
    store = CatalogStore(processed / CATALOG_DB)
    build_indexes(store, HashingEmbedder(query_prefix=query_prefix), processed / INDEX_DIR)
    store.close()


@pytest.fixture(scope="session")
def build_data() -> BuildData:
    return _build_data


@pytest.fixture(scope="session")
def built_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built catalog shared by the server tests; read-only, so one build serves them all."""
    path = tmp_path_factory.mktemp("built")
    _build_data(path)
    return path
