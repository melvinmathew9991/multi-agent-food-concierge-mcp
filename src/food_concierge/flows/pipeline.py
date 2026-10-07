"""The ``ingest`` and ``build_index`` flows, and ``build``, which runs both.

Tasks retry only on transient failures (a dropped connection while the embedding model downloads); a validation
error in the catalog files fails at once. Task results are not persisted, so Prefect never pickles anything; the
"re-run does nothing" behaviour comes from the input-hash checks in ``ingestion.build``. Flow parameters are plain
values (Prefect records them), so the embedder comes from settings.
"""

from __future__ import annotations

from typing import Any

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from food_concierge.config import get_settings
from food_concierge.ingestion.build import StepResult, build_catalog, build_search_index
from food_concierge.models.embeddings import get_embedder

TRANSIENT = (ConnectionError, TimeoutError)


def _transient(_task: Any, _run: Any, state: Any) -> bool:
    try:
        state.result()
    except TRANSIENT:
        return True
    except Exception:  # anything else (an invalid catalog, a bad index) is not worth retrying
        return False
    return False


@task(cache_policy=NO_CACHE, persist_result=False)
def catalog_task(force: bool) -> StepResult:
    return build_catalog(get_settings(), force=force)


@task(cache_policy=NO_CACHE, persist_result=False, retries=2, retry_delay_seconds=10, retry_condition_fn=_transient)
def index_task(force: bool) -> StepResult:
    settings = get_settings()
    return build_search_index(settings, get_embedder(settings), force=force)


@flow(name="ingest", persist_result=False)
def ingest_flow(force: bool = False) -> StepResult:
    """Validate the catalog files, tag allergens and build the SQLite catalog."""
    return catalog_task(force)


@flow(name="build_index", persist_result=False)
def build_index_flow(force: bool = False) -> StepResult:
    """Embed the catalog and build the FAISS, Qdrant and BM25 indexes."""
    return index_task(force)


@flow(name="build", persist_result=False)
def build_flow(force: bool = False) -> list[StepResult]:
    """The whole offline build: ``ingest`` then ``build_index``."""
    return [ingest_flow(force), build_index_flow(force)]
