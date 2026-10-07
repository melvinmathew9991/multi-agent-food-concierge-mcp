"""Build and load the search indexes, bound by a manifest to the embedder and the catalog they came from.

Files in the index directory (no pickle anywhere):

- ``vectors.npy``: one normalised row per item, saved by numpy with pickling disabled;
- ``item_ids.json``: the item id of each row;
- ``faiss.index``: FAISS's native format;
- ``manifest.json``: embedder fingerprint, dimension, item count, catalog data hash, document-text version, build
  time, and the SHA-256 of each file above.

Loading refuses (``NotReadyError``) an index built with another embedder, from other catalog data or document
text, or whose files changed after the build. The BM25 index and the in-memory Qdrant collection are rebuilt at
load time from the catalog text and the saved vectors.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from food_concierge.errors import NotReadyError
from food_concierge.models.embeddings import Embedder, EmbedderFingerprint
from food_concierge.services.retrieval import Bm25Index, FaissStore, QdrantStore
from food_concierge.storage.catalog_db import CatalogStore

INDEX_DIR = "indexes"  # under Settings.processed_dir
MANIFEST = "manifest.json"
VECTORS = "vectors.npy"
ITEM_IDS = "item_ids.json"
FAISS_INDEX = "faiss.index"


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    embedder: EmbedderFingerprint
    dim: int
    n_items: int
    data_sha256: str
    doc_text_version: int
    created_at: datetime
    files: dict[str, str]  # file name → SHA-256


@dataclass(frozen=True)
class SearchIndexes:
    manifest: IndexManifest
    faiss: FaissStore
    qdrant: QdrantStore
    bm25: Bm25Index


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_indexes(
    store: CatalogStore,
    embedder: Embedder,
    index_dir: Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> IndexManifest:
    """Embed every item's document text and write the indexes, replacing any previous build as a whole."""
    texts = store.doc_texts()
    item_ids = list(texts)
    vectors = np.ascontiguousarray(embedder.embed_documents(list(texts.values())), dtype=np.float32)
    meta = store.meta()

    partial = index_dir.with_name(index_dir.name + ".part")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    np.save(partial / VECTORS, vectors, allow_pickle=False)
    (partial / ITEM_IDS).write_text(json.dumps(item_ids), encoding="utf-8")
    faiss.write_index(FaissStore.build(vectors, item_ids).index, str(partial / FAISS_INDEX))
    manifest = IndexManifest(
        embedder=embedder.fingerprint,
        dim=int(vectors.shape[1]),
        n_items=len(item_ids),
        data_sha256=meta["data_sha256"],
        doc_text_version=int(meta["doc_text_version"]),
        created_at=now(),
        files={name: _sha256(partial / name) for name in (VECTORS, ITEM_IDS, FAISS_INDEX)},
    )
    (partial / MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    previous = index_dir.with_name(index_dir.name + ".old")
    shutil.rmtree(previous, ignore_errors=True)
    if index_dir.exists():
        index_dir.rename(previous)
    partial.rename(index_dir)
    shutil.rmtree(previous, ignore_errors=True)
    return manifest


def load_indexes(store: CatalogStore, embedder: Embedder, index_dir: Path) -> SearchIndexes:
    """Load the indexes after checking they match this embedder, this catalog and their own manifest."""
    try:
        manifest = IndexManifest.model_validate_json((index_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise NotReadyError("The search index is missing or unreadable; build it.") from exc
    embedder.fingerprint.require_match(manifest.embedder)
    meta = store.meta()
    if manifest.data_sha256 != meta["data_sha256"] or str(manifest.doc_text_version) != meta["doc_text_version"]:
        raise NotReadyError("The search index was built from different catalog data; rebuild it.")
    for name, expected in manifest.files.items():
        if not (index_dir / name).exists() or _sha256(index_dir / name) != expected:
            raise NotReadyError(f"The search index file {name} changed after it was built; rebuild it.")

    item_ids: list[str] = json.loads((index_dir / ITEM_IDS).read_text(encoding="utf-8"))
    vectors = np.load(index_dir / VECTORS, allow_pickle=False)
    texts = store.doc_texts()
    if (
        set(item_ids) != set(texts)
        or len(item_ids) != manifest.n_items
        or vectors.shape
        != (
            manifest.n_items,
            manifest.dim,
        )
    ):
        raise NotReadyError("The search index does not cover the catalog's items; rebuild it.")
    return SearchIndexes(
        manifest=manifest,
        faiss=FaissStore(faiss.read_index(str(index_dir / FAISS_INDEX)), item_ids),
        qdrant=QdrantStore(vectors, item_ids),
        bm25=Bm25Index(item_ids, [texts[item_id] for item_id in item_ids]),
    )
