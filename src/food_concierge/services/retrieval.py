"""Search backends behind one interface: FAISS and Qdrant (dense), BM25 (lexical).

Hard filters are decided once, in SQL (``CatalogStore.candidate_ids``), and each backend only restricts its search
to those ids: FAISS through an id selector, Qdrant through an id filter, BM25 through a mask. So no backend can
return a dish the filters exclude, and the filter rules live in one place (PRD F4, Phase 2 DoD).

Qdrant runs in memory, loaded from the same saved vectors as FAISS: its on-disk local mode persists points with
pickle, which the engineering rules forbid.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, HasIdCondition, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from food_concierge.models.embeddings import Vectors
from food_concierge.storage.catalog_db import CatalogStore, Filters

_TOKEN = re.compile(r"\w+")
_COLLECTION = "dishes"


@dataclass(frozen=True)
class Hit:
    item_id: str
    score: float


class VectorStore(Protocol):
    name: str

    def search(self, query: Vectors, k: int, allowed: Collection[str] | None = None) -> list[Hit]:
        """Up to ``k`` nearest items by inner product, only among ``allowed`` when it is given."""
        ...


class FaissStore:
    """Exact inner-product search (``IndexIDMap2(IndexFlatIP)``); ids are positions in ``item_ids``."""

    name = "faiss"

    def __init__(self, index: faiss.Index, item_ids: Sequence[str]) -> None:
        if index.ntotal != len(item_ids):
            raise ValueError(f"the FAISS index holds {index.ntotal} vectors for {len(item_ids)} ids")
        self.index = index
        self._ids = list(item_ids)
        self._positions = {item_id: position for position, item_id in enumerate(self._ids)}

    @classmethod
    def build(cls, vectors: Vectors, item_ids: Sequence[str]) -> FaissStore:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.arange(len(item_ids), dtype=np.int64))
        return cls(index, item_ids)

    def search(self, query: Vectors, k: int, allowed: Collection[str] | None = None) -> list[Hit]:
        params = None
        limit = len(self._ids)
        if allowed is not None:
            positions = np.array([self._positions[i] for i in allowed if i in self._positions], dtype=np.int64)
            if not len(positions):
                return []
            selector = faiss.IDSelectorBatch(positions)  # kept referenced until the search returns
            params = faiss.SearchParameters()
            params.sel = selector
            limit = len(positions)
        top = min(k, limit)
        if top <= 0:
            return []
        scores, ids = self.index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), top, params=params)
        return [Hit(self._ids[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]


class QdrantStore:
    """Qdrant in memory, with dot-product distance (vectors are normalised, so this is cosine)."""

    name = "qdrant"

    def __init__(self, vectors: Vectors, item_ids: Sequence[str]) -> None:
        self._ids = list(item_ids)
        self._positions = {item_id: position for position, item_id in enumerate(self._ids)}
        self._client = QdrantClient(location=":memory:")
        self._client.create_collection(
            _COLLECTION, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.DOT)
        )
        self._client.upsert(
            _COLLECTION,
            points=[
                PointStruct(id=position, vector=vector.tolist(), payload={"item_id": item_id})
                for position, (item_id, vector) in enumerate(zip(self._ids, vectors, strict=True))
            ],
        )

    def search(self, query: Vectors, k: int, allowed: Collection[str] | None = None) -> list[Hit]:
        condition = None
        if allowed is not None:
            point_ids: list[int | str | UUID] = [self._positions[i] for i in allowed if i in self._positions]
            if not point_ids:
                return []
            condition = Filter(must=[HasIdCondition(has_id=point_ids)])
        if k <= 0:
            return []
        points = self._client.query_points(
            _COLLECTION, query=np.asarray(query, dtype=np.float32).tolist(), limit=k, query_filter=condition
        ).points
        return [Hit(self._ids[int(point.id)], float(point.score)) for point in points]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Index:
    """Lexical search (Okapi BM25) over the same document text; only items sharing a term with the query score."""

    name = "bm25"

    def __init__(self, item_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(item_ids) != len(texts):
            raise ValueError("one text per item id")
        self._ids = list(item_ids)
        self._bm25 = BM25Okapi([tokenize(text) for text in texts])

    def search(self, query: str, k: int, allowed: Collection[str] | None = None) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        permitted = None if allowed is None else set(allowed)
        ranked = sorted(
            (
                Hit(item_id, float(score))
                for item_id, score in zip(self._ids, scores, strict=True)
                if score > 0 and (permitted is None or item_id in permitted)
            ),
            key=lambda hit: (-hit.score, hit.item_id),
        )
        return ranked[: max(k, 0)]


def filtered_search(store: CatalogStore, backend: VectorStore, query: Vectors, filters: Filters, k: int) -> list[Hit]:
    """Dense search restricted to the items that satisfy every hard filter."""
    return backend.search(query, k, allowed=store.candidate_ids(filters))
