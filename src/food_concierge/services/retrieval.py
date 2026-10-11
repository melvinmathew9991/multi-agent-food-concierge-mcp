"""Search backends behind one interface: FAISS and Qdrant (dense), BM25 (lexical).

Hard filters are decided once, in SQL (``CatalogStore.candidate_ids``), and each backend only restricts its search
to those ids: FAISS through an id selector, Qdrant through an id filter, BM25 through a mask. So no backend can
return a dish the filters exclude, and the filter rules live in one place (PRD F4, Phase 2 DoD).

Qdrant runs in memory, loaded from the same saved vectors as FAISS: its on-disk local mode persists points with
pickle, which the engineering rules forbid.

``Retriever`` composes the stages that Phase 3 compares (BM25, dense, hybrid RRF, cross-encoder rerank,
``dish_key`` de-duplication) behind one ``RetrievalConfig``, so the evaluation measures the code that serves
search, not a copy of it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar
from uuid import UUID

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, HasIdCondition, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from food_concierge.errors import ConfigError
from food_concierge.models.embeddings import Embedder, Reranker, Vectors
from food_concierge.storage.catalog_db import CatalogStore, Filters

if TYPE_CHECKING:  # storage.indexes imports this module to build the backends
    from food_concierge.storage.indexes import SearchIndexes

_TOKEN = re.compile(r"\w+")
_T = TypeVar("_T")
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


class RetrievalConfig(BaseModel):
    """One retrieval stack; the defaults are the Phase 2 behaviour until ADR-0001 chooses otherwise."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["bm25", "dense", "hybrid"] = "dense"
    backend: Literal["faiss", "qdrant"] = "faiss"
    k: int = Field(default=3, gt=0)
    fuse_depth: int = Field(default=20, gt=0)  # results taken from each list before RRF
    rrf_k: int = Field(default=60, gt=0)  # the usual constant; fixed, not tuned on the dev split (A16)
    rerank_depth: int | None = Field(default=None, gt=0)  # cross-encoder over the top N; None skips it
    dedupe: bool = False  # keep only the best-ranked item per dish_key


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)  # stage → milliseconds, plus "total"


def rrf_fuse(rankings: Sequence[Sequence[Hit]], rrf_k: int) -> list[Hit]:
    """Reciprocal rank fusion: each item scores the sum of 1 / (rrf_k + rank) over the lists it appears in."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.item_id] = scores.get(hit.item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted((Hit(item_id, score) for item_id, score in scores.items()), key=lambda h: (-h.score, h.item_id))


class Retriever:
    """Filtered search through the stages a ``RetrievalConfig`` selects, timing each stage."""

    def __init__(
        self,
        store: CatalogStore,
        indexes: SearchIndexes,
        embedder: Embedder,
        reranker: Reranker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        embedder.fingerprint.require_match(indexes.manifest.embedder)
        self._store = store
        self._indexes = indexes
        self._embedder = embedder
        self._reranker = reranker
        self._clock = clock
        self._texts = store.doc_texts()
        self._dish_keys = store.dish_keys()

    def search(self, query: str, filters: Filters, config: RetrievalConfig) -> SearchResult:
        reranker = self._reranker
        if config.rerank_depth is not None and reranker is None:
            raise ConfigError("This retrieval configuration reranks, but no reranker was given.")
        timings: dict[str, float] = {}
        start = self._clock()

        def timed(stage: str, step: Callable[[], _T]) -> _T:
            began = self._clock()
            value = step()
            timings[stage] = (self._clock() - began) * 1000.0
            return value

        ids = timed("filter", lambda: self._store.candidate_ids(filters))
        if not ids:
            timings["total"] = (self._clock() - start) * 1000.0
            return SearchResult([], timings)
        allowed = None if len(ids) == len(self._texts) else ids
        # De-duplication can drop at most this many items, so every list goes that much deeper.
        extra = len(ids) - len({self._dish_keys[i] for i in ids}) if config.dedupe else 0
        depth = max(config.k, config.rerank_depth or 0) + extra

        if config.mode == "bm25":
            hits = timed("bm25", lambda: self._indexes.bm25.search(query, depth, allowed))
        else:
            vector = timed("embed", lambda: self._embedder.embed_query(query))
            backend: VectorStore = self._indexes.faiss if config.backend == "faiss" else self._indexes.qdrant
            if config.mode == "dense":
                hits = timed("dense", lambda: backend.search(vector, depth, allowed))
            else:
                list_depth = max(depth, config.fuse_depth + extra)
                lexical = timed("bm25", lambda: self._indexes.bm25.search(query, list_depth, allowed))
                dense = timed("dense", lambda: backend.search(vector, list_depth, allowed))
                hits = timed("fuse", lambda: rrf_fuse([lexical, dense], config.rrf_k))
        if config.dedupe:
            hits = timed("dedupe", lambda: self._dedupe(hits))
        if reranker is not None and config.rerank_depth is not None:
            rerank_depth = config.rerank_depth
            hits = timed("rerank", lambda: self._rerank(reranker, query, hits, rerank_depth))
        timings["total"] = (self._clock() - start) * 1000.0
        return SearchResult(hits[: config.k], timings)

    def _dedupe(self, hits: list[Hit]) -> list[Hit]:
        seen: set[str] = set()
        kept: list[Hit] = []
        for hit in hits:
            key = self._dish_keys[hit.item_id]
            if key not in seen:
                seen.add(key)
                kept.append(hit)
        return kept

    def _rerank(self, reranker: Reranker, query: str, hits: list[Hit], depth: int) -> list[Hit]:
        """Cross-encoder scores reorder the top ``depth``; anything below keeps its place after them."""
        head, tail = hits[:depth], hits[depth:]
        scores = reranker.score(query, [self._texts[hit.item_id] for hit in head])
        reranked = sorted(
            (Hit(hit.item_id, score) for hit, score in zip(head, scores, strict=True)),
            key=lambda h: (-h.score, h.item_id),
        )
        return reranked + tail
