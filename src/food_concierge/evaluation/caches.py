"""Committed evaluation caches, so CI runs the real retrieval pipeline without downloading a model (rules §5.4).

Files in the cache directory (no pickle anywhere):

- ``documents.json`` and ``doc_vectors.npy``: every item's document text and its vector;
- ``queries.json``, one ``query_vectors_<variant>.npy`` per query-prefix variant, and ``rerank_scores.npy``
  (every query against every document, so any rerank depth is served without a miss);
- ``manifest.json``: the embedder fingerprint per variant, the reranker fingerprint, the catalog data hash,
  the document-text version, the hash of the dataset's query texts (``texts_sha256``), and the SHA-256 of each
  file above. Grades are not part of it: they change scores, not vectors.

A text the caches don't hold is an error, never a silent zero, and caches built from other catalog data, document
text or queries are refused with the stale field named. The nightly ``drift`` check re-runs freshly downloaded
models over the cached texts, so an upstream model change is noticed rather than frozen into the gate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from food_concierge.errors import ConfigError, EvaluationError
from food_concierge.models.embeddings import Embedder, EmbedderFingerprint, Reranker, RerankerFingerprint, Vectors

MANIFEST = "manifest.json"
DOCUMENTS = "documents.json"
DOC_VECTORS = "doc_vectors.npy"
QUERIES = "queries.json"
RERANK_SCORES = "rerank_scores.npy"
MIN_COSINE = 0.999  # a fresh vector this close to its cached one is the same model
MAX_RERANK_DIFF = 1e-3


def _query_vectors(variant: str) -> str:
    return f"query_vectors_{variant}.npy"


class CacheManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    embedders: dict[str, EmbedderFingerprint]  # variant ("prefix", "no_prefix") → fingerprint
    reranker: RerankerFingerprint
    data_sha256: str
    doc_text_version: int
    dataset_sha256: str
    created_at: datetime
    files: dict[str, str]  # file name → SHA-256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def texts_sha256(texts: Iterable[str]) -> str:
    """One hash over a set of texts, in sorted order, so the order they were listed in doesn't matter."""
    payload = json.dumps(sorted(set(texts)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _miss(kind: str, text: str) -> EvaluationError:
    shown = text if len(text) <= 60 else text[:57] + "..."
    return EvaluationError(f"The evaluation cache has no {kind} for '{shown}'; rebuild the caches.")


class CachedEmbedder:
    """``Embedder`` served from cached vectors, carrying the fingerprint of the model that made them."""

    def __init__(
        self,
        fingerprint: EmbedderFingerprint,
        documents: Sequence[str],
        doc_vectors: Vectors,
        queries: Sequence[str],
        query_vectors: Vectors,
    ) -> None:
        self._fingerprint = fingerprint
        self._documents = {text: row for row, text in enumerate(documents)}
        self._doc_vectors = doc_vectors
        self._queries = {text: row for row, text in enumerate(queries)}
        self._query_vectors = query_vectors

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        return self._fingerprint

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        rows = []
        for text in texts:
            if text not in self._documents:
                raise _miss("document vector", text)
            rows.append(self._documents[text])
        selected: Vectors = self._doc_vectors[rows] if rows else self._doc_vectors[:0]
        return selected

    def embed_query(self, text: str) -> Vectors:
        if text not in self._queries:
            raise _miss("query vector", text)
        vector: Vectors = self._query_vectors[self._queries[text]]
        return vector


class CachedReranker:
    """``Reranker`` served from a cached query-by-document score matrix."""

    def __init__(
        self, fingerprint: RerankerFingerprint, queries: Sequence[str], documents: Sequence[str], scores: Vectors
    ) -> None:
        self._fingerprint = fingerprint
        self._queries = {text: row for row, text in enumerate(queries)}
        self._documents = {text: column for column, text in enumerate(documents)}
        self._scores = scores

    @property
    def fingerprint(self) -> RerankerFingerprint:
        return self._fingerprint

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if query not in self._queries:
            raise _miss("rerank scores", query)
        row = self._scores[self._queries[query]]
        missing = next((text for text in documents if text not in self._documents), None)
        if missing is not None:
            raise _miss("rerank score", missing)
        return [float(row[self._documents[text]]) for text in documents]


@dataclass(frozen=True)
class EvalCaches:
    manifest: CacheManifest
    embedders: dict[str, CachedEmbedder]
    reranker: CachedReranker
    documents: tuple[str, ...]
    queries: tuple[str, ...]

    def require_current(self, data_sha256: str, doc_text_version: int, dataset_sha256: str) -> None:
        """Refuse caches built from other inputs, naming every input that changed."""
        current = {"data_sha256": data_sha256, "doc_text_version": doc_text_version, "dataset_sha256": dataset_sha256}
        stale = [name for name, value in current.items() if getattr(self.manifest, name) != value]
        if stale:
            raise EvaluationError(f"The evaluation caches are stale ({', '.join(stale)} changed); rebuild them.")


def build_caches(
    cache_dir: Path,
    embedders: Mapping[str, Embedder],
    reranker: Reranker,
    documents: Sequence[str],
    queries: Sequence[str],
    data_sha256: str,
    doc_text_version: int,
    dataset_sha256: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CacheManifest:
    """Embed and score everything once with the real models, replacing any previous caches as a whole."""
    if not embedders or not queries:
        raise ConfigError("Building evaluation caches needs at least one embedder and one query.")
    models = {e.fingerprint.model_copy(update={"query_prefix": ""}).model_dump_json() for e in embedders.values()}
    if len(models) != 1:
        raise ConfigError("Every cached query variant must come from the same embedding model.")
    documents = sorted(set(documents))
    queries = sorted(set(queries))

    partial = cache_dir.with_name(cache_dir.name + ".part")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    first = next(iter(embedders.values()))
    np.save(partial / DOC_VECTORS, first.embed_documents(documents).astype(np.float32), allow_pickle=False)
    for variant, embedder in embedders.items():
        vectors = np.stack([embedder.embed_query(q) for q in queries])
        np.save(partial / _query_vectors(variant), vectors.astype(np.float32), allow_pickle=False)
    scores = np.array([reranker.score(q, documents) for q in queries], dtype=np.float32)
    np.save(partial / RERANK_SCORES, scores, allow_pickle=False)
    # LF on every OS: the manifest hashes the bytes, and git checks text files out with LF (.gitattributes).
    (partial / DOCUMENTS).write_text(
        json.dumps(documents, ensure_ascii=False, indent=0), encoding="utf-8", newline="\n"
    )
    (partial / QUERIES).write_text(json.dumps(queries, ensure_ascii=False, indent=0), encoding="utf-8", newline="\n")
    manifest = CacheManifest(
        embedders={variant: embedder.fingerprint for variant, embedder in embedders.items()},
        reranker=reranker.fingerprint,
        data_sha256=data_sha256,
        doc_text_version=doc_text_version,
        dataset_sha256=dataset_sha256,
        created_at=now(),
        files={path.name: _sha256(path) for path in sorted(partial.iterdir())},
    )
    (partial / MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    shutil.rmtree(cache_dir, ignore_errors=True)
    partial.rename(cache_dir)
    return manifest


def load_caches(cache_dir: Path) -> EvalCaches:
    try:
        manifest = CacheManifest.model_validate_json((cache_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise EvaluationError("The evaluation caches are missing or unreadable; build them.") from exc
    for name, expected in manifest.files.items():
        if not (cache_dir / name).exists() or _sha256(cache_dir / name) != expected:
            raise EvaluationError(f"The evaluation cache file {name} changed after it was built; rebuild the caches.")
    documents: list[str] = json.loads((cache_dir / DOCUMENTS).read_text(encoding="utf-8"))
    queries: list[str] = json.loads((cache_dir / QUERIES).read_text(encoding="utf-8"))
    doc_vectors = np.load(cache_dir / DOC_VECTORS, allow_pickle=False)
    embedders = {
        variant: CachedEmbedder(
            fingerprint,
            documents,
            doc_vectors,
            queries,
            np.load(cache_dir / _query_vectors(variant), allow_pickle=False),
        )
        for variant, fingerprint in manifest.embedders.items()
    }
    scores = np.load(cache_dir / RERANK_SCORES, allow_pickle=False)
    reranker = CachedReranker(manifest.reranker, queries, documents, scores)
    return EvalCaches(manifest, embedders, reranker, tuple(documents), tuple(queries))


def _min_cosine(fresh: Vectors, cached: Vectors) -> float:
    fresh = fresh / np.linalg.norm(fresh, axis=1, keepdims=True)
    cached = cached / np.linalg.norm(cached, axis=1, keepdims=True)
    return float(np.min(np.sum(fresh * cached, axis=1)))


def drift(caches: EvalCaches, embedders: Mapping[str, Embedder], reranker: Reranker) -> list[str]:
    """Where freshly loaded models disagree with the caches; empty when they still match."""
    problems = []
    for variant, cached in caches.embedders.items():
        if variant not in embedders:
            problems.append(f"no fresh embedder for the variant '{variant}'")
            continue
        fresh = embedders[variant]
        if fresh.fingerprint != cached.fingerprint:
            problems.append(f"{variant}: the embedder fingerprint changed")
        pairs = {
            "documents": (fresh.embed_documents(caches.documents), cached.embed_documents(caches.documents)),
            "queries": (
                np.stack([fresh.embed_query(q) for q in caches.queries]),
                np.stack([cached.embed_query(q) for q in caches.queries]),
            ),
        }
        for kind, (new, old) in pairs.items():
            cosine = _min_cosine(new, old)
            if cosine < MIN_COSINE:
                problems.append(f"{variant}: {kind} drifted (lowest cosine {cosine:.4f} < {MIN_COSINE})")
    if reranker.fingerprint != caches.reranker.fingerprint:
        problems.append("the reranker fingerprint changed")
    fresh_scores = np.array([reranker.score(q, caches.documents) for q in caches.queries])
    cached_scores = np.array([caches.reranker.score(q, caches.documents) for q in caches.queries])
    diff = float(np.max(np.abs(fresh_scores - cached_scores)))
    if diff > MAX_RERANK_DIFF:
        problems.append(f"rerank scores drifted (largest difference {diff:.4g} > {MAX_RERANK_DIFF})")
    return problems
