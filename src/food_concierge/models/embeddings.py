"""Text embeddings and cross-encoder reranking, run locally with fastembed (ONNX, no torch).

Queries and documents are embedded through separate methods: BGE retrieves
better with an instruction prefix on short queries, fastembed does not add it,
and mixing the two up degrades search without any error. Vectors are
L2-normalised, which inner-product search (FAISS ``IndexFlatIP``) relies on.

Every index records the ``EmbedderFingerprint`` it was built with and refuses
to load under a different one. fastembed cannot pin a model revision, so the
fingerprint records the downloaded snapshot and a hash of the model file.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from food_concierge.config import Settings, get_settings
from food_concierge.errors import ConfigError, NotReadyError

Vectors = NDArray[np.float32]

_TOKEN = re.compile(r"\w+")


class EmbedderFingerprint(BaseModel):
    """Identifies exactly which embedding function produced a set of vectors."""

    model_config = ConfigDict(frozen=True)

    model: str
    revision: str
    model_sha256: str
    dim: int
    normalized: bool
    query_prefix: str

    def require_match(self, index_fingerprint: EmbedderFingerprint) -> None:
        """Raise unless an index built under ``index_fingerprint`` can be searched with this embedder."""
        if index_fingerprint != self:
            changed = sorted(
                name for name in type(self).model_fields if getattr(self, name) != getattr(index_fingerprint, name)
            )
            raise NotReadyError(
                f"The search index was built with a different embedder ({', '.join(changed)} changed); rebuild it."
            )


class Embedder(Protocol):
    @property
    def fingerprint(self) -> EmbedderFingerprint: ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """One normalised row per text, shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """One normalised vector, shape ``(dim,)``."""
        ...


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance of each document to ``query``; higher is more relevant."""
        ...


def _normalized(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    normalized: Vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return normalized


class FastEmbedder:
    """fastembed ``TextEmbedding`` behind the ``Embedder`` protocol."""

    def __init__(self, settings: Settings, model: Any = None) -> None:
        # ``model`` is injectable so tests can check the query/document split without a download.
        self._query_prefix = settings.embed_query_prefix
        self._batch_size = settings.embed_batch_size
        if model is None:
            model = _load_text_embedding(settings)
        self._model = model
        self._fingerprint = EmbedderFingerprint(
            model=settings.embed_model,
            revision=_snapshot_revision(model),
            model_sha256=_model_file_sha256(model),
            dim=int(model.embedding_size),
            normalized=True,
            query_prefix=self._query_prefix,
        )

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        return self._fingerprint

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._fingerprint.dim), dtype=np.float32)
        rows = self._model.passage_embed(list(texts), batch_size=self._batch_size)
        return _normalized(np.asarray(list(rows), dtype=np.float32))

    def embed_query(self, text: str) -> Vectors:
        (row,) = self._model.query_embed([self._query_prefix + text])
        return _normalized(np.asarray(row, dtype=np.float32))


class FastReranker:
    """fastembed cross-encoder behind the ``Reranker`` protocol (adopted in Phase 3 only on evidence)."""

    def __init__(self, settings: Settings, model: Any = None) -> None:
        if model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            model = _load(TextCrossEncoder, settings.rerank_model, settings)
        self._model = model
        self._batch_size = settings.embed_batch_size

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(s) for s in self._model.rerank(query, list(documents), batch_size=self._batch_size)]


class HashingEmbedder:
    """Deterministic offline embedder for tests: hashed word counts, so shared words mean similar vectors.

    Unlike random vectors, rankings are meaningful ("vegan curry" is closer to a curry than to a cake),
    which lets retrieval tests assert order without downloading a model.
    """

    def __init__(self, dim: int = 384, query_prefix: str = "") -> None:
        self._dim = dim
        self._fingerprint = EmbedderFingerprint(
            model="fake-hashing",
            revision="1",
            model_sha256="",
            dim=dim,
            normalized=True,
            query_prefix=query_prefix,
        )

    @property
    def fingerprint(self) -> EmbedderFingerprint:
        return self._fingerprint

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            # blake2b, not hash(): Python's string hash changes between processes.
            digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vector[digest % self._dim] += 1.0 if digest & (1 << 63) else -1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return _normalized(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> Vectors:
        return _normalized(self._vector(text))


class OverlapReranker:
    """Deterministic offline reranker for tests: the share of query words found in each document."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        words = set(_TOKEN.findall(query.lower()))
        if not words:
            return [0.0] * len(documents)
        return [len(words & set(_TOKEN.findall(doc.lower()))) / len(words) for doc in documents]


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embed_provider == "fastembed":
        return FastEmbedder(settings)
    if settings.embed_provider == "fake":
        return HashingEmbedder(query_prefix=settings.embed_query_prefix)
    raise ConfigError(f"Embedding provider '{settings.embed_provider}' is not implemented; use fastembed.")


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if settings.embed_provider == "fake":
        return OverlapReranker()
    return FastReranker(settings)


def _load_text_embedding(settings: Settings) -> Any:
    from fastembed import TextEmbedding

    return _load(TextEmbedding, settings.embed_model, settings)


def _load(model_cls: Any, model_name: str, settings: Settings) -> Any:
    # Imported lazily: loading ONNX Runtime is slow, and tests with fakes never need it.
    cache_dir = settings.model_cache_dir / "fastembed"
    try:
        return model_cls(model_name, cache_dir=str(cache_dir))
    except Exception as exc:
        raise NotReadyError(f"Model '{model_name}' could not be loaded.") from exc


def _model_dir(model: Any) -> Path | None:
    # fastembed keeps the downloaded snapshot directory on the inner ONNX model; private, so read defensively.
    inner = getattr(model, "model", model)
    directory = getattr(inner, "_model_dir", None)
    return Path(directory) if directory else None


def _snapshot_revision(model: Any) -> str:
    directory = _model_dir(model)
    return directory.name if directory else "unknown"


def _model_file_sha256(model: Any) -> str:
    directory = _model_dir(model)
    if directory is None:
        return ""
    onnx_files = sorted(directory.glob("*.onnx"))
    if not onnx_files:
        return ""
    digest = hashlib.sha256()
    with onnx_files[0].open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
