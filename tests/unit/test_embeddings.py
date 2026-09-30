"""Offline tests for embeddings and reranking: fakes and stubbed fastembed models, no download."""

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from food_concierge import errors
from food_concierge.config import Settings
from food_concierge.models import embeddings
from food_concierge.models.embeddings import (
    EmbedderFingerprint,
    FastEmbedder,
    FastReranker,
    HashingEmbedder,
    OverlapReranker,
    get_embedder,
    get_reranker,
)


class StubTextEmbedding:
    """Stands in for fastembed's TextEmbedding: records calls and returns unnormalised vectors."""

    embedding_size = 3

    def __init__(self, model_dir: Path | None = None) -> None:
        self.passages: list[tuple[list[str], int]] = []
        self.queries: list[list[str]] = []
        self.model = type("Inner", (), {"_model_dir": str(model_dir) if model_dir else None})()

    def passage_embed(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[np.ndarray]:
        texts = list(texts)
        self.passages.append((texts, batch_size))
        return iter(np.array([len(t), 1.0, 0.0]) for t in texts)

    def query_embed(self, query: Iterable[str]) -> Iterator[np.ndarray]:
        self.queries.append(list(query))
        return iter([np.array([0.0, 3.0, 4.0])])


class StubCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> Iterator[np.float32]:
        documents = list(documents)
        self.calls.append((query, documents, batch_size))
        return iter(np.float32(i) for i in range(len(documents)))


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    directory = tmp_path / "snapshots" / "abc123"
    directory.mkdir(parents=True)
    (directory / "model_optimized.onnx").write_bytes(b"onnx weights")
    return directory


def _fingerprint(**changes: Any) -> EmbedderFingerprint:
    base = {
        "model": "m",
        "revision": "r1",
        "model_sha256": "h",
        "dim": 3,
        "normalized": True,
        "query_prefix": "q: ",
    }
    return EmbedderFingerprint(**{**base, **changes})


def test_documents_and_queries_use_separate_paths(settings: Settings, snapshot: Path) -> None:
    stub = StubTextEmbedding(snapshot)
    embedder = FastEmbedder(settings, model=stub)

    documents = embedder.embed_documents(["abc", "abcd"])
    query = embedder.embed_query("vegan curry")

    assert stub.passages == [(["abc", "abcd"], settings.embed_batch_size)]
    assert stub.queries == [[settings.embed_query_prefix + "vegan curry"]]
    assert documents.shape == (2, 3)
    assert documents.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(query, [0.0, 0.6, 0.8], rtol=1e-6)


def test_fingerprint_records_the_downloaded_snapshot(settings: Settings, snapshot: Path) -> None:
    fingerprint = FastEmbedder(settings, model=StubTextEmbedding(snapshot)).fingerprint

    assert fingerprint.model == settings.embed_model
    assert fingerprint.revision == "abc123"
    assert fingerprint.model_sha256 == hashlib.sha256(b"onnx weights").hexdigest()
    assert (fingerprint.dim, fingerprint.normalized) == (3, True)
    assert fingerprint.query_prefix == settings.embed_query_prefix


def test_fingerprint_without_a_known_snapshot(settings: Settings, tmp_path: Path) -> None:
    unknown = FastEmbedder(settings, model=StubTextEmbedding(None)).fingerprint
    no_onnx = FastEmbedder(settings, model=StubTextEmbedding(tmp_path)).fingerprint

    assert (unknown.revision, unknown.model_sha256) == ("unknown", "")
    assert no_onnx.model_sha256 == ""


def test_empty_document_list(settings: Settings, snapshot: Path) -> None:
    stub = StubTextEmbedding(snapshot)

    assert FastEmbedder(settings, model=stub).embed_documents([]).shape == (0, 3)
    assert stub.passages == []


def test_mismatched_index_is_refused() -> None:
    current = _fingerprint()

    current.require_match(_fingerprint())
    with pytest.raises(errors.NotReadyError, match=r"\(model_sha256, revision changed\)"):
        current.require_match(_fingerprint(revision="r2", model_sha256="other"))


def test_fingerprint_round_trips_through_json() -> None:
    fingerprint = _fingerprint()

    assert EmbedderFingerprint.model_validate_json(fingerprint.model_dump_json()) == fingerprint


def test_hashing_embedder_ranks_by_shared_words() -> None:
    embedder = HashingEmbedder()
    dishes = ["vegan chickpea curry", "chocolate lava cake", "paneer curry with rice"]

    scores = embedder.embed_documents(dishes) @ embedder.embed_query("vegan curry")

    assert scores.argmax() == 0
    assert scores[2] > scores[1]


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    first = HashingEmbedder().embed_documents(["Tofu Stir Fry", "tofu stir fry!"])
    second = HashingEmbedder().embed_documents(["tofu stir fry"])

    np.testing.assert_allclose(first[0], first[1])
    np.testing.assert_allclose(first[0], second[0])
    assert first.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-6)


def test_hashing_embedder_edge_cases() -> None:
    embedder = HashingEmbedder(dim=8)

    assert embedder.embed_documents([]).shape == (0, 8)
    assert not embedder.embed_query("").any()  # no words: a zero vector, not NaN
    assert embedder.fingerprint.model == "fake-hashing"


def test_fast_reranker_scores_each_document(settings: Settings) -> None:
    stub = StubCrossEncoder()
    reranker = FastReranker(settings, model=stub)

    scores = reranker.score("curry", ["a", "b"])

    assert scores == [0.0, 1.0]
    assert all(type(s) is float for s in scores)
    assert stub.calls == [("curry", ["a", "b"], settings.embed_batch_size)]
    assert reranker.score("curry", []) == []


def test_overlap_reranker() -> None:
    reranker = OverlapReranker()

    assert reranker.score("vegan curry", ["vegan chickpea curry", "curry", "cake"]) == [1.0, 0.5, 0.0]
    assert reranker.score("", ["cake"]) == [0.0]


def test_factories_follow_the_embed_provider(monkeypatch: pytest.MonkeyPatch, snapshot: Path) -> None:
    fake = Settings(_env_file=None, embed_provider="fake")
    assert isinstance(get_embedder(fake), HashingEmbedder)
    assert isinstance(get_reranker(fake), OverlapReranker)

    loaded: list[str] = []

    def load(model_cls: Any, model_name: str, settings: Settings) -> Any:
        loaded.append(model_name)
        return StubTextEmbedding(snapshot) if model_name == settings.embed_model else StubCrossEncoder()

    monkeypatch.setattr(embeddings, "_load", load)
    real = Settings(_env_file=None)
    assert isinstance(get_embedder(real), FastEmbedder)
    assert isinstance(get_reranker(real), FastReranker)
    assert loaded == [real.embed_model, real.rerank_model]


def test_unimplemented_embed_provider_is_a_config_error() -> None:
    with pytest.raises(errors.ConfigError, match="ollama"):
        get_embedder(Settings(_env_file=None, embed_provider="ollama"))


def test_model_loads_into_the_cache_outside_the_repo(settings: Settings) -> None:
    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, model_name: str, cache_dir: str) -> None:
            seen.update(model_name=model_name, cache_dir=cache_dir)

    embeddings._load(Recorder, "some/model", settings)

    assert seen == {"model_name": "some/model", "cache_dir": str(settings.model_cache_dir / "fastembed")}


def test_load_failure_is_not_ready(settings: Settings) -> None:
    def broken(model_name: str, cache_dir: str) -> None:
        raise OSError("download failed")

    with pytest.raises(errors.NotReadyError, match="could not be loaded") as caught:
        embeddings._load(broken, "some/model", settings)
    assert isinstance(caught.value.__cause__, OSError)


def test_default_loader_uses_fastembed(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    import fastembed

    calls: list[str] = []

    def text_embedding(name: str, cache_dir: str) -> str:
        calls.append(name)
        return "model"

    monkeypatch.setattr(fastembed, "TextEmbedding", text_embedding)

    assert embeddings._load_text_embedding(settings) == "model"
    assert calls == [settings.embed_model]
