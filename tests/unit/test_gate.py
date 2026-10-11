"""The PR retrieval gate: its rules, and deliberately degraded pipelines failing it; real catalog, fake models."""

import dataclasses
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from food_concierge.config import REPO_ROOT
from food_concierge.errors import EvaluationError
from food_concierge.evaluation.caches import EvalCaches, build_caches, load_caches, texts_sha256
from food_concierge.evaluation.dataset import RetrievalDataset, TextQuery
from food_concierge.evaluation.gate import (
    GateBaseline,
    GatedCandidate,
    check_gate,
    run_gate,
    updated_baseline,
)
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.runner import Candidate, QueryResult
from food_concierge.ingestion.allergens import load_lexicon, tag_item
from food_concierge.ingestion.loader import CatalogLoad, load_catalog
from food_concierge.models.embeddings import HashingEmbedder, OverlapReranker, RerankerFingerprint
from food_concierge.services.retrieval import RetrievalConfig
from food_concierge.storage.catalog_db import CatalogStore, Filters, build_catalog_db

RERANKED = Candidate(name="reranked", config=RetrievalConfig(mode="hybrid", rerank_depth=10))


def row(query_id: str, violation: float = 0.0, ndcg: float | None = None) -> QueryResult:
    metrics = {"violation@3": violation} | ({} if ndcg is None else {"ndcg@3": ndcg})
    return QueryResult(query_id, "vague", (), metrics, ())


def baseline(ndcg: float | None) -> GateBaseline:
    return GateBaseline(reason="test", gated=(GatedCandidate(candidate=RERANKED, ndcg_at_3=ndcg),))


def test_the_gate_fails_on_any_violation_and_on_a_drop_beyond_the_tolerance() -> None:
    healthy = {"reranked": [row("t001", ndcg=0.8), row("t002", ndcg=0.6), row("t003")]}

    assert check_gate(healthy, baseline(0.71)).passed  # 0.70 is within 0.02
    assert check_gate(healthy, baseline(None)).ndcg_at_3 == {"reranked": pytest.approx(0.7)}
    dropped = check_gate(healthy, baseline(0.75))
    assert dropped.failures == ("reranked: nDCG@3 0.7000 is more than 0.02 below the baseline 0.7500",)
    violating = check_gate({"reranked": [row("t001", violation=1 / 3), row("t002")]}, baseline(None))
    assert violating.failures == ("reranked: dishes that break the filters on t001",)
    ungraded = check_gate({"reranked": [row("t001")]}, baseline(0.5))
    assert ungraded.ndcg_at_3 == {"reranked": None}
    assert ungraded.failures == ("reranked: no graded query to compare with the baseline nDCG@3",)


@pytest.fixture(scope="module")
def loaded() -> CatalogLoad:
    return load_catalog(REPO_ROOT / "data" / "raw")


@pytest.fixture(scope="module")
def store(loaded: CatalogLoad, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CatalogStore]:
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    tags = {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}
    path = tmp_path_factory.mktemp("db") / "catalog.db"
    build_catalog_db(loaded.catalog, tags, path, lexicon_version=1, source_sha256="abc")
    opened = CatalogStore(path)
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def dataset(loaded: CatalogLoad) -> RetrievalDataset:
    """Each dish's own name, graded 3 for that dish, plus dairy dishes asked for without milk."""
    named = [
        TextQuery(id=f"t{n:03d}", text=item.name, type="known_item", author="drafted", grades={item.item_id: 3})
        for n, item in enumerate(loaded.catalog.items[:12], start=1)
    ]
    no_milk = Filters.model_validate({"exclude_allergens": ["milk"]})
    constrained = [
        TextQuery(id=f"t{n:03d}", text=text, type="constrained", author="drafted", filters=no_milk)
        for n, text in enumerate(["paneer butter masala", "creamy kheer with milk", "mango lassi"], start=13)
    ]
    return RetrievalDataset(queries=(*named, *constrained))


@pytest.fixture(scope="module")
def caches(store: CatalogStore, dataset: RetrievalDataset, tmp_path_factory: pytest.TempPathFactory) -> EvalCaches:
    texts = [query.text for query in dataset.queries]
    directory = tmp_path_factory.mktemp("caches") / "retrieval"
    documents = list(store.doc_texts().values())
    build_caches(
        directory, {"prefix": HashingEmbedder()}, OverlapReranker(), documents, texts, "abc", 1, texts_sha256(texts)
    )
    return load_caches(directory)


@pytest.fixture(scope="module")
def truth(loaded: CatalogLoad) -> GroundTruth:
    return GroundTruth.from_load(loaded)


@pytest.fixture(scope="module")
def healthy(
    caches: EvalCaches,
    store: CatalogStore,
    truth: GroundTruth,
    dataset: RetrievalDataset,
    tmp_path_factory: pytest.TempPathFactory,
) -> GateBaseline:
    """The baseline as measured on the healthy pipeline, which passes its own gate."""
    report = run_gate(caches, store, truth, dataset, baseline(None), tmp_path_factory.mktemp("work"))
    assert report.passed
    measured = updated_baseline(baseline(None), report, "measured")
    assert measured.reason == "measured"
    assert measured.gated[0].ndcg_at_3 is not None
    assert measured.gated[0].ndcg_at_3 > 0.5
    return measured


class RandomReranker:
    """Scores in random order: a ranker that ignores the query."""

    fingerprint = RerankerFingerprint(model="fake-random", revision="1", model_sha256="")

    def __init__(self) -> None:
        self._rng = random.Random(7)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [self._rng.random() for _ in documents]


def test_a_random_order_ranker_fails_the_gate(
    caches: EvalCaches,
    store: CatalogStore,
    truth: GroundTruth,
    dataset: RetrievalDataset,
    healthy: GateBaseline,
    tmp_path: Path,
) -> None:
    degraded = dataclasses.replace(caches, reranker=RandomReranker())  # type: ignore[arg-type]

    report = run_gate(degraded, store, truth, dataset, healthy, tmp_path)

    assert not report.passed
    assert "below the baseline" in report.failures[-1]


def test_a_pipeline_without_sql_filters_fails_the_gate(
    caches: EvalCaches,
    store: CatalogStore,
    truth: GroundTruth,
    dataset: RetrievalDataset,
    healthy: GateBaseline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    everything = store.candidate_ids(Filters())
    monkeypatch.setattr(store, "candidate_ids", lambda _filters: everything)

    report = run_gate(caches, store, truth, dataset, healthy, tmp_path)

    assert not report.passed
    assert "dishes that break the filters" in report.failures[0]


def test_stale_caches_fail_the_gate_naming_the_field(
    caches: EvalCaches, store: CatalogStore, truth: GroundTruth, dataset: RetrievalDataset, tmp_path: Path
) -> None:
    reworded = dataset.model_copy(
        update={"queries": (dataset.queries[0].model_copy(update={"text": "something new"}), *dataset.queries[1:])}
    )

    with pytest.raises(EvaluationError, match="dataset_sha256 changed"):
        run_gate(caches, store, truth, reworded, baseline(None), tmp_path)


def test_the_committed_baseline_reads() -> None:
    committed = GateBaseline.model_validate_json(
        (REPO_ROOT / "eval" / "gates" / "retrieval_baseline.json").read_text(encoding="utf-8")
    )

    assert committed.tolerance == 0.02
    assert [gated.candidate.name for gated in committed.gated] == ["production"]
