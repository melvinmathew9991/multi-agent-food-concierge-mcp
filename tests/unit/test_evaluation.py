"""Evaluation core: dataset split and lock, metrics against hand-computed values, bootstrap, caches, runner."""

import math
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from food_concierge.config import REPO_ROOT
from food_concierge.errors import ConfigError, EvaluationError
from food_concierge.evaluation import caches as caches_module
from food_concierge.evaluation.bootstrap import Interval, bootstrap_mean, paired_bootstrap
from food_concierge.evaluation.caches import build_caches, load_caches
from food_concierge.evaluation.dataset import (
    RetrievalDataset,
    TextQuery,
    assign_splits,
    load_dataset,
    lock,
    select,
    split_digest,
)
from food_concierge.evaluation.metrics import (
    GroundTruth,
    judged_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
    violation_at_k,
)
from food_concierge.evaluation.record import current_machine, git_sha, make_record
from food_concierge.evaluation.runner import (
    Candidate,
    QueryResult,
    compare,
    latency,
    run_candidates,
    score_query,
    summarize,
)
from food_concierge.ingestion.allergens import load_lexicon, tag_item
from food_concierge.ingestion.loader import CatalogLoad, load_catalog
from food_concierge.ingestion.taxonomy import Allergen, Diet
from food_concierge.models.embeddings import HashingEmbedder, OverlapReranker
from food_concierge.services.retrieval import RetrievalConfig
from food_concierge.storage.catalog_db import CatalogStore, Filters, build_catalog_db

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def query(query_id: str, **fields: object) -> TextQuery:
    base: dict[str, object] = {"id": query_id, "text": f"query {query_id}", "type": "vague", "author": "drafted"}
    return TextQuery.model_validate(base | fields)


# Dataset


def test_query_grades_must_be_zero_to_three() -> None:
    with pytest.raises(ValidationError, match="0-3"):
        query("t001", grades={"a01": 4})
    assert query("t001", grades={"a01": 3, "b02": 2}).exact == frozenset({"a01"})
    with pytest.raises(ValidationError, match="target"):
        query("t001", type="vague", target="dal-makhani")


def test_dataset_rejects_duplicate_ids_and_groups_split_across() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        RetrievalDataset(queries=(query("t001"), query("t001")))
    with pytest.raises(ValidationError, match="both splits"):
        RetrievalDataset(queries=(query("t001", group="g", split="dev"), query("t002", group="g", split="test")))


def test_assign_splits_halves_each_type_keeps_groups_together_and_is_seeded() -> None:
    queries = [query(f"t{n:03d}", type="vague" if n < 8 else "attribute") for n in range(12)]
    queries[0] = queries[0].model_copy(update={"group": "g"})
    queries[1] = queries[1].model_copy(update={"group": "g"})

    split = assign_splits(queries, seed=7)

    assert split == assign_splits(queries, seed=7)
    assert split != assign_splits(queries, seed=8)
    assert split[0].split == split[1].split
    by_type = Counter((q.type, q.split) for q in split)
    assert by_type[("attribute", "dev")] == by_type[("attribute", "test")] == 2
    units = Counter((q.unit, q.split) for q in split if q.type == "vague")
    sides = Counter(side for _, side in units)  # 7 units (6 single queries and one group of 2): split 4/3
    assert abs(sides["dev"] - sides["test"]) == 1


def test_lock_fixes_the_test_split_and_select_guards_it() -> None:
    dataset = RetrievalDataset(queries=assign_splits([query(f"t{n:03d}") for n in range(6)], seed=7))
    with pytest.raises(EvaluationError, match="not locked"):
        select(dataset, "test", final=True)

    locked = lock(dataset)
    assert locked.test_sha256 == split_digest(locked.queries)
    assert len(select(locked, "dev")) == 3
    with pytest.raises(EvaluationError, match="final"):
        select(locked, "test")
    assert len(select(locked, "test", final=True)) == 3

    tampered = [q.model_copy(update={"text": "changed"}) if q.split == "test" else q for q in locked.queries]
    with pytest.raises(ValidationError, match="changed after it was locked"):
        RetrievalDataset(test_sha256=locked.test_sha256, queries=tuple(tampered))
    with pytest.raises(EvaluationError, match="split"):
        lock(RetrievalDataset(queries=(query("t001"),)))


def test_load_dataset_reads_yaml_and_reports_problems(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "queries": [
                    {
                        "id": "t001",
                        "text": "light vegan lunch",
                        "type": "constrained",
                        "author": "owner",
                        "filters": {"diet": "vegan", "exclude_allergens": ["peanuts"], "max_kcal": 450},
                        "grades": {"h01": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_dataset(path).queries[0]

    assert loaded.filters == Filters(diet=Diet.VEGAN, exclude_allergens=frozenset({Allergen.PEANUTS}), max_kcal=450)
    path.write_text("queries: [{id: bad}]", encoding="utf-8")
    with pytest.raises(EvaluationError, match=r"retrieval\.yaml"):
        load_dataset(path)
    with pytest.raises(EvaluationError):
        load_dataset(tmp_path / "missing.yaml")


# Metrics


def test_ndcg_matches_a_hand_computed_value() -> None:
    grades = {"a": 3, "b": 2, "c": 1, "d": 0}
    # DCG of b, x, a = 3/log2(2) + 0 + 7/log2(4); ideal a, b, c = 7 + 3/log2(3) + 1/log2(4).
    expected = (3 + 7 / 2) / (7 + 3 / math.log2(3) + 1 / 2)

    assert ndcg_at_k(["b", "x", "a"], grades) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b", "c"], grades) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no relevant"):
        ndcg_at_k(["a"], {"a": 0})


def test_success_recall_rank_and_judged() -> None:
    grades = {"a": 1, "b": 2}
    assert success_at_k(["a", "x", "y", "b"], grades) == 0.0  # b is 4th
    assert success_at_k(["a", "b"], grades) == 1.0
    assert recall_at_k(["a", "x", "b"], {"b", "z"}, 3) == 0.5
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], set(), 3)
    assert reciprocal_rank(["x", "y", "b"], {"b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"b"}) == 0.0
    assert judged_at_k(["a", "x", "b"], grades) == pytest.approx(2 / 3)
    assert judged_at_k([], grades) == 1.0


@pytest.fixture(scope="module")
def loaded() -> CatalogLoad:
    return load_catalog(REPO_ROOT / "data" / "raw")


@pytest.fixture(scope="module")
def truth(loaded: CatalogLoad) -> GroundTruth:
    return GroundTruth.from_load(loaded)


def test_violations_are_judged_against_the_reviewed_allergens(truth: GroundTruth, loaded: CatalogLoad) -> None:
    with_milk = next(i for i, found in loaded.true_allergens.items() if Allergen.MILK in found)
    without_milk = next(i for i, found in loaded.true_allergens.items() if not found)
    no_milk = Filters(exclude_allergens=frozenset({Allergen.MILK}))

    assert violation_at_k([with_milk, without_milk], no_milk, truth) == 0.5
    assert violation_at_k([with_milk], Filters(), truth) == 0.0
    assert violation_at_k([], no_milk, truth) == 0.0


# Bootstrap


def test_bootstrap_mean_is_seeded_and_brackets_the_mean() -> None:
    values = [0.0, 1.0, 1.0, 0.5, 0.25, 0.75]
    interval = bootstrap_mean(values, seed=7, resamples=2000)

    assert interval == bootstrap_mean(values, seed=7, resamples=2000)
    assert interval.mean == pytest.approx(3.5 / 6)
    assert interval.low <= interval.mean <= interval.high
    assert bootstrap_mean([0.4] * 5, seed=1) == Interval(0.4, 0.4, 0.4)
    with pytest.raises(ValueError, match="no values"):
        bootstrap_mean([], seed=1)


def test_paired_bootstrap_resamples_queries_together() -> None:
    better = paired_bootstrap([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6], seed=7)

    assert better.mean == pytest.approx(0.2)
    assert better.excludes_zero
    assert not paired_bootstrap([0.1, 0.5], [0.5, 0.1], seed=7).excludes_zero
    assert Interval(-0.2, -0.3, -0.1).excludes_zero
    with pytest.raises(ValueError, match="one value per query"):
        paired_bootstrap([0.1], [0.1, 0.2], seed=7)


# Caches

DOCS = ["vegan chickpea curry", "paneer butter masala", "mango lassi"]
QUERIES = ["chickpea curry", "sweet drink"]


def make_caches(directory: Path) -> Path:
    embedders = {"prefix": HashingEmbedder(query_prefix="query: "), "no_prefix": HashingEmbedder()}
    build_caches(directory, embedders, OverlapReranker(), DOCS, QUERIES, "abc", 1, "def", now=lambda: NOW)
    return directory


def test_caches_serve_what_the_real_models_returned(tmp_path: Path) -> None:
    caches = load_caches(make_caches(tmp_path / "caches"))
    real = HashingEmbedder(query_prefix="query: ")
    cached = caches.embedders["prefix"]

    assert cached.fingerprint == real.fingerprint
    assert caches.embedders["no_prefix"].fingerprint.query_prefix == ""
    np.testing.assert_allclose(cached.embed_documents(DOCS[::-1]), real.embed_documents(DOCS[::-1]), atol=1e-6)
    np.testing.assert_allclose(cached.embed_query("chickpea curry"), real.embed_query("chickpea curry"), atol=1e-6)
    assert cached.embed_documents([]).shape == (0, real.fingerprint.dim)
    assert caches.reranker.score("chickpea curry", DOCS) == OverlapReranker().score("chickpea curry", DOCS)
    assert caches.reranker.fingerprint == OverlapReranker.fingerprint


def test_a_cache_miss_is_an_error_not_a_zero(tmp_path: Path) -> None:
    caches = load_caches(make_caches(tmp_path / "caches"))
    cached = caches.embedders["prefix"]

    with pytest.raises(EvaluationError, match="document vector for 'pizza'"):
        cached.embed_documents(["pizza"])
    with pytest.raises(EvaluationError, match="query vector"):
        cached.embed_query("x" * 80)
    with pytest.raises(EvaluationError, match="rerank scores for 'unknown'"):
        caches.reranker.score("unknown", DOCS)
    with pytest.raises(EvaluationError, match="rerank score for 'pizza'"):
        caches.reranker.score("chickpea curry", ["pizza"])


def test_stale_or_changed_caches_are_refused(tmp_path: Path) -> None:
    directory = make_caches(tmp_path / "caches")
    caches = load_caches(directory)

    caches.require_current("abc", 1, "def")
    with pytest.raises(EvaluationError, match=r"data_sha256, dataset_sha256 changed"):
        caches.require_current("other", 1, "changed")

    (directory / caches_module.QUERIES).write_text("[]", encoding="utf-8")
    with pytest.raises(EvaluationError, match=r"queries\.json changed"):
        load_caches(directory)
    with pytest.raises(EvaluationError, match="missing"):
        load_caches(tmp_path / "nowhere")


def test_caches_need_queries_and_one_embedding_model(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="one query"):
        build_caches(tmp_path, {}, OverlapReranker(), DOCS, QUERIES, "abc", 1, "def")
    mixed = {"a": HashingEmbedder(dim=8), "b": HashingEmbedder()}
    with pytest.raises(ConfigError, match="same embedding model"):
        build_caches(tmp_path, mixed, OverlapReranker(), DOCS, QUERIES, "abc", 1, "def")


# Run record


def test_run_record_names_the_commit_lock_and_machine() -> None:
    fingerprint = HashingEmbedder().fingerprint
    record = make_record(REPO_ROOT, "abc", {"prefix": fingerprint}, None, {"eval": 7}, now=lambda: NOW)

    assert len(record.git_sha) == 40
    assert len(record.lock_sha256) == 64
    assert record.machine == current_machine()
    assert record.machine.cores >= 1
    assert record.models == {}


def test_git_sha_outside_a_checkout_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="git commit"):
        git_sha(tmp_path / "missing")


# Runner


@pytest.fixture(scope="module")
def store(loaded: CatalogLoad, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CatalogStore]:
    lexicon = load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")
    tags = {item.item_id: tag_item(item, lexicon) for item in loaded.catalog.items}
    path = tmp_path_factory.mktemp("db") / "catalog.db"
    build_catalog_db(loaded.catalog, tags, path, lexicon_version=1, source_sha256="abc")
    opened = CatalogStore(path)
    yield opened
    opened.close()


def test_runner_scores_each_candidate_and_variant(store: CatalogStore, truth: GroundTruth, tmp_path: Path) -> None:
    keys = store.dish_keys()
    naan = sorted(i for i, key in keys.items() if key == "butter-naan")
    queries = [
        query("t001", text="butter naan", type="known_item", grades=dict.fromkeys(naan, 3)),
        query("t002", text="something to drink", filters=Filters(max_kcal=200)),
    ]
    caches = load_caches(
        build_caches_for(tmp_path / "caches", store, [q.text for q in queries]),
    )
    candidates = [
        Candidate(name="bm25", config=RetrievalConfig(mode="bm25")),
        Candidate(name="hybrid", config=RetrievalConfig(mode="hybrid", rerank_depth=5), variant="no_prefix"),
    ]

    results = run_candidates(
        store, truth, queries, candidates, caches.embedders, caches.reranker, tmp_path / "work", repeats=2
    )

    bm25 = {row.query_id: row for row in results["bm25"]}
    assert bm25["t001"].metrics["recall@10"] == 1.0
    assert set(bm25["t001"].metrics) == {
        "violation@3",
        "ndcg@3",
        "success@3",
        "judged@3",
        "recall@3",
        "recall@10",
        "mrr",
    }
    assert set(bm25["t002"].metrics) == {"violation@3"}  # ungraded: only the violation metric applies
    assert bm25["t002"].metrics["violation@3"] == 0.0
    assert len(bm25["t001"].ranked) == 10
    assert len(bm25["t001"].timings_ms) == 2
    assert "rerank" in latency(results["hybrid"])
    assert summarize(results["bm25"], seed=7)["recall@10"].mean == 1.0
    assert compare(results["bm25"], results["hybrid"], "violation@3", seed=7).mean == 0.0
    with pytest.raises(EvaluationError, match="no_such_metric"):
        compare(results["bm25"], results["hybrid"], "no_such_metric", seed=7)
    missing = [Candidate(name="x", config=RetrievalConfig(), variant="missing")]
    with pytest.raises(EvaluationError, match="variant 'missing'"):
        run_candidates(store, truth, queries, missing, caches.embedders, None, tmp_path / "work2")


def build_caches_for(directory: Path, store: CatalogStore, queries: list[str]) -> Path:
    meta = store.meta()
    embedders = {"prefix": HashingEmbedder(), "no_prefix": HashingEmbedder(query_prefix="q: ")}
    documents = list(store.doc_texts().values())
    build_caches(directory, embedders, OverlapReranker(), documents, queries, meta["data_sha256"], 1, "x")
    return directory


def test_score_query_leaves_out_metrics_a_query_cannot_define(truth: GroundTruth) -> None:
    item = next(iter(truth.items))
    known_without_target = query("t001", type="known_item", grades={item: 2})

    metrics = score_query(known_without_target, [item], truth)

    assert "recall@3" not in metrics
    assert metrics["success@3"] == 1.0
    assert QueryResult("t001", "vague", (), {}, ()).metrics == {}
