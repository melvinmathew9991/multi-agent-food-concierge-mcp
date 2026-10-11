"""The evaluate and report flows: every candidate scored from caches, results files, the report and the mirror."""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from prefect.testing.utilities import prefect_test_harness

from food_concierge.config import REPO_ROOT, get_settings
from food_concierge.errors import EvaluationError
from food_concierge.evaluation.caches import EvalCaches, build_caches, load_caches, texts_sha256
from food_concierge.evaluation.dataset import RetrievalDataset, TextQuery, assign_splits, lock
from food_concierge.evaluation.gate import GateBaseline, GatedCandidate
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.mirror import mirror_results
from food_concierge.evaluation.original import PROMPT_VERSION, OriginalRun
from food_concierge.evaluation.record import RunRecord, make_record
from food_concierge.evaluation.report import render_report
from food_concierge.evaluation.results import (
    CANDIDATES,
    GateOutcome,
    RetrievalResults,
    differences,
    evaluate,
    latest_results,
    read_results,
    write_results,
)
from food_concierge.evaluation.runner import Candidate
from food_concierge.flows import __main__ as cli
from food_concierge.flows import evaluation as flows
from food_concierge.ingestion.allergens import load_lexicon, tag_item
from food_concierge.ingestion.loader import CatalogLoad, load_catalog
from food_concierge.models.embeddings import HashingEmbedder, OverlapReranker
from food_concierge.services.retrieval import RetrievalConfig
from food_concierge.storage.catalog_db import CatalogStore, build_catalog_db
from food_concierge.telemetry import Telemetry

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
MODEL = "groq-model"
PRODUCTION = GateBaseline(
    reason="test",
    gated=(
        GatedCandidate(candidate=Candidate(name="production", config=RetrievalConfig())),
        GatedCandidate(candidate=CANDIDATES[0]),  # already a candidate: run once, not twice
    ),
)


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
def truth(loaded: CatalogLoad) -> GroundTruth:
    return GroundTruth.from_load(loaded)


def named_queries(loaded: CatalogLoad, graded: bool = True) -> tuple[TextQuery, ...]:
    return tuple(
        TextQuery(
            id=f"t{n:03d}",
            text=item.name,
            type="known_item" if n % 2 else "vague",
            author="owner" if n == 1 else "drafted",
            grades={item.item_id: 3} if graded else {},
        )
        for n, item in enumerate(loaded.catalog.items[:8], start=1)
    )


@pytest.fixture(scope="module")
def split(loaded: CatalogLoad) -> RetrievalDataset:
    return lock(RetrievalDataset(queries=assign_splits(named_queries(loaded), seed=7)))


@pytest.fixture(scope="module")
def caches(store: CatalogStore, loaded: CatalogLoad, tmp_path_factory: pytest.TempPathFactory) -> EvalCaches:
    texts = [query.text for query in named_queries(loaded)]
    directory = tmp_path_factory.mktemp("caches") / "retrieval"
    embedders = {"prefix": HashingEmbedder(), "no_prefix": HashingEmbedder()}
    documents = list(store.doc_texts().values())
    build_caches(directory, embedders, OverlapReranker(), documents, texts, "abc", 1, texts_sha256(texts))
    return load_caches(directory)


def verdicts(queries: tuple[TextQuery, ...]) -> dict[str, OriginalRun]:
    return {
        q.id: OriginalRun(
            query_id=q.id,
            request=q.text,
            model=MODEL,
            prompt_version=PROMPT_VERSION,
            retrieved=tuple(q.grades) or ("k01",),
            replies={item: "Yes." for item in tuple(q.grades) or ("k01",)},
        )
        for q in queries
    }


@pytest.fixture(scope="module")
def record() -> RunRecord:
    return make_record(REPO_ROOT, "abc", {}, None, {"eval_seed": 7}, now=lambda: NOW)


def run(
    dataset: RetrievalDataset,
    store: CatalogStore,
    truth: GroundTruth,
    caches: EvalCaches,
    record: RunRecord,
    tmp_path: Path,
    **options: Any,
) -> RetrievalResults:
    return evaluate(
        store,
        truth,
        dataset,
        caches,
        verdicts(dataset.queries),
        PRODUCTION,
        record,
        tmp_path,
        c0_model=MODEL,
        **options,
    )


def test_evaluate_scores_every_candidate_and_compares_the_ladder(
    split: RetrievalDataset,
    store: CatalogStore,
    truth: GroundTruth,
    caches: EvalCaches,
    record: RunRecord,
    tmp_path: Path,
) -> None:
    results = run(split, store, truth, caches, record, tmp_path)

    names = [c.name for c in results.candidates]
    assert names[:2] == ["C0_strict", "C0_fixed"]
    assert names[2:] == [c.name for c in CANDIDATES] + ["production"]
    assert results.split == "dev"
    assert results.test_sha256 == split.test_sha256
    assert {len(c.queries) for c in results.candidates} == {4}
    fixed = results.candidates[1]
    assert fixed.summary["ndcg@3"].mean == 1.0  # "Yes." on the graded dish; the strict parser drops it
    assert results.candidates[0].summary["empty"].mean == 1.0
    assert fixed.slices["type:known_item"]["ndcg@3"].n + fixed.slices["type:vague"]["ndcg@3"].n == 4
    assert [(c.baseline, c.candidate) for c in results.comparisons] == [
        ("C1_bm25", "C4_hybrid"),
        ("C2_dense", "C4_hybrid"),
        ("C4_hybrid", "C5_hybrid_rerank10"),
        ("C4_hybrid", "C5_hybrid_rerank20"),
        ("C5_hybrid_rerank10", "C5_hybrid_rerank20"),
        ("C2_dense", "C2_dense_no_prefix"),
    ]
    assert results.comparisons[0].delta.n == 4
    assert results.gate.passed
    assert run(split, store, truth, caches, record, tmp_path) == results  # deterministic


def test_the_test_split_needs_final_and_unsplit_queries_run_as_all(
    split: RetrievalDataset,
    loaded: CatalogLoad,
    store: CatalogStore,
    truth: GroundTruth,
    caches: EvalCaches,
    record: RunRecord,
    tmp_path: Path,
) -> None:
    with pytest.raises(EvaluationError, match="final"):
        run(split, store, truth, caches, record, tmp_path, split="test")
    assert run(split, store, truth, caches, record, tmp_path, split="test", final=True).split == "test"

    ungraded = RetrievalDataset(queries=named_queries(loaded, graded=False))
    results = run(ungraded, store, truth, caches, record, tmp_path)
    assert results.split == "all"
    assert results.comparisons == ()
    assert "ndcg@3" not in results.candidates[2].summary


def test_evaluate_refuses_queries_c0_has_not_asked(
    split: RetrievalDataset,
    store: CatalogStore,
    truth: GroundTruth,
    caches: EvalCaches,
    record: RunRecord,
    tmp_path: Path,
) -> None:
    asked = verdicts(split.queries)
    first_dev = next(q.id for q in split.queries if q.split == "dev")
    del asked[first_dev]

    with pytest.raises(EvaluationError, match=f"no cached verdicts for {first_dev}"):
        evaluate(store, truth, split, caches, asked, PRODUCTION, record, tmp_path, c0_model=MODEL)


def test_results_round_trip_and_differences_name_what_changed(
    split: RetrievalDataset,
    store: CatalogStore,
    truth: GroundTruth,
    caches: EvalCaches,
    record: RunRecord,
    tmp_path: Path,
) -> None:
    results = run(split, store, truth, caches, record, tmp_path)
    path = write_results(tmp_path / "results", results)

    assert path.name == "retrieval_2026-10-11_dev.json"
    assert latest_results(tmp_path / "results", "dev") == path
    assert read_results(path) == results
    assert differences(results, results.model_copy(update={"record": record.model_copy(update={"git_sha": "x"})})) == []
    first = results.candidates[2]
    changed = results.model_copy(
        update={
            "split": "all",
            "candidates": (first.model_copy(update={"queries": first.queries[:1]}), *results.candidates[3:], first),
            "comparisons": (),
            "gate": GateOutcome(passed=False, failures=("x",)),
        }
    )
    changed = changed.model_copy(update={"candidates": changed.candidates[:-1]})
    assert differences(results, changed) == [
        "split differs",
        "C0_fixed: only in one run",
        "C0_strict: only in one run",
        "C1_bm25: numbers differ",
        "comparisons differ",
        "gate outcome differs",
    ]
    with pytest.raises(EvaluationError, match="No retrieval results"):
        latest_results(tmp_path / "nowhere")


def test_the_report_is_deterministic_and_shows_what_exists(
    split: RetrievalDataset,
    loaded: CatalogLoad,
    store: CatalogStore,
    truth: GroundTruth,
    caches: EvalCaches,
    record: RunRecord,
    tmp_path: Path,
) -> None:
    graded = run(split, store, truth, caches, record, tmp_path)
    page = render_report(graded)

    assert page == render_report(graded)
    assert "- **Split:** dev\n" in page
    assert "| C4_hybrid vs C1_bm25 | nDCG@3 |" in page
    assert "nDCG@3 by slice." in page
    assert "author:drafted" in page
    assert "\n## Gate\n\nPassed.\n" in page

    ungraded = run(
        RetrievalDataset(queries=named_queries(loaded, graded=False)), store, truth, caches, record, tmp_path
    )
    failing = ungraded.model_copy(update={"gate": GateOutcome(passed=False, failures=("production: broke t001",))})
    page = render_report(failing)
    assert "(queries not split yet)" in page
    assert "None yet: comparisons are on nDCG@3" in page
    assert "Violation@3 by slice." in page
    assert "- production: broke t001\n" in page


class FakeLangfuse:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.runs: dict[str, list[tuple[list[str], list[Any]]]] = {}
        self.flushed = False

    def create_dataset(self, **_: Any) -> None:
        return None

    def create_dataset_item(self, **fields: Any) -> Any:
        self.items.append(fields)
        return type("Item", (), {"metadata": fields["metadata"]})()

    def run_experiment(self, *, run_name: str, data: list[Any], task: Any, evaluators: list[Any], **_: Any) -> None:
        self.runs[run_name] = [
            (task(item=item), evaluators[0](input=None, output=None, expected_output=None, metadata=item.metadata))
            for item in data
        ]

    def flush(self) -> None:
        self.flushed = True


class BrokenLangfuse(FakeLangfuse):
    def create_dataset(self, **_: Any) -> None:
        raise ConnectionError("refused")


def test_the_mirror_sends_items_and_runs_and_swallows_failures(
    split: RetrievalDataset,
    store: CatalogStore,
    truth: GroundTruth,
    caches: EvalCaches,
    record: RunRecord,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    results = run(split, store, truth, caches, record, tmp_path)
    client = FakeLangfuse()

    assert not mirror_results(None, results, split.queries)
    assert mirror_results(client, results, split.queries)
    assert len(client.items) == 4  # dev queries only
    assert client.items[0]["id"].startswith("retrieval-dev-t")
    ranked, scores = client.runs[f"C0_fixed @ {record.git_sha[:7]}"][0]
    assert len(ranked) == 1
    assert {s.name for s in scores} == {"ndcg@3", "success@3", "violation@3"}
    assert client.flushed
    with caplog.at_level(logging.WARNING):
        assert not mirror_results(BrokenLangfuse(), results, split.queries)
    assert "langfuse mirror failed" in caplog.text


@pytest.fixture
def repo_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(flows, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(flows, "REPORT", tmp_path / "evaluation.md")
    monkeypatch.setattr(flows, "init_telemetry", lambda settings: Telemetry())  # no Langfuse
    get_settings.cache_clear()  # settings from the repository, not a test's temporary data directory
    return tmp_path


@pytest.fixture
def prefect_harness() -> Iterator[None]:
    with prefect_test_harness():
        yield


@pytest.mark.usefixtures("prefect_harness")
def test_the_flows_evaluate_the_committed_caches_reproducibly_then_report(
    repo_outputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["evaluate"]) == 0
    written = sorted((repo_outputs / "results").iterdir())
    assert len(written) == 1
    assert cli.main(["evaluate", "--check"]) == 0
    assert cli.main(["report"]) == 0
    out = capsys.readouterr().out

    assert f"results: {written[0]}" in out
    assert "reproduced the committed results" in out
    assert (repo_outputs / "evaluation.md").read_text(encoding="utf-8").startswith("# Retrieval evaluation\n")
    outcome = flows.evaluate_flow(check=True)
    assert outcome.path is None
    assert not outcome.mirrored


def test_the_cli_fails_on_a_failed_gate_or_a_difference(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = read_results(latest_results(REPO_ROOT / "eval" / "results"))
    failing = results.model_copy(update={"gate": GateOutcome(passed=False, failures=("production: broke t001",))})
    outcomes = iter(
        [
            flows.EvaluationOutcome(failing, Path("r.json"), (), True),
            flows.EvaluationOutcome(results, None, ("C1_bm25: numbers differ",), False),
        ]
    )
    monkeypatch.setattr(cli, "evaluate_flow", lambda split, final, check: next(outcomes))

    assert cli.main(["evaluate", "--split", "test", "--final"]) == 1
    assert cli.main(["evaluate", "--check"]) == 1
    out = capsys.readouterr().out
    assert "gate: FAIL production: broke t001" in out
    assert "results: r.json (mirrored to Langfuse)" in out
    assert "differs: C1_bm25: numbers differ" in out
    assert "results differ from the committed" in out
