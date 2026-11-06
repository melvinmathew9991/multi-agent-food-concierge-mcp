"""The weak-match calibration set and its scoring (#143): labels, the split, cut-off choice and the report."""

import runpy
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from food_concierge.config import REPO_ROOT, Settings
from food_concierge.evaluation.record import Machine, RunRecord
from food_concierge.evaluation.weak_match import (
    MAX_FALSE_WEAK,
    SIGNALS,
    FeatureRow,
    WeakMatchRun,
    WeakMatchSet,
    WeakQuery,
    calibrate,
    load_weak_match_set,
    measure,
    render,
    report,
    score,
    split_of,
)
from food_concierge.mcp_server.services import Services
from food_concierge.models.embeddings import HashingEmbedder, OverlapReranker
from food_concierge.services.relevance import MenuWords
from food_concierge.services.retrieval import Retriever
from food_concierge.storage.catalog_db import CATALOG_DB, CatalogStore
from food_concierge.storage.indexes import INDEX_DIR, load_indexes

DATASET = REPO_ROOT / "eval" / "datasets" / "weak_match.yaml"
COSINE_RULE, COSINE, MARGIN, CROSS = SIGNALS


def row(query: str = "q", *, on_menu: bool, split: str = "dev", **values: Any) -> FeatureRow:
    fields: dict[str, Any] = {
        "query": query,
        "on_menu": on_menu,
        "group": "named" if on_menu else "near_miss",
        "split": split,
        "returned": 10,
        "top_score": 0.6,
        "margin": 0.1,
        "menu_word": False,
        "cross_encoder": 0.0,
        "served_weak": False,
        "top_dish": "Dish",
    }
    return FeatureRow(**(fields | values))


def a_run(rows: list[FeatureRow], weak_score: float | None = 0.55) -> WeakMatchRun:
    record = RunRecord(
        git_sha="a" * 40,
        lock_sha256="b" * 64,
        data_sha256="c" * 64,
        embedders={},
        reranker=None,
        models={},
        prompt_versions={},
        seeds={"bootstrap": 143},
        machine=Machine(cpu="x", cores=1, os="Linux"),
        created_at=datetime(2026, 11, 6, tzinfo=UTC),
    )
    return WeakMatchRun(
        record=record, dataset_sha256="d" * 64, dataset_version=1, weak_score=weak_score, rows=tuple(rows)
    )


def test_the_committed_set_is_balanced_and_split_by_group() -> None:
    queries = load_weak_match_set(DATASET).queries
    splits = split_of(queries)

    assert len(queries) == 200
    assert sum(q.on_menu for q in queries) == 100
    assert sum(q.group == "near_miss" for q in queries) >= 50  # at least half the off-menu ones are food
    assert all(splits[q.query] == "dev" for q in queries if q.seen)
    for group in {q.group for q in queries}:
        members = [q for q in queries if q.group == group]
        dev = sum(splits[q.query] == "dev" for q in members)
        assert dev == (len(members) + 1) // 2


def test_the_split_ignores_order_and_a_repeated_query_is_refused(tmp_path: Path) -> None:
    queries = [WeakQuery(query=f"dish {i}", on_menu=True, group="named") for i in range(6)]

    assert split_of(queries) == split_of(list(reversed(queries)))
    duplicated = tmp_path / "set.yaml"
    duplicated.write_bytes(
        b"version: 1\nqueries:\n  - {query: a, on_menu: true, group: named}\n"
        b"  - {query: a, on_menu: false, group: near_miss}\n"
    )
    with pytest.raises(ValueError, match="twice"):
        load_weak_match_set(duplicated)


def test_signals_flag_below_their_cut_off_and_the_word_veto_holds() -> None:
    low = row(on_menu=False, top_score=0.5, margin=0.01, cross_encoder=-9.0)

    assert [s.weak(low, s.value(low) + 0.01) for s in SIGNALS] == [True, True, True, True]  # type: ignore[operator]
    assert not COSINE_RULE.weak(low.model_copy(update={"menu_word": True}), 0.6)
    assert COSINE.weak(low.model_copy(update={"menu_word": True}), 0.6)
    assert not CROSS.weak(low.model_copy(update={"cross_encoder": None}), 0.0)  # no reranker, never weak


def test_calibration_takes_the_best_recall_within_the_false_weak_budget() -> None:
    on = [row(f"on{i}", on_menu=True, top_score=0.70 + i / 100) for i in range(19)]
    on.append(row("on-low", on_menu=True, top_score=0.50))  # one on-menu query below every off-menu one
    off = [row(f"off{i}", on_menu=False, top_score=0.55 + i / 100) for i in range(10)]

    cutoff = calibrate(COSINE, on + off)

    assert cutoff == pytest.approx((0.64 + 0.70) / 2)  # all ten off-menu, and the one on-menu: 1/20 = 5%
    assert calibrate(CROSS, [row(on_menu=True, cross_encoder=None)]) is None
    assert MAX_FALSE_WEAK == 0.05


def test_with_no_budget_the_lowest_on_menu_value_bounds_the_cut_off() -> None:
    rows = [row("on", on_menu=True, top_score=0.50), row("off", on_menu=False, top_score=0.60)]

    cutoff = calibrate(COSINE, rows, max_false_weak=0.0)

    assert cutoff is not None
    assert cutoff < 0.50  # flagging the off-menu query would flag the on-menu one too


def test_scores_count_precision_recall_and_false_weak_by_group() -> None:
    rows = [
        row("a", on_menu=False, top_score=0.5),
        row("b", on_menu=False, top_score=0.7),
        row("c", on_menu=True, top_score=0.5),
        row("d", on_menu=True, top_score=0.8),
        row("e", on_menu=True, returned=0, top_score=None, margin=None, cross_encoder=None),  # left out
    ]

    scores = score(COSINE, rows, 0.6)

    assert (scores.flagged, scores.precision) == (2, 0.5)
    assert (scores.recall.mean, scores.false_weak.mean) == (0.5, 0.5)
    assert scores.by_group == {"named": 0.5, "near_miss": 0.5}
    assert score(COSINE, rows, 0.0).precision == 1.0  # nothing flagged


def test_the_report_scores_production_then_each_calibrated_signal() -> None:
    rows = [
        row(f"{s}{i}", on_menu=i % 2 == 0, split=s, top_score=0.5 + i / 50, cross_encoder=float(i))
        for s in ("dev", "test")
        for i in range(10)
    ]
    rows.append(row("empty", on_menu=True, split="test", returned=0, top_score=None, margin=None, cross_encoder=None))

    reports = report(a_run(rows))
    page = render(a_run(rows), reports)

    assert [(r.signal.name, r.calibrated) for r in reports] == [
        ("cosine_no_menu_word", False),
        *((s.name, True) for s in SIGNALS),
    ]
    assert "Left out (the query's own constraints exclude every dish): empty." in page
    assert "| cosine_no_menu_word (production) | 0.550 |" in page
    assert "| named | 11 |" in page
    assert len(report(a_run(rows, weak_score=None))) == len(SIGNALS)
    assert "n/a" in render(a_run([row(on_menu=True, cross_encoder=None)]), [])


@pytest.fixture(scope="module")
def store(built_data_dir: Path) -> Iterator[CatalogStore]:
    opened = CatalogStore(built_data_dir / "processed" / CATALOG_DB)
    yield opened
    opened.close()


def test_measure_runs_each_query_through_the_serving_search(store: CatalogStore, built_data_dir: Path) -> None:
    embedder = HashingEmbedder()
    indexes = load_indexes(store, embedder, built_data_dir / "processed" / INDEX_DIR)
    settings = Settings(_env_file=None)
    search = Services.over(store, Retriever(store, indexes, embedder), settings, weak_score=0.99).search
    texts = store.doc_texts()
    dataset = WeakMatchSet(
        version=1,
        queries=(
            WeakQuery(query="butter chicken", on_menu=True, group="named"),
            WeakQuery(query="asdfghjkl", on_menu=False, group="off_topic", seen=True),
            WeakQuery(
                query="vegan dish without soy, gluten, peanuts or tree nuts, no sesame", on_menu=True, group="vague"
            ),
        ),
    )

    rows = measure(dataset, search, texts, MenuWords(texts.values()), OverlapReranker())
    without = measure(dataset, search, texts, MenuWords(texts.values()), None)

    butter, nonsense, _ = rows
    assert (butter.returned, butter.menu_word, butter.served_weak) == (10, True, False)
    assert butter.top_score is not None
    assert butter.margin is not None
    assert butter.cross_encoder is not None
    assert (nonsense.split, nonsense.menu_word, nonsense.served_weak) == ("dev", False, True)
    assert all(r.cross_encoder is None for r in without)


def test_the_report_script_reads_the_committed_run(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["weak_match_report.py", "report"])  # prints only; --write would save it
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO_ROOT / "scripts" / "weak_match_report.py"), run_name="__main__")

    assert exit_info.value.code == 0
    assert "| cosine_no_menu_word (production) | 0.550 |" in capsys.readouterr().out
