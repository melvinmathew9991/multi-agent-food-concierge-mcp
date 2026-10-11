"""Evaluate every candidate from the caches and record the results (Phase 3 plan §3, §4 and §7).

One run scores C0 (from its cached verdicts) and C1-C5 (from the retrieval caches), each C1-C5 with and without
``dish_key`` de-duplication, plus the gated configurations; summarises every metric with a bootstrap interval,
overall and per slice; compares the ladder steps on nDCG@3 with a paired bootstrap; and checks the gate.

Everything is deterministic: the same commit, caches and seed give the same numbers, so ``differences`` against
the committed results is the reproduction check. Timings are not recorded here: cached models look scores up
instead of running, so latency is measured with the real models for ADR-0001.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from food_concierge.errors import EvaluationError
from food_concierge.evaluation.bootstrap import Interval, bootstrap_mean
from food_concierge.evaluation.caches import EvalCaches, texts_sha256
from food_concierge.evaluation.dataset import RetrievalDataset, Split, TextQuery, select
from food_concierge.evaluation.gate import GateBaseline, check_gate
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.original import OriginalRun, missing, replay
from food_concierge.evaluation.record import RunRecord
from food_concierge.evaluation.runner import Candidate, QueryResult, compare, run_candidates
from food_concierge.services.retrieval import RetrievalConfig
from food_concierge.storage.catalog_db import CatalogStore

RunSplit = Literal["dev", "test", "all"]  # "all" before the queries are split
PRIMARY = "ndcg@3"

_BASE: tuple[Candidate, ...] = (
    Candidate(name="C1_bm25", config=RetrievalConfig(mode="bm25")),
    Candidate(name="C2_dense", config=RetrievalConfig(mode="dense")),
    Candidate(name="C2_dense_no_prefix", config=RetrievalConfig(mode="dense"), variant="no_prefix"),
    Candidate(name="C3_qdrant", config=RetrievalConfig(mode="dense", backend="qdrant")),
    Candidate(name="C4_hybrid", config=RetrievalConfig(mode="hybrid")),
    Candidate(name="C5_hybrid_rerank20", config=RetrievalConfig(mode="hybrid", rerank_depth=20)),
)
CANDIDATES: tuple[Candidate, ...] = _BASE + tuple(
    candidate.model_copy(
        update={"name": f"{candidate.name}+dedupe", "config": candidate.config.model_copy(update={"dedupe": True})}
    )
    for candidate in _BASE
)
# (baseline, candidate) pairs the decision rules read: the ladder steps and the query prefix (plan §5).
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("C1_bm25", "C4_hybrid"),
    ("C2_dense", "C4_hybrid"),
    ("C4_hybrid", "C5_hybrid_rerank20"),
    ("C2_dense", "C2_dense_no_prefix"),
)


class Stat(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float
    low: float
    high: float
    n: int  # queries the metric is defined on


class QueryRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    ranked: tuple[str, ...]
    metrics: dict[str, float]


class CandidateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    config: RetrievalConfig | None  # None for C0, which is not a retrieval configuration
    variant: str | None
    summary: dict[str, Stat]
    slices: dict[str, dict[str, Stat]]  # "type:<query type>" or "author:<author>" → metric → stat
    queries: tuple[QueryRow, ...]


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline: str
    candidate: str
    metric: str
    delta: Stat  # candidate - baseline, paired over queries


class GateOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    failures: tuple[str, ...]


class RetrievalResults(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: RunSplit
    queries_sha256: str
    test_sha256: str | None
    record: RunRecord
    candidates: tuple[CandidateResult, ...]
    comparisons: tuple[Comparison, ...]
    gate: GateOutcome


def _stat(values: Sequence[float], seed: int) -> Stat:
    interval = bootstrap_mean(values, seed)
    return Stat(mean=interval.mean, low=interval.low, high=interval.high, n=len(values))


def _summary(rows: Sequence[QueryResult], seed: int) -> dict[str, Stat]:
    names = sorted({name for row in rows for name in row.metrics})
    return {name: _stat([row.metrics[name] for row in rows if name in row.metrics], seed) for name in names}


def _slices(rows: Sequence[QueryResult], queries: Mapping[str, TextQuery], seed: int) -> dict[str, dict[str, Stat]]:
    groups: dict[str, list[QueryResult]] = {}
    for row in rows:
        query = queries[row.query_id]
        for key in (f"type:{query.type}", f"author:{query.author}"):
            groups.setdefault(key, []).append(row)
    return {key: _summary(groups[key], seed) for key in sorted(groups)}


def _result(
    name: str, candidate: Candidate | None, rows: Sequence[QueryResult], queries: Mapping[str, TextQuery], seed: int
) -> CandidateResult:
    return CandidateResult(
        name=name,
        config=candidate.config if candidate else None,
        variant=candidate.variant if candidate else None,
        summary=_summary(rows, seed),
        slices=_slices(rows, queries, seed),
        queries=tuple(QueryRow(query_id=r.query_id, ranked=r.ranked, metrics=r.metrics) for r in rows),
    )


def run_queries(dataset: RetrievalDataset, split: Split, final: bool) -> tuple[RunSplit, tuple[TextQuery, ...]]:
    if split == "test":
        return "test", select(dataset, "test", final=final)
    if all(query.split is None for query in dataset.queries):
        return "all", dataset.queries
    return "dev", select(dataset, "dev")


def evaluate(
    store: CatalogStore,
    truth: GroundTruth,
    dataset: RetrievalDataset,
    caches: EvalCaches,
    verdicts: Mapping[str, OriginalRun],
    baseline: GateBaseline,
    record: RunRecord,
    work_dir: Path,
    *,
    c0_model: str,
    split: Split = "dev",
    final: bool = False,
) -> RetrievalResults:
    meta = store.meta()
    caches.require_current(
        meta["data_sha256"], int(meta["doc_text_version"]), texts_sha256(q.text for q in dataset.queries)
    )
    run_split, queries = run_queries(dataset, split, final)
    uncached = missing(queries, verdicts, c0_model)
    if uncached:
        raise EvaluationError(f"C0 has no cached verdicts for {', '.join(uncached)}; run original_baseline collect.")

    gated = [g.candidate for g in baseline.gated if g.candidate.name not in {c.name for c in CANDIDATES}]
    candidates = [*CANDIDATES, *gated]
    rows = replay(queries, verdicts, truth) | run_candidates(
        store, truth, queries, candidates, caches.embedders, caches.reranker, work_dir
    )
    by_id = {query.id: query for query in queries}
    by_name = {candidate.name: candidate for candidate in candidates}
    seed = record.seeds["eval_seed"]
    results = tuple(_result(name, by_name.get(name), rows[name], by_id, seed) for name in rows)
    comparisons = []
    for base, other in COMPARISONS:
        try:
            delta: Interval = compare(rows[base], rows[other], PRIMARY, seed)
        except EvaluationError:  # nothing graded yet: no query defines nDCG@3
            continue
        shared = sum(1 for row in rows[base] if PRIMARY in row.metrics)
        comparisons.append(
            Comparison(
                baseline=base,
                candidate=other,
                metric=PRIMARY,
                delta=Stat(mean=delta.mean, low=delta.low, high=delta.high, n=shared),
            )
        )
    report = check_gate(rows, baseline)
    return RetrievalResults(
        split=run_split,
        queries_sha256=texts_sha256(q.text for q in queries),
        test_sha256=dataset.test_sha256,
        record=record,
        candidates=results,
        comparisons=tuple(comparisons),
        gate=GateOutcome(passed=report.passed, failures=report.failures),
    )


def results_path(results_dir: Path, results: RetrievalResults) -> Path:
    return results_dir / f"retrieval_{results.record.created_at:%Y-%m-%d}_{results.split}.json"


def write_results(results_dir: Path, results: RetrievalResults) -> Path:
    path = results_path(results_dir, results)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results.model_dump(mode="json"), indent=1, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")
    return path


def read_results(path: Path) -> RetrievalResults:
    return RetrievalResults.model_validate_json(path.read_text(encoding="utf-8"))


def latest_results(results_dir: Path, split: RunSplit | None = None) -> Path:
    """The newest results file, optionally of one split (ISO dates sort as text)."""
    pattern = f"retrieval_*_{split}.json" if split else "retrieval_*.json"
    found = sorted(results_dir.glob(pattern))
    if not found:
        raise EvaluationError(f"No retrieval results in {results_dir.name}; run the evaluate flow first.")
    return found[-1]


def differences(committed: RetrievalResults, fresh: RetrievalResults) -> list[str]:
    """Every number that differs between two runs; the run record (commit, time, machine) is not compared."""
    found = [
        f"{field} differs"
        for field in ("split", "queries_sha256")
        if getattr(committed, field) != getattr(fresh, field)
    ]
    old = {c.name: c for c in committed.candidates}
    new = {c.name: c for c in fresh.candidates}
    found += [f"{name}: only in one run" for name in sorted(old.keys() ^ new.keys())]
    for name in sorted(old.keys() & new.keys()):
        if old[name].summary != new[name].summary or old[name].queries != new[name].queries:
            found.append(f"{name}: numbers differ")
    if committed.comparisons != fresh.comparisons:
        found.append("comparisons differ")
    if committed.gate != fresh.gate:
        found.append("gate outcome differs")
    return found
