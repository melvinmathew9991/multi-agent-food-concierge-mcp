"""Run candidate retrieval configurations over a split, score each query, and summarise with intervals.

Each embedder variant gets its own index, built in a scratch directory from the same catalog: the query prefix
is part of the embedder fingerprint, so an index refuses queries embedded under another prefix.

Quality metrics are the same with cached or real models, but timings are not: a cached embedder or reranker
looks scores up instead of running the model. Latency for the decision rules is measured with the real models.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from food_concierge.errors import EvaluationError
from food_concierge.evaluation.bootstrap import Interval, bootstrap_mean, paired_bootstrap
from food_concierge.evaluation.dataset import TARGETED_TYPES, TextQuery
from food_concierge.evaluation.metrics import (
    GroundTruth,
    judged_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
    violation_at_k,
)
from food_concierge.models.embeddings import Embedder, Reranker
from food_concierge.services.retrieval import RetrievalConfig, Retriever
from food_concierge.storage.catalog_db import CatalogStore
from food_concierge.storage.indexes import build_indexes, load_indexes

DEPTH = 10  # results kept per query, for Recall@10 and MRR; the @3 metrics read the top 3 of these


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    config: RetrievalConfig
    variant: str = "prefix"  # which cached query embedding to use


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    query_type: str
    ranked: tuple[str, ...]
    metrics: dict[str, float]
    timings_ms: tuple[dict[str, float], ...]  # one per measured repeat


def score_query(query: TextQuery, ranked: Sequence[str], truth: GroundTruth) -> dict[str, float]:
    """Every metric that applies to this query; a metric a query cannot define is left out, not zeroed."""
    metrics = {"violation@3": violation_at_k(ranked, query.filters, truth)}
    if any(grade > 0 for grade in query.grades.values()):
        metrics |= {
            "ndcg@3": ndcg_at_k(ranked, query.grades),
            "success@3": success_at_k(ranked, query.grades),
            "judged@3": judged_at_k(ranked, query.grades),
        }
    if query.type in TARGETED_TYPES and query.exact:
        metrics |= {
            "recall@3": recall_at_k(ranked, query.exact, 3),
            "recall@10": recall_at_k(ranked, query.exact, 10),
            "mrr": reciprocal_rank(ranked, query.exact),
        }
    return metrics


def run_candidates(
    store: CatalogStore,
    truth: GroundTruth,
    queries: Sequence[TextQuery],
    candidates: Sequence[Candidate],
    embedders: Mapping[str, Embedder],
    reranker: Reranker | None,
    work_dir: Path,
    repeats: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, list[QueryResult]]:
    retrievers: dict[str, Retriever] = {}
    for variant in sorted({candidate.variant for candidate in candidates}):
        if variant not in embedders:
            raise EvaluationError(f"No cached embeddings for the query variant '{variant}'.")
        index_dir = work_dir / f"index_{variant}"
        build_indexes(store, embedders[variant], index_dir)
        indexes = load_indexes(store, embedders[variant], index_dir)
        retrievers[variant] = Retriever(store, indexes, embedders[variant], reranker, clock)

    results: dict[str, list[QueryResult]] = {}
    for candidate in candidates:
        retriever = retrievers[candidate.variant]
        config = candidate.config.model_copy(update={"k": max(candidate.config.k, DEPTH)})
        rows = []
        for query in queries:
            runs = [retriever.search(query.text, query.filters, config) for _ in range(repeats)]
            ranked = tuple(hit.item_id for hit in runs[-1].hits)
            rows.append(
                QueryResult(
                    query.id,
                    query.type,
                    ranked,
                    score_query(query, ranked, truth),
                    tuple(run.timings_ms for run in runs),
                )
            )
        results[candidate.name] = rows
    return results


def summarize(rows: Sequence[QueryResult], seed: int) -> dict[str, Interval]:
    """Mean and bootstrap interval of each metric, over the queries that define it."""
    names = sorted({name for row in rows for name in row.metrics})
    return {name: bootstrap_mean([row.metrics[name] for row in rows if name in row.metrics], seed) for name in names}


def compare(baseline: Sequence[QueryResult], candidate: Sequence[QueryResult], metric: str, seed: int) -> Interval:
    """Paired bootstrap of ``candidate - baseline`` on one metric, over queries scored by both."""
    base = {row.query_id: row.metrics[metric] for row in baseline if metric in row.metrics}
    other = {row.query_id: row.metrics[metric] for row in candidate if metric in row.metrics}
    shared = sorted(base.keys() & other.keys())
    if not shared:
        raise EvaluationError(f"No query is scored on {metric} by both configurations.")
    return paired_bootstrap([base[q] for q in shared], [other[q] for q in shared], seed)


def latency(rows: Sequence[QueryResult]) -> dict[str, dict[str, float]]:
    """p50 and p95 per stage over every measured search."""
    samples: dict[str, list[float]] = {}
    for row in rows:
        for timings in row.timings_ms:
            for stage, value in timings.items():
                samples.setdefault(stage, []).append(value)
    return {
        stage: {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
        for stage, values in samples.items()
    }
