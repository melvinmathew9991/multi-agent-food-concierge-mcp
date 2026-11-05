"""Run candidate retrieval configurations over a split, score each query, and summarise with intervals.

Each embedder variant gets its own index, built in a scratch directory from the same catalog: the query prefix
is part of the embedder fingerprint, so an index refuses queries embedded under another prefix.

Every query goes through ``SearchService``, the path the MCP tools and the API serve (#140): constraints read from
the query text, the safety re-check and the weak-match signal are all part of what is measured, so a change to any
of them shows up here and in the PR gate. The retriever under it is recorded, for its timings and raw top score.

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

from food_concierge.config import get_settings
from food_concierge.errors import EvaluationError
from food_concierge.evaluation.bootstrap import Interval, bootstrap_mean, paired_bootstrap
from food_concierge.evaluation.dataset import TARGETED_TYPES, TextQuery
from food_concierge.evaluation.metrics import (
    GroundTruth,
    hit_at_k,
    judged_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
    violation_at_k,
)
from food_concierge.models.embeddings import Embedder, Reranker
from food_concierge.services.catalog import CatalogService
from food_concierge.services.constraints import Constraints
from food_concierge.services.relevance import WEAK_SCORES, MenuWords
from food_concierge.services.retrieval import RetrievalConfig, Retriever, SearchResult
from food_concierge.services.search import MAX_RESULTS, SearchService
from food_concierge.storage.catalog_db import CatalogStore, Filters
from food_concierge.storage.indexes import build_indexes, load_indexes

DEPTH = MAX_RESULTS  # results kept per query (10), for Recall@10 and MRR; the @3 metrics read the top 3 of these


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
    top_score: float | None = None  # the best hit's score; None when nothing was found
    weak_match: bool = False  # search told the client even the best dish is a weak match
    # Dishes the SQL filter let through and the safety re-check dropped: a disagreement between the two, a bug.
    dropped: tuple[str, ...] = ()


class _Recorded:
    """The retriever under a ``SearchService``, keeping its results for the timings and the raw top score."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.runs: list[SearchResult] = []

    def search(self, query: str, filters: Filters, config: RetrievalConfig) -> SearchResult:
        self.runs.append(self._retriever.search(query, filters, config))
        return self.runs[-1]


def score_query(query: TextQuery, ranked: Sequence[str], truth: GroundTruth) -> dict[str, float]:
    """Every metric that applies to this query; a metric a query cannot define is left out, not zeroed."""
    metrics = {"violation@3": violation_at_k(ranked, query.filters, truth)}
    if any(grade > 0 for grade in query.grades.values()):
        metrics |= {
            "ndcg@3": ndcg_at_k(ranked, query.grades),
            "success@3": success_at_k(ranked, query.grades),
            "judged@3": judged_at_k(ranked, query.grades),
        }
    # A target dish's items are exact by rule, as the labelling page grades them, so these need no grades (#141).
    exact = query.exact | truth.dish_items(query.target)
    if query.type in TARGETED_TYPES and exact:
        metrics |= {
            "hit@1": hit_at_k(ranked, exact, 1),
            "hit@3": hit_at_k(ranked, exact, 3),
            "recall@3": recall_at_k(ranked, exact, 3),
            "recall@10": recall_at_k(ranked, exact, 10),
            "mrr": reciprocal_rank(ranked, exact),
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

    catalog = CatalogService(store)
    words = MenuWords(store.doc_texts().values())
    settings = get_settings()
    results: dict[str, list[QueryResult]] = {}
    for candidate in candidates:
        recorded = _Recorded(retrievers[candidate.variant])
        weak_score = WEAK_SCORES.get(embedders[candidate.variant].fingerprint.model)
        service = SearchService(catalog, recorded, candidate.config, settings, words, weak_score)
        rows = []
        for query in queries:
            constraints = Constraints(**query.filters.model_dump())
            recorded.runs.clear()
            found = [service.search(query.text, constraints, DEPTH) for _ in range(repeats)][-1]
            ranked = tuple(dish.item_id for dish in found.dishes)
            raw = [hit.item_id for hit in recorded.runs[-1].hits][:DEPTH]
            rows.append(
                QueryResult(
                    query.id,
                    query.type,
                    ranked,
                    score_query(query, ranked, truth),
                    tuple(run.timings_ms for run in recorded.runs),
                    recorded.runs[-1].hits[0].score if ranked else None,
                    found.weak_match,
                    tuple(item_id for item_id in raw if item_id not in ranked),
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
