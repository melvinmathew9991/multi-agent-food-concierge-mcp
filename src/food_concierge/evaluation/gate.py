"""The PR retrieval gate (Phase 3 plan §6): the gated configurations over the dev split, from the committed caches.

Deterministic, with no model download: any change in a number comes from a code or data change, never noise. The
gate fails when

- a gated configuration returns a dish that breaks a query's filters by the ground truth (any Violation@3);
- its nDCG@3 falls more than the tolerance below ``eval/gates/retrieval_baseline.json``;
- a known-item or described query's target dish leaves the top 3, unless the baseline accepts that miss
  (``target_misses``): a check that needs no grades, so gross regressions fail before grading exists;
- search would call a dev query's best match weak (``services/relevance``), so the weak-match cut-off stays below
  every real query;
- the caches are stale (the field is named) or miss a query or item (``EvaluationError``).

nDCG@3 is gated once the owner's grades exist; until then the baseline holds no number and only violations gate.
The baseline changes only in a PR that says why (``reason``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from food_concierge.evaluation.caches import EvalCaches, texts_sha256
from food_concierge.evaluation.dataset import RetrievalDataset, dev_or_all
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.runner import Candidate, QueryResult, run_candidates
from food_concierge.services.relevance import WEAK_SCORES, MenuWords, is_weak
from food_concierge.storage.catalog_db import CatalogStore


class GatedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: Candidate
    ndcg_at_3: float | None = None  # the baseline mean over the gated queries; None until grades exist
    target_misses: tuple[str, ...] = ()  # queries accepted as missing their target dish in the top 3


class GateBaseline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(min_length=1)  # why the baseline last changed
    tolerance: float = Field(default=0.02, ge=0)
    gated: tuple[GatedCandidate, ...] = Field(min_length=1)


@dataclass(frozen=True)
class GateReport:
    ndcg_at_3: dict[str, float | None]  # measured mean per gated configuration; None when no query is graded
    failures: tuple[str, ...]
    target_misses: dict[str, tuple[str, ...]] = field(default_factory=dict)  # per configuration
    targeted: int = 0  # queries naming a target dish

    @property
    def passed(self) -> bool:
        return not self.failures


def _mean_ndcg(rows: Sequence[QueryResult]) -> float | None:
    values = [row.metrics["ndcg@3"] for row in rows if "ndcg@3" in row.metrics]
    return sum(values) / len(values) if values else None


def check_gate(
    results: Mapping[str, Sequence[QueryResult]],
    baseline: GateBaseline,
    targets: Mapping[str, frozenset[str]] | None = None,
    weak: Mapping[str, Sequence[str]] | None = None,
) -> GateReport:
    """``targets`` maps a query to its target dish's items; ``weak`` a configuration to the queries it calls weak."""
    targets = targets or {}
    measured: dict[str, float | None] = {}
    misses: dict[str, tuple[str, ...]] = {}
    failures = []
    for gated in baseline.gated:
        name = gated.candidate.name
        rows = results[name]
        violating = [row.query_id for row in rows if row.metrics["violation@3"] > 0]
        if violating:
            failures.append(f"{name}: dishes that break the filters on {', '.join(violating)}")
        misses[name] = tuple(
            row.query_id for row in rows if row.query_id in targets and not set(row.ranked[:3]) & targets[row.query_id]
        )
        new_misses = [query_id for query_id in misses[name] if query_id not in gated.target_misses]
        if new_misses:
            failures.append(f"{name}: the target dish is not in the top 3 on {', '.join(new_misses)}")
        if weak and weak.get(name):
            failures.append(f"{name}: real queries would be called weak matches: {', '.join(weak[name])}")
        measured[name] = ndcg = _mean_ndcg(rows)
        if gated.ndcg_at_3 is None:
            continue
        if ndcg is None:
            failures.append(f"{name}: no graded query to compare with the baseline nDCG@3")
        elif ndcg < gated.ndcg_at_3 - baseline.tolerance:
            failures.append(
                f"{name}: nDCG@3 {ndcg:.4f} is more than {baseline.tolerance} below the baseline {gated.ndcg_at_3:.4f}"
            )
    return GateReport(measured, tuple(failures), misses, len(targets))


def run_gate(
    caches: EvalCaches,
    store: CatalogStore,
    truth: GroundTruth,
    dataset: RetrievalDataset,
    baseline: GateBaseline,
    work_dir: Path,
) -> GateReport:
    """Check the caches are current, run every gated configuration over the dev split and compare."""
    meta = store.meta()
    caches.require_current(
        meta["data_sha256"], int(meta["doc_text_version"]), texts_sha256(q.text for q in dataset.queries)
    )
    candidates = [gated.candidate for gated in baseline.gated]
    queries = dev_or_all(dataset)
    results = run_candidates(store, truth, queries, candidates, caches.embedders, caches.reranker, work_dir)
    keys = store.dish_keys()
    targets = {
        query.id: frozenset(item for item, key in keys.items() if key == query.target)
        for query in queries
        if query.target is not None
    }
    words = MenuWords(store.doc_texts().values())
    texts = {query.id: query.text for query in queries}
    weak: dict[str, list[str]] = {}
    for candidate in candidates:
        cutoff = WEAK_SCORES.get(caches.embedders[candidate.variant].fingerprint.model)
        if candidate.config.mode != "dense" or cutoff is None:
            continue  # search reports weak matches only for dense search with a measured model, as here
        weak[candidate.name] = [
            row.query_id
            for row in results[candidate.name]
            if row.top_score is not None and is_weak(texts[row.query_id], row.top_score, cutoff, words)
        ]
    return check_gate(results, baseline, targets, weak)


def updated_baseline(baseline: GateBaseline, report: GateReport, reason: str) -> GateBaseline:
    """The baseline moved to what was just measured, with the reason recorded."""
    gated = tuple(
        gated.model_copy(
            update={
                "ndcg_at_3": report.ndcg_at_3[gated.candidate.name],
                "target_misses": report.target_misses.get(gated.candidate.name, ()),
            }
        )
        for gated in baseline.gated
    )
    return baseline.model_copy(update={"gated": gated, "reason": reason})
