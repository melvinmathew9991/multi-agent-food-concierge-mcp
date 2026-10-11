"""The PR retrieval gate (Phase 3 plan §6): the gated configurations over the dev split, from the committed caches.

Deterministic, with no model download: any change in a number comes from a code or data change, never noise. The
gate fails when

- a gated configuration returns a dish that breaks a query's filters by the ground truth (any Violation@3);
- its nDCG@3 falls more than the tolerance below ``eval/gates/retrieval_baseline.json``;
- the caches are stale (the field is named) or miss a query or item (``EvaluationError``).

nDCG@3 is gated once the owner's grades exist; until then the baseline holds no number and only violations gate.
The baseline changes only in a PR that says why (``reason``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from food_concierge.evaluation.caches import EvalCaches, texts_sha256
from food_concierge.evaluation.dataset import RetrievalDataset, dev_or_all
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.runner import Candidate, QueryResult, run_candidates
from food_concierge.storage.catalog_db import CatalogStore


class GatedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: Candidate
    ndcg_at_3: float | None = None  # the baseline mean over the gated queries; None until grades exist


class GateBaseline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(min_length=1)  # why the baseline last changed
    tolerance: float = Field(default=0.02, ge=0)
    gated: tuple[GatedCandidate, ...] = Field(min_length=1)


@dataclass(frozen=True)
class GateReport:
    ndcg_at_3: dict[str, float | None]  # measured mean per gated configuration; None when no query is graded
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def _mean_ndcg(rows: Sequence[QueryResult]) -> float | None:
    values = [row.metrics["ndcg@3"] for row in rows if "ndcg@3" in row.metrics]
    return sum(values) / len(values) if values else None


def check_gate(results: Mapping[str, Sequence[QueryResult]], baseline: GateBaseline) -> GateReport:
    measured: dict[str, float | None] = {}
    failures = []
    for gated in baseline.gated:
        name = gated.candidate.name
        rows = results[name]
        violating = [row.query_id for row in rows if row.metrics["violation@3"] > 0]
        if violating:
            failures.append(f"{name}: dishes that break the filters on {', '.join(violating)}")
        measured[name] = ndcg = _mean_ndcg(rows)
        if gated.ndcg_at_3 is None:
            continue
        if ndcg is None:
            failures.append(f"{name}: no graded query to compare with the baseline nDCG@3")
        elif ndcg < gated.ndcg_at_3 - baseline.tolerance:
            failures.append(
                f"{name}: nDCG@3 {ndcg:.4f} is more than {baseline.tolerance} below the baseline {gated.ndcg_at_3:.4f}"
            )
    return GateReport(measured, tuple(failures))


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
    results = run_candidates(store, truth, dev_or_all(dataset), candidates, caches.embedders, caches.reranker, work_dir)
    return check_gate(results, baseline)


def updated_baseline(baseline: GateBaseline, report: GateReport, reason: str) -> GateBaseline:
    """The baseline moved to what was just measured, with the reason recorded."""
    gated = tuple(
        gated.model_copy(update={"ndcg_at_3": report.ndcg_at_3[gated.candidate.name]}) for gated in baseline.gated
    )
    return baseline.model_copy(update={"gated": gated, "reason": reason})
