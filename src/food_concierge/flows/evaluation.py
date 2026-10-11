"""The ``evaluate`` and ``report`` flows (Phase 3 plan §7). Offline: committed caches and verdicts, no network.

``evaluate`` checks the caches, runs every candidate, writes ``eval/results/retrieval_<date>_<split>.json``,
checks the gate and mirrors to Langfuse when keys are configured. With ``check`` it writes nothing and compares
with the newest committed results of the same split instead: the reproduction check the nightly runs.

``report`` renders ``docs/evaluation.md`` from the newest results file.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from prefect import flow

from food_concierge.config import REPO_ROOT, get_settings
from food_concierge.evaluation.caches import load_caches
from food_concierge.evaluation.dataset import Split, load_dataset
from food_concierge.evaluation.gate import GateBaseline
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.mirror import mirror_results
from food_concierge.evaluation.original import PROMPT_VERSION, read_runs
from food_concierge.evaluation.record import make_record
from food_concierge.evaluation.report import render_report
from food_concierge.evaluation.results import (
    RetrievalResults,
    differences,
    evaluate,
    latest_results,
    read_results,
    run_queries,
    write_results,
)
from food_concierge.ingestion.build import build_catalog
from food_concierge.ingestion.loader import load_catalog
from food_concierge.storage.catalog_db import CatalogStore
from food_concierge.telemetry import init_telemetry

DATASET = REPO_ROOT / "eval" / "datasets" / "retrieval.yaml"
CACHE_DIR = REPO_ROOT / "eval" / "caches" / "retrieval"
VERDICTS = REPO_ROOT / "eval" / "caches" / "original_verdicts.jsonl"
BASELINE = REPO_ROOT / "eval" / "gates" / "retrieval_baseline.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
REPORT = REPO_ROOT / "docs" / "evaluation.md"


@dataclass(frozen=True)
class EvaluationOutcome:
    results: RetrievalResults
    path: Path | None  # the file written; None for a check
    differences: tuple[str, ...]  # against the committed results, for a check
    mirrored: bool


def run_evaluation(split: Split = "dev", final: bool = False, check: bool = False) -> EvaluationOutcome:
    settings = get_settings()
    store = CatalogStore(build_catalog(settings).output)
    try:
        dataset = load_dataset(DATASET)
        caches = load_caches(CACHE_DIR)
        c0_model = settings.require_model_id("groq", "router")
        record = make_record(
            REPO_ROOT,
            store.meta()["data_sha256"],
            {variant: embedder.fingerprint for variant, embedder in caches.embedders.items()},
            caches.reranker.fingerprint,
            seeds={"eval_seed": settings.eval_seed},
            models={"c0": c0_model},
            prompt_versions={"c0": PROMPT_VERSION},
        )
        baseline = GateBaseline.model_validate_json(BASELINE.read_text(encoding="utf-8"))
        truth = GroundTruth.from_load(load_catalog(settings.raw_dir))
        with tempfile.TemporaryDirectory() as work_dir:
            results = evaluate(
                store,
                truth,
                dataset,
                caches,
                read_runs(VERDICTS),
                baseline,
                record,
                Path(work_dir),
                c0_model=c0_model,
                split=split,
                final=final,
            )
    finally:
        store.close()
    if check:
        committed = read_results(latest_results(RESULTS_DIR, results.split))
        return EvaluationOutcome(results, None, tuple(differences(committed, results)), False)
    path = write_results(RESULTS_DIR, results)
    _, queries = run_queries(dataset, split, final)
    mirrored = mirror_results(init_telemetry(settings).client, results, queries)
    return EvaluationOutcome(results, path, (), mirrored)


def run_report() -> Path:
    REPORT.write_text(render_report(read_results(latest_results(RESULTS_DIR))), encoding="utf-8", newline="\n")
    return REPORT


@flow(name="evaluate", persist_result=False)
def evaluate_flow(split: Split = "dev", final: bool = False, check: bool = False) -> EvaluationOutcome:
    """Run every retrieval candidate from the committed caches, record the results and check the gate."""
    return run_evaluation(split, final, check)


@flow(name="report", persist_result=False)
def report_flow() -> Path:
    """Render docs/evaluation.md from the newest results."""
    return run_report()
