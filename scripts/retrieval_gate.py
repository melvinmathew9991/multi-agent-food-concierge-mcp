"""The PR retrieval gate (plan §6): the gated configurations over the dev split, from the committed caches only.

No model download and no network. Runs in CI after the tests; exits 1 on any failure:

    python scripts/retrieval_gate.py

Move the baseline to what is measured now, in a PR that says why (refused while any dish breaks a filter):

    python scripts/retrieval_gate.py --update --reason "owner grades imported"
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from food_concierge.config import REPO_ROOT, get_settings
from food_concierge.evaluation.caches import load_caches
from food_concierge.evaluation.dataset import load_dataset
from food_concierge.evaluation.gate import GateBaseline, run_gate, updated_baseline
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.ingestion.build import build_catalog
from food_concierge.ingestion.loader import load_catalog
from food_concierge.storage.catalog_db import CatalogStore

DATASET = REPO_ROOT / "eval" / "datasets" / "retrieval.yaml"
CACHE_DIR = REPO_ROOT / "eval" / "caches" / "retrieval"
BASELINE = REPO_ROOT / "eval" / "gates" / "retrieval_baseline.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="move the baseline to the measured numbers")
    parser.add_argument("--reason", default="", help="why the baseline changes (required with --update)")
    args = parser.parse_args(argv)
    if args.update and not args.reason.strip():
        parser.error("--update needs --reason")

    settings = get_settings()
    store = CatalogStore(build_catalog(settings).output)  # offline: the catalog files and the lexicon only
    baseline = GateBaseline.model_validate_json(BASELINE.read_text(encoding="utf-8"))
    truth = GroundTruth.from_load(load_catalog(settings.raw_dir))
    with tempfile.TemporaryDirectory() as work_dir:
        report = run_gate(load_caches(CACHE_DIR), store, truth, load_dataset(DATASET), baseline, Path(work_dir))

    for name, ndcg in report.ndcg_at_3.items():
        print(f"{name}: nDCG@3 {'not graded yet' if ndcg is None else f'{ndcg:.4f}'}")
        found = report.targeted - len(report.target_misses.get(name, ()))
        print(f"{name}: target dish in the top 3 on {found} of {report.targeted} queries")
    for failure in report.failures:
        print(f"FAIL {failure}")
    if args.update:
        if any("break the filters" in failure for failure in report.failures):
            raise SystemExit("not updating the baseline while a dish breaks a filter")
        updated = updated_baseline(baseline, report, args.reason.strip())
        BASELINE.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"baseline updated: {BASELINE.relative_to(REPO_ROOT)}")
        return 0
    if not report.passed:
        return 1
    print("retrieval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
