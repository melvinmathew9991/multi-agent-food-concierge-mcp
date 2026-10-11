"""Command line for the offline build and the retrieval evaluation.

python -m food_concierge.flows build          # catalog, then indexes; does nothing when both are current
python -m food_concierge.flows ingest         # catalog only
python -m food_concierge.flows index          # indexes only
python -m food_concierge.flows build --force  # rebuild even when current
python -m food_concierge.flows evaluate       # every candidate from the committed caches; writes eval/results
python -m food_concierge.flows evaluate --check   # compare with the committed results instead of writing
python -m food_concierge.flows evaluate --split test --final   # the one test-split run, after ADR-0001
python -m food_concierge.flows report         # docs/evaluation.md from the newest results
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from food_concierge.flows.evaluation import evaluate_flow, report_flow
from food_concierge.flows.pipeline import build_flow, build_index_flow, ingest_flow
from food_concierge.ingestion.build import StepResult


def _evaluate(split: str, final: bool, check: bool) -> int:
    outcome = evaluate_flow("test" if split == "test" else "dev", final, check)
    for failure in outcome.results.gate.failures:
        print(f"gate: FAIL {failure}")
    if check:
        for difference in outcome.differences:
            print(f"differs: {difference}")
        print("reproduced the committed results" if not outcome.differences else "results differ from the committed")
    else:
        print(f"results: {outcome.path}" + (" (mirrored to Langfuse)" if outcome.mirrored else ""))
    return 0 if outcome.results.gate.passed and not outcome.differences else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m food_concierge.flows", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["build", "ingest", "index", "evaluate", "report"])
    parser.add_argument("--force", action="store_true", help="rebuild even when the outputs are current")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="evaluate: which split")
    parser.add_argument("--final", action="store_true", help="evaluate: allow the locked test split (once)")
    parser.add_argument("--check", action="store_true", help="evaluate: compare with the committed results")
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        return _evaluate(args.split, args.final, args.check)
    if args.command == "report":
        print(f"report: {report_flow()}")
        return 0
    results: list[StepResult]
    if args.command == "build":
        results = build_flow(args.force)
    elif args.command == "ingest":
        results = [ingest_flow(args.force)]
    else:
        results = [build_index_flow(args.force)]
    for result in results:
        print(f"{result.step}: {'rebuilt' if result.rebuilt else 'up to date'} ({result.output})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
