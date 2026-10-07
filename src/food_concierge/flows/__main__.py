"""Command line for the offline build.

python -m food_concierge.flows build          # catalog, then indexes; does nothing when both are current
python -m food_concierge.flows ingest         # catalog only
python -m food_concierge.flows index          # indexes only
python -m food_concierge.flows build --force  # rebuild even when current
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from food_concierge.flows.pipeline import build_flow, build_index_flow, ingest_flow
from food_concierge.ingestion.build import StepResult


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m food_concierge.flows", description=__doc__)
    parser.add_argument("command", choices=["build", "ingest", "index"])
    parser.add_argument("--force", action="store_true", help="rebuild even when the outputs are current")
    args = parser.parse_args(argv)

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
