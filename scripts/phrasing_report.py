"""How the query reader reads the phrasing set (#142); exits 1 on any miss or over-read.

python scripts/phrasing_report.py
"""

from __future__ import annotations

import sys

from food_concierge.config import REPO_ROOT
from food_concierge.evaluation.phrasings import load_phrasings, render, score_phrasings

DATASET = REPO_ROOT / "eval" / "datasets" / "query_constraints.yaml"


def main() -> int:
    report = score_phrasings(load_phrasings(DATASET).phrasings)
    sys.stdout.write(render(report))
    return 1 if report.missed or report.over_read else 0


if __name__ == "__main__":
    sys.exit(main())
