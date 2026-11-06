"""Weak-match signals on held-out queries (#143).

python scripts/weak_match_report.py measure   # search every query with the real models; writes the run to eval/results
python scripts/weak_match_report.py report    # calibrate on dev, score on test, from the latest run; no models needed

`measure` needs the built catalog and indexes (`python -m food_concierge.flows build`) and the local embedding and
reranker models, and saves the run and its table. `report` reads only the committed run; `--write` saves the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from food_concierge.config import REPO_ROOT, Settings
from food_concierge.evaluation.record import make_record
from food_concierge.evaluation.weak_match import SEED, WeakMatchRun, load_weak_match_set, measure, render, report
from food_concierge.mcp_server.services import build_services
from food_concierge.models.embeddings import get_reranker
from food_concierge.services.relevance import WEAK_SCORES, MenuWords
from food_concierge.storage.indexes import INDEX_DIR, MANIFEST

DATASET = REPO_ROOT / "eval" / "datasets" / "weak_match.yaml"
RESULTS = REPO_ROOT / "eval" / "results"
PREFIX = "weak_match_"


def run_measure(settings: Settings) -> Path:
    dataset = load_weak_match_set(DATASET)
    services = build_services(settings)
    reranker = get_reranker(settings)
    texts = services.store.doc_texts()
    rows = measure(dataset, services.search, texts, MenuWords(texts.values()), reranker)
    manifest = json.loads((settings.processed_dir / INDEX_DIR / MANIFEST).read_text(encoding="utf-8"))
    embedder = manifest["embedder"]
    run = WeakMatchRun(
        record=make_record(
            REPO_ROOT, manifest["data_sha256"], {"dense": embedder}, reranker.fingerprint, {"bootstrap": SEED}
        ),
        dataset_sha256=hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        dataset_version=dataset.version,
        weak_score=WEAK_SCORES.get(embedder["model"]),
        rows=tuple(rows),
    )
    path = RESULTS / f"{PREFIX}{datetime.now(UTC):%Y-%m-%d}.json"
    path.write_bytes((run.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return path


def latest() -> Path:
    runs = sorted(RESULTS.glob(f"{PREFIX}*.json"))
    if not runs:
        raise SystemExit("No weak-match run in eval/results; run `measure` first.")
    return runs[-1]


def run_report(write: bool) -> None:
    path = latest()
    run = WeakMatchRun.model_validate_json(path.read_bytes())
    if run.dataset_sha256 != hashlib.sha256(DATASET.read_bytes()).hexdigest():
        raise SystemExit(f"{path.name} was measured on another version of the dataset; run `measure` again.")
    text = render(run, report(run))
    if write:
        path.with_suffix(".md").write_bytes(text.encode("utf-8"))
    sys.stdout.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weak-match signals on held-out queries (#143).")
    parser.add_argument("command", choices=["measure", "report"])
    parser.add_argument("--write", action="store_true", help="save the table next to the run (`measure` always does)")
    args = parser.parse_args(argv)
    if args.command == "measure":
        sys.stdout.write(f"wrote {run_measure(Settings()).relative_to(REPO_ROOT)}\n")
    run_report(write=args.write or args.command == "measure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
