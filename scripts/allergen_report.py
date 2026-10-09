"""Print the allergen tagger's report on the catalog: the safety gate, the unverified rate and per-allergen precision.

Offline and deterministic (no model calls): photo descriptions come from the tracked cache, as in the build. Run
after changing the lexicon, the catalog or the descriptions:

    python scripts/allergen_report.py
"""

from __future__ import annotations

import sys

from food_concierge.config import get_settings
from food_concierge.ingestion.allergens import Source, TaggerReport, evaluate, load_lexicon, tag_item
from food_concierge.ingestion.descriptions import DESCRIPTIONS_FILE, read_descriptions, vision_texts
from food_concierge.ingestion.loader import load_catalog


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render(report: TaggerReport) -> str:
    lines = [
        f"Lexicon version {report.lexicon_version}: {report.dishes} dishes, "
        f"{report.unverified} unverified ({report.unverified / report.dishes:.0%}).",
        "Safety gate: " + ("PASS, no true allergen untagged on a verified dish." if not report.missed else "FAIL"),
    ]
    lines += [f"  missed {item_id}: {', '.join(sorted(missed))}" for item_id, missed in sorted(report.missed.items())]
    lines += ["", "| Allergen | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
    for score in report.scores:
        lines.append(
            f"| {score.allergen.value} | {score.true_positives} | {score.false_positives} | {score.false_negatives} "
            f"| {_ratio(score.precision)} | {_ratio(score.recall)} |"
        )
    lines.append("")
    lines.append("Recall here counts tags only; a false negative on an unverified dish still passes the gate.")
    return "\n".join(lines)


def main() -> int:
    lexicon = load_lexicon()
    loaded = load_catalog()
    vision = vision_texts(loaded.catalog.photos, read_descriptions(get_settings().processed_dir / DESCRIPTIONS_FILE))
    tagged = [tag_item(item, lexicon, vision.get(item.item_id)) for item in loaded.catalog.items]
    report = evaluate(tagged, loaded.true_allergens, lexicon.version)
    print(render(report))
    from_vision = [(dish.item_id, t.allergen) for dish in tagged for t in dish.tags if t.source is Source.VISION]
    confirmed = sum(allergen in loaded.true_allergens[item_id] for item_id, allergen in from_vision)
    print(
        f"\nVision source: {len(vision)} dishes described, {len(from_vision)} may_contain tags, "
        f"{confirmed} of them in the ground truth."
    )
    return 1 if report.missed else 0


if __name__ == "__main__":
    sys.exit(main())
