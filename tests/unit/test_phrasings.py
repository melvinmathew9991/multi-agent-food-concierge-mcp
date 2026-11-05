"""The phrasing set (#142): the reader reads every phrasing as written, and the scorer counts misses and over-reads."""

import runpy

import pytest

from food_concierge.config import REPO_ROOT
from food_concierge.evaluation.phrasings import Phrasing, load_phrasings, render, score_phrasings
from food_concierge.ingestion.taxonomy import Allergen, Diet

A = Allergen
DATASET = REPO_ROOT / "eval" / "datasets" / "query_constraints.yaml"


def test_every_phrasing_reads_as_written() -> None:
    phrasings = load_phrasings(DATASET).phrasings
    report = score_phrasings(phrasings)

    assert len(phrasings) >= 150
    assert len({p.text.lower() for p in phrasings}) == len(phrasings)
    assert sum(not (p.allergens or p.diet or p.unsupported) for p in phrasings) >= 30  # wishes that must add nothing
    assert (report.recall, report.missed, report.over_read) == (1.0, (), ())


@pytest.mark.parametrize(
    ("phrasing", "missed", "over"),
    [
        (Phrasing(text="no peanuts", allergens=frozenset({A.PEANUTS})), False, False),
        (Phrasing(text="peanut chikki", allergens=frozenset({A.PEANUTS})), True, False),
        (Phrasing(text="no meat", diet=Diet.VEGAN), True, True),
        (Phrasing(text="no onion"), False, True),
        (Phrasing(text="hello", unsupported=("onion",)), True, False),
        (Phrasing(text="no milk"), False, True),
        (Phrasing(text="vegan curry"), False, True),
    ],
)
def test_misses_and_over_reads_are_counted(phrasing: Phrasing, missed: bool, over: bool) -> None:
    report = score_phrasings([phrasing])

    assert (bool(report.missed), bool(report.over_read)) == (missed, over)


def test_recall_counts_implied_allergens_and_the_report_lists_each_phrasing() -> None:
    report = score_phrasings(
        [Phrasing(text="wheat-free roti", allergens=frozenset({A.WHEAT})), Phrasing(text="peanut chikki")]
    )

    assert (report.expected_allergens, report.read_allergens, report.recall) == (2, 2, 1.0)
    assert score_phrasings([]).recall == 1.0
    page = render(score_phrasings([Phrasing(text="no milk", allergens=frozenset({A.EGGS}))]))
    assert "missed: no milk\nover-read: no milk\n" in page
    assert page.startswith("1 phrasings; allergen recall 0/1 (0.000); 1 with something missed, 1 over-read")


def test_the_report_script_passes_on_the_committed_set(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO_ROOT / "scripts" / "phrasing_report.py"), run_name="__main__")

    assert exit_info.value.code == 0
    assert "0 with something missed, 0 over-read" in capsys.readouterr().out
