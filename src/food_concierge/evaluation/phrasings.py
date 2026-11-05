"""How the query reader reads diners' phrasings (#142), from ``eval/datasets/query_constraints.yaml``.

Each phrasing lists what its words should add to a search: the allergens avoided, the diet, and what is avoided or
required that the menu doesn't record. Recall is over expected allergens (with the implied ones, so "wheat" also
expects gluten), like the tagger gate's; an over-read is anything read that wasn't expected, a wish taken as
avoidance or "not vegan" taken as vegan. The set was drafted alongside the reader, so it is in-sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from food_concierge.ingestion.taxonomy import Allergen, Diet, with_implied
from food_concierge.services.constraints import read_query


class Phrasing(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    allergens: frozenset[Allergen] = frozenset()
    diet: Diet | None = None
    unsupported: tuple[str, ...] = ()


class PhrasingSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    phrasings: tuple[Phrasing, ...]


@dataclass(frozen=True)
class PhrasingReport:
    phrasings: int
    expected_allergens: int
    read_allergens: int  # of the expected ones
    missed: tuple[str, ...]  # phrasings where something expected wasn't read
    over_read: tuple[str, ...]  # phrasings where something was read that wasn't expected

    @property
    def recall(self) -> float:
        return self.read_allergens / self.expected_allergens if self.expected_allergens else 1.0


def load_phrasings(path: Path) -> PhrasingSet:
    return PhrasingSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def score_phrasings(phrasings: Sequence[Phrasing]) -> PhrasingReport:
    expected_total = read_total = 0
    missed: list[str] = []
    over: list[str] = []
    for phrasing in phrasings:
        reading = read_query(phrasing.text)
        expected = with_implied(phrasing.allergens)
        read = reading.constraints.excluded
        expected_total += len(expected)
        read_total += len(expected & read)
        diet = reading.constraints.diet
        untracked = set(reading.untracked)
        if (
            expected - read
            or (phrasing.diet is not None and diet != phrasing.diet)
            or set(phrasing.unsupported) - untracked
        ):
            missed.append(phrasing.text)
        if read - expected or (diet is not None and diet != phrasing.diet) or untracked - set(phrasing.unsupported):
            over.append(phrasing.text)
    return PhrasingReport(len(phrasings), expected_total, read_total, tuple(missed), tuple(over))


def render(report: PhrasingReport) -> str:
    lines = [
        f"{report.phrasings} phrasings; allergen recall {report.read_allergens}/{report.expected_allergens} "
        f"({report.recall:.3f}); {len(report.missed)} with something missed, {len(report.over_read)} over-read",
        *(f"missed: {text}" for text in report.missed),
        *(f"over-read: {text}" for text in report.over_read),
    ]
    return "\n".join(lines) + "\n"
