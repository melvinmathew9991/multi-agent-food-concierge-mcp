"""Allergen tags from three sources (audit A1, A2; PRD F5).

- **label**: what the restaurant declares → ``contains``.
- **lexicon**: ingredient terms from the versioned ``data/lexicon/allergens.yaml`` → ``contains``.
- **vision**: allergen terms in the photo's description → ``may_contain`` (a photo shows, it doesn't prove).

A dish listing an opaque ingredient (a masala mix, an unnamed sauce, a chutney) is ``unverified``: its recipe can't
be checked, so the agent must ask before recommending it to someone avoiding an allergen (PRD F8).

The safety property, checked against the hand-reviewed ground truth by the tests, is recall: every true allergen of
every dish is tagged, or the dish is unverified. Precision is reported, not gated.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from food_concierge.config import get_settings
from food_concierge.errors import ConfigError
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.ingestion.taxonomy import Allergen, with_implied

LEXICON_PATH = Path("lexicon") / "allergens.yaml"  # under Settings.data_dir


class Source(StrEnum):
    LABEL = "label"
    LEXICON = "lexicon"
    VISION = "vision"


class Level(StrEnum):
    CONTAINS = "contains"
    MAY_CONTAIN = "may_contain"


_TERM = re.compile(r"[a-z][a-z' -]*[a-z]")


def _term(value: str) -> str:
    if not _TERM.fullmatch(value):
        raise ValueError(f"{value!r}: terms are lower-case words separated by single spaces or hyphens")
    return value


Term = Annotated[str, AfterValidator(_term)]


def _duplicates(terms: Iterable[str]) -> list[str]:
    return sorted(term for term, count in Counter(terms).items() if count > 1)


class _LexiconFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    allergens: dict[Allergen, list[Term]]
    neutral: list[Term]
    opaque: list[Term]

    @model_validator(mode="after")
    def _consistent(self) -> _LexiconFile:
        lists = [(key.value, terms) for key, terms in self.allergens.items()]
        lists += [("neutral", self.neutral), ("opaque", self.opaque)]
        problems = [f"{name} lists {', '.join(dup)} twice" for name, terms in lists if (dup := _duplicates(terms))]
        positive = {term for terms in self.allergens.values() for term in terms}
        clash = sorted(set(self.neutral) & (positive | set(self.opaque)))
        if clash:
            problems.append(f"neutral terms also listed elsewhere: {', '.join(clash)}")
        if problems:
            raise ValueError("; ".join(problems))
        return self


@dataclass(frozen=True)
class TermHit:
    term: str  # the lexicon phrase that matched
    allergens: frozenset[Allergen]
    opaque: bool


@dataclass(frozen=True)
class Lexicon:
    version: int
    phrases: Mapping[str, TermHit]
    pattern: re.Pattern[str]

    def match(self, text: str) -> list[TermHit]:
        """Every lexicon phrase in ``text``, left to right, the longest phrase winning at each position."""
        hits = []
        for found in self.pattern.finditer(text.lower()):
            hits.append(self.phrases[" ".join(re.split(r"[\s-]+", found.group("term")))])
        return hits


def build_lexicon(data: Any) -> Lexicon:
    parsed = _LexiconFile.model_validate(data)
    allergens: dict[str, set[Allergen]] = {}
    for key, terms in parsed.allergens.items():
        for term in terms:
            allergens.setdefault(term, set()).add(key)
    opaque = set(parsed.opaque)
    names = set(allergens) | opaque | set(parsed.neutral)
    phrases = {
        name: TermHit(term=name, allergens=frozenset(allergens.get(name, ())), opaque=name in opaque) for name in names
    }
    # Longest first, so the regex alternation prefers "peanut butter" to "peanut" at the same position.
    alternatives = (r"[\s-]+".join(re.escape(word) for word in name.split()) for name in sorted(names, key=len)[::-1])
    pattern = re.compile(r"\b(?P<term>" + "|".join(alternatives) + r")(?:s|es)?\b")
    return Lexicon(version=parsed.version, phrases=phrases, pattern=pattern)


def load_lexicon(path: Path | None = None) -> Lexicon:
    path = path or get_settings().data_dir / LEXICON_PATH
    try:
        return build_lexicon(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"The allergen lexicon {path.name} could not be loaded: {exc}") from exc


@dataclass(frozen=True)
class AllergenTag:
    allergen: Allergen
    source: Source
    level: Level
    evidence: str  # the ingredient or description term behind the tag; "label" for label tags


@dataclass(frozen=True)
class DishAllergens:
    item_id: str
    tags: tuple[AllergenTag, ...]
    opaque_ingredients: tuple[str, ...]

    @property
    def unverified(self) -> bool:
        return bool(self.opaque_ingredients)

    @property
    def contains(self) -> frozenset[Allergen]:
        return frozenset(tag.allergen for tag in self.tags if tag.level is Level.CONTAINS)

    @property
    def may_contain(self) -> frozenset[Allergen]:
        return frozenset(tag.allergen for tag in self.tags if tag.level is Level.MAY_CONTAIN) - self.contains

    @property
    def flagged(self) -> frozenset[Allergen]:
        return self.contains | self.may_contain


def tag_item(item: MenuItem, lexicon: Lexicon, description: str | None = None) -> DishAllergens:
    """Tag one menu item; ``description`` is the vision model's description of its photo, when there is one."""
    tags: list[AllergenTag] = []
    for allergen in sorted(with_implied(item.label_allergens or ())):
        tags.append(AllergenTag(allergen, Source.LABEL, Level.CONTAINS, "label"))
    opaque: list[str] = []
    for ingredient in item.ingredients:
        for hit in lexicon.match(ingredient):
            tags += [
                AllergenTag(a, Source.LEXICON, Level.CONTAINS, ingredient) for a in sorted(with_implied(hit.allergens))
            ]
            if hit.opaque and ingredient not in opaque:
                opaque.append(ingredient)
    for hit in lexicon.match(description or ""):
        tags += [
            AllergenTag(a, Source.VISION, Level.MAY_CONTAIN, hit.term) for a in sorted(with_implied(hit.allergens))
        ]
    return DishAllergens(item_id=item.item_id, tags=tuple(dict.fromkeys(tags)), opaque_ingredients=tuple(opaque))


@dataclass(frozen=True)
class AllergenScore:
    allergen: Allergen
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float | None:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else None


@dataclass(frozen=True)
class TaggerReport:
    lexicon_version: int
    dishes: int
    unverified: int
    # Per dish, the true allergens that were neither tagged nor covered by an unverified flag. Must be empty.
    missed: Mapping[str, frozenset[Allergen]]
    scores: tuple[AllergenScore, ...]


def evaluate(
    results: Iterable[DishAllergens], truth: Mapping[str, frozenset[Allergen]], lexicon_version: int
) -> TaggerReport:
    """Score tags against the hand-checked allergens: the safety gate (``missed``) and per-allergen precision."""
    counts = {allergen: [0, 0, 0] for allergen in Allergen}
    missed: dict[str, frozenset[Allergen]] = {}
    dishes = unverified = 0
    for result in results:
        dishes += 1
        unverified += result.unverified
        actual = truth[result.item_id]
        for allergen in Allergen:
            tagged, present = allergen in result.flagged, allergen in actual
            if tagged or present:
                counts[allergen][0 if tagged and present else 1 if tagged else 2] += 1
        untagged = actual - result.flagged
        if untagged and not result.unverified:
            missed[result.item_id] = untagged
    scores = tuple(AllergenScore(allergen, *count) for allergen, count in counts.items())
    return TaggerReport(lexicon_version, dishes, unverified, missed, scores)
