"""A diner's constraints: parsed from loose input, merged, and checked against a proposed meal (architecture §4;
PRD F4, F7, F8).

- **Parse**: tool arguments arrive as text ("Dairy, nuts", "veg", "eggless"). They map onto the catalog's terms
  through the taxonomy's aliases; an unknown term is refused, never dropped, since a dropped allergen is a missed one.
- **Read from text**: a query that says what to avoid ("peanut-free snack", "no dairy", "vegan dessert") yields
  those constraints too (``constraints_in_text``), because embeddings place "peanut-free" next to "peanut": search
  would otherwise rank the very allergen first (audit H1). Only avoidance is read, never a wish ("peanut chikki").
  It is a backstop to the arguments, not the contract (#142, ``read_query``): what the menu doesn't record ("no
  onion", "halal") is named for the note, and so is an allergen mentioned near a negation but not read.
- **Merge**: constraints from several turns or diners only tighten: allergens are joined, caps take the lower
  value, the strictest diet wins. Cuisine is a preference, not a safety rule, so the latest one given wins.
- **Check**: each dish of a proposal against the per-dish rules, and the meal's totals against the meal caps.
  Search applies the same per-dish rules in SQL (``candidate_ids``); a test holds the two to the same verdict.
  A dish with an unverified recipe breaks the rules whenever allergens matter, until the diner confirms it
  (``include_unverified``, PRD F8). Allergens only the photo suggests are warnings (ADR-0007).

The Phase 5 verifier builds on ``check``; grounding and numeric-claim checks belong there.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from food_concierge.errors import InputValidationError
from food_concierge.ingestion.taxonomy import (
    DIET_ALIASES,
    USER_TERMS,
    Allergen,
    Cuisine,
    Diet,
    parse_allergen_terms,
    slug,
    with_implied,
)
from food_concierge.services.catalog import CatalogService, DishDetail
from food_concierge.services.nutrition import MealTotals, NutritionService
from food_concierge.storage.catalog_db import ALLOWED_DIETS, CatalogStore, Filters

# Strictest first: a vegan dish suits every diet, a non-vegetarian one only a non-vegetarian diner.
_STRICTNESS = (Diet.VEGAN, Diet.VEGETARIAN, Diet.NON_VEGETARIAN)
_EGGLESS = {"eggless", "egg_free", "no_egg", "no_eggs"}


class Constraints(BaseModel):
    """What a diner asked for: per-dish rules, the ones search filters on, and caps on the whole meal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    diet: Diet | None = None
    eggless: bool = False
    exclude_allergens: frozenset[Allergen] = frozenset()
    max_kcal: int | None = Field(default=None, gt=0)  # per serving of a dish
    max_price_inr: int | None = Field(default=None, gt=0)  # per dish
    cuisines: frozenset[Cuisine] = frozenset()
    include_unverified: bool = False  # the diner has confirmed dishes whose recipe can't be checked
    meal_max_kcal: int | None = Field(default=None, gt=0)  # per diner, for the most the order may hold
    meal_budget_inr: int | None = Field(default=None, gt=0)  # the whole order

    @model_validator(mode="before")
    @classmethod
    def _eggless_diet(cls, data: Any) -> Any:
        # "Eggless" is a vegetarian diet without egg, so it sets two fields.
        if isinstance(data, Mapping) and isinstance(data.get("diet"), str) and slug(data["diet"]) in _EGGLESS:
            return {**data, "diet": Diet.VEGETARIAN, "eggless": True}
        return data

    @field_validator("diet", mode="before")
    @classmethod
    def _diet(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        key = slug(value)
        return DIET_ALIASES.get(key, key) or None

    @field_validator("exclude_allergens", mode="before")
    @classmethod
    def _allergens(cls, value: object) -> object:
        if isinstance(value, str):
            return parse_allergen_terms(value)
        if isinstance(value, Iterable) and all(isinstance(term, str) for term in value):
            return parse_allergen_terms(";".join(value))
        return value

    @field_validator("cuisines", mode="before")
    @classmethod
    def _cuisines(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if isinstance(value, Iterable):
            return frozenset(slug(term) for term in value if isinstance(term, str) and term.strip())
        return value

    @property
    def excluded(self) -> frozenset[Allergen]:
        """The allergens to avoid; eggless avoids eggs too."""
        return self.exclude_allergens | ({Allergen.EGGS} if self.eggless else frozenset())

    @property
    def filters(self) -> Filters:
        """The per-dish rules, as search applies them."""
        return Filters(
            diet=self.diet,
            eggless=self.eggless,
            exclude_allergens=self.exclude_allergens,
            max_kcal=self.max_kcal,
            max_price_inr=self.max_price_inr,
            cuisines=self.cuisines,
            include_unverified=self.include_unverified,
        )


def parse_constraints(data: Mapping[str, Any]) -> Constraints:
    """Constraints from loose input; raises ``InputValidationError`` naming each field it can't read."""
    try:
        return Constraints.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'constraints'}: {e['msg'].removeprefix('Value error, ')}"
            for e in error.errors()
        )
        raise InputValidationError(f"The constraints could not be read: {problems}") from error


_TERM = "|".join(r"[\s-]+".join(map(re.escape, term.split())) for term in sorted(USER_TERMS, key=len, reverse=True))
_SEP = r"\s*(?:,|/|&|\band\b|\bor\b|\bnor\b)\s*"
_TERMS = rf"(?:{_TERM})(?:{_SEP}(?:{_TERM}))*"
# Words that say what follows is avoided (#142 widened these from the audit's misses).
_CUES = (
    r"no|non|without|avoid|avoiding|exclude|excluding|minus|zero|free\s+(?:of|from)|"
    r"allergic\s+to|allerg(?:y|ies)\s*(?:to|:)|(?:intolerant|sensitive|sensitivity|react|reacts)\s+to|"
    r"(?:can'?t|cannot|don'?t|do\s+not|never)\s+(?:eat|have|take|tolerate|handle|do)|"
    r"nothing\s+(?:with|containing|that\s+has)|not\s+with|hold|skip|leave\s+(?:out|off)|except(?:\s+for)?|"
    r"anything\s+but|away\s+from|steer\s+clear\s+of"
)
_FILLER = r"(?:(?:any|the|all|ones?|with|of|those|containing)\s+)*"
_AVOIDED = re.compile(
    rf"\b(?:{_CUES})[\s-]+{_FILLER}(?P<listed>{_TERMS})\b"
    rf"|\b(?P<suffixed>{_TERM})[\s-]*(?:free|less)\b"
    rf"|\b(?P<named>{_TERM})[\s-]+(?:allergy|allergies|allergic|intolerance|intolerant|sensitivity|sensitive)\b"
    rf"|\b(?P<refused>{_TERM})\s+(?:(?:are|is)\s+(?:a\s+no(?:-go)?|off[\s-]+limits)|makes?\s+me\s+(?:sick|ill))\b"
    r"|\b(?P<condition>coeliac|celiac)\b",
    re.IGNORECASE,
)
_SPLIT_TERMS = re.compile(_SEP, re.IGNORECASE)
_NOT_ALLERGIC = re.compile(r"\b(?:not|never\s+been|isn'?t|aren'?t)\s+(?:allergic|intolerant|sensitive)\b", re.I)
_DIET_WORD = r"(?:vegan|plant[\s-]+based|vegetarian|veggie|veg|pure[\s-]+veg)"
_NOT_A_DIET = re.compile(
    r"\b(?:not|isn'?t|aren'?t|(?:doesn'?t|does\s+not|don'?t|do\s+not)\s+(?:have|need)\s+to\s+be|"
    rf"no\s+need\s+(?:for|to\s+be)|needn'?t\s+be)\s+(?:(?:a|necessarily|strictly|fully)\s+)?{_DIET_WORD}\b",
    re.IGNORECASE,
)
_NOT_DIET = r"(?<!non[\s-])(?<!non)"
_VEGAN = re.compile(rf"{_NOT_DIET}\b(?:vegan|plant[\s-]+based)\b", re.IGNORECASE)
_VEGETARIAN = re.compile(rf"{_NOT_DIET}\b(?:vegetarian|veggie|veg|pure[\s-]+veg)\b", re.IGNORECASE)
_NO_MEAT = re.compile(rf"\b(?:{_CUES})[\s-]+{_FILLER}meats?\b|\bmeat[\s-]*(?:free|less)\b", re.IGNORECASE)
_JAIN = re.compile(r"\bjain\b", re.IGNORECASE)
# Avoided or required, but no tag records it: named in the note, never applied.
_UNTRACKED = {
    "onion": "onions?",
    "garlic": "garlic",
    "beef": "beef",
    "pork": "pork|ham|bacon",
    "alcohol": "alcohol|alcoholic",
    "mushroom": "mushrooms?",
}
_UNTRACKED_TERM = "|".join(_UNTRACKED.values())
_UNTRACKED_AVOIDED = re.compile(
    rf"\b(?:{_CUES})[\s-]+{_FILLER}(?P<listed>(?:{_UNTRACKED_TERM})(?:{_SEP}(?:{_UNTRACKED_TERM}))*)\b"
    rf"|\b(?P<suffixed>{_UNTRACKED_TERM})[\s-]*(?:free|less)\b",
    re.IGNORECASE,
)
_UNTRACKED_REQUIRED = re.compile(r"\b(halal|kosher)\b", re.IGNORECASE)
JAIN_UNTRACKED = ("onion", "garlic", "root vegetables")
_NAMED = re.compile(rf"\b(?:{_TERM})\b", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:no|not|nothing|none|without|hold|skip|except|avoid\w*|away|free|allerg\w*|intoleran\w*|can'?t|cannot|"
    r"don'?t|never|zero|minus|exclud\w*|leave|steer|sensitiv\w*)\b",
    re.IGNORECASE,
)


def _term(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


class QueryReading(BaseModel):
    """What a request's words say: constraints to apply, and what can't be (#142)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraints: Constraints
    # Avoided or required but recorded by no tag ("no onion", "halal"): the note names them; nothing is applied.
    untracked: tuple[str, ...] = ()
    # Allergens the words name near a negation without being read as avoided ("nuts are a no for me" read, but a
    # phrasing the patterns don't know): the note asks the client to pass them as arguments.
    unread: frozenset[Allergen] = frozenset()


def read_query(text: str, include_unverified: bool = False) -> QueryReading:
    """The constraints a request states in words, and what it states that can't be applied.

    Read: allergens it avoids, a vegan or vegetarian diet ("no meat" is vegetarian; "not vegan" is no diet), and
    Jain (vegetarian without eggs). ``include_unverified`` is carried through, so merging these with the caller's
    own constraints keeps a confirmation the diner already gave.
    """
    allergens: set[Allergen] = set()
    for match in _AVOIDED.finditer(_NOT_ALLERGIC.sub(" ", text)):
        listed = next(group for group in match.groups() if group)
        for term in _SPLIT_TERMS.split(listed):
            allergens |= USER_TERMS.get(_term(term), frozenset())
    jain = bool(_JAIN.search(text))
    diet_text = _NOT_A_DIET.sub(" ", text)
    diet = (
        Diet.VEGAN
        if _VEGAN.search(diet_text)
        else Diet.VEGETARIAN
        if _VEGETARIAN.search(diet_text) or _NO_MEAT.search(text) or jain
        else None
    )
    if jain:
        allergens.add(Allergen.EGGS)
    untracked: dict[str, None] = dict.fromkeys(JAIN_UNTRACKED if jain else ())
    for match in _UNTRACKED_AVOIDED.finditer(text):
        for word in _SPLIT_TERMS.split(match.group("listed") or match.group("suffixed")):
            untracked |= {name: None for name, pattern in _UNTRACKED.items() if re.fullmatch(pattern, word, re.I)}
    untracked |= dict.fromkeys(match.lower() for match in _UNTRACKED_REQUIRED.findall(text))
    excluded = with_implied(frozenset(allergens))
    named = frozenset().union(*(USER_TERMS[_term(word)] for word in _NAMED.findall(text)))
    return QueryReading(
        constraints=Constraints(
            diet=diet,
            eggless=Allergen.EGGS in allergens,
            exclude_allergens=excluded,
            include_unverified=include_unverified,
        ),
        untracked=tuple(untracked),
        unread=named - excluded if _NEGATION.search(text) else frozenset(),
    )


def constraints_in_text(text: str, include_unverified: bool = False) -> Constraints:
    """The constraints a request states in words (``read_query`` without what can't be applied)."""
    return read_query(text, include_unverified).constraints


def describe(constraints: Constraints) -> str:
    """The per-dish rules in words, for a note: "vegan, no peanuts or tree nuts"."""
    parts = [constraints.diet.value.replace("_", " ")] if constraints.diet else []
    if constraints.excluded:
        parts.append("no " + " or ".join(sorted(a.value.replace("_", " ") for a in constraints.excluded)))
    return ", ".join(parts)


def _lowest(values: Iterable[int | None]) -> int | None:
    given = [value for value in values if value is not None]
    return min(given) if given else None


def merge(*constraints: Constraints) -> Constraints:
    """One set of constraints that every given set allows; later cuisines replace earlier ones."""
    diets = [c.diet for c in constraints if c.diet is not None]
    return Constraints(
        diet=min(diets, key=_STRICTNESS.index) if diets else None,
        eggless=any(c.eggless for c in constraints),
        exclude_allergens=frozenset().union(*(c.exclude_allergens for c in constraints)),
        max_kcal=_lowest(c.max_kcal for c in constraints),
        max_price_inr=_lowest(c.max_price_inr for c in constraints),
        cuisines=next((c.cuisines for c in reversed(constraints) if c.cuisines), frozenset()),
        include_unverified=bool(constraints) and all(c.include_unverified for c in constraints),
        meal_max_kcal=_lowest(c.meal_max_kcal for c in constraints),
        meal_budget_inr=_lowest(c.meal_budget_inr for c in constraints),
    )


class Rule(StrEnum):
    DIET = "diet"
    EGGLESS = "eggless"
    ALLERGEN = "allergen"
    UNVERIFIED = "unverified"
    MAX_KCAL = "max_kcal"
    MAX_PRICE = "max_price_inr"
    CUISINE = "cuisine"
    MEAL_KCAL = "meal_max_kcal"
    MEAL_BUDGET = "meal_budget_inr"


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: Rule
    item_id: str | None  # None for a rule on the whole meal
    message: str


class ConstraintCheck(BaseModel):
    """A proposed meal against the constraints; ``ok`` only without violations. Warnings don't block."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    violations: list[Violation]
    warnings: list[str]
    totals: MealTotals


def dish_violations(dish: DishDetail, constraints: Constraints) -> list[Violation]:
    """Every per-dish rule ``dish`` breaks; empty exactly when search would return it."""
    c = constraints
    found: list[tuple[Rule, str]] = []
    if c.diet is not None and dish.diet not in ALLOWED_DIETS[c.diet]:
        found.append((Rule.DIET, f"is {dish.diet.value.replace('_', '-')}, not {c.diet.value.replace('_', '-')}"))
    if c.eggless and not dish.eggless:
        found.append((Rule.EGGLESS, "contains egg"))
    excluded = c.excluded
    if hit := sorted(excluded & set(dish.contains)):
        found.append((Rule.ALLERGEN, f"contains {', '.join(a.value.replace('_', ' ') for a in hit)}"))
    if excluded and dish.unverified and not c.include_unverified:
        found.append((Rule.UNVERIFIED, "has an ingredient whose recipe can't be checked; ask the diner to confirm"))
    if c.max_kcal is not None and dish.kcal > c.max_kcal:
        found.append((Rule.MAX_KCAL, f"has {dish.kcal} kcal, over {c.max_kcal}"))
    if c.max_price_inr is not None and dish.price_inr > c.max_price_inr:
        found.append((Rule.MAX_PRICE, f"costs ₹{dish.price_inr}, over ₹{c.max_price_inr}"))
    if c.cuisines and dish.cuisine not in c.cuisines:
        found.append((Rule.CUISINE, f"is {dish.cuisine.value.replace('_', ' ')}, not a cuisine asked for"))
    return [Violation(rule=rule, item_id=dish.item_id, message=f"{dish.name} {text}.") for rule, text in found]


class ConstraintService:
    """Checks proposed meals against a diner's constraints, over one built catalog."""

    def __init__(self, store: CatalogStore) -> None:
        self._catalog = CatalogService(store)
        self._nutrition = NutritionService(store)

    def check(self, values: Iterable[str], constraints: Constraints, diners: int = 1) -> ConstraintCheck:
        """Each dish (a repeat counts in the totals) and the meal shared by ``diners``; raises ``NotFoundError``
        naming any unknown id."""
        totals = self._nutrition.meal_totals(values, diners)
        dishes = self._catalog.get_dishes(totals.dish_ids)
        violations = [violation for dish in dishes for violation in dish_violations(dish, constraints)]
        kcal = totals.per_person_max.kcal  # a dish serving a range of people may hold this much
        if constraints.meal_max_kcal is not None and kcal > constraints.meal_max_kcal:
            message = f"The meal has up to {kcal} kcal a person, over {constraints.meal_max_kcal}."
            violations.append(Violation(rule=Rule.MEAL_KCAL, item_id=None, message=message))
        if constraints.meal_budget_inr is not None and totals.price_inr > constraints.meal_budget_inr:
            message = f"The meal costs ₹{totals.price_inr}, over the budget of ₹{constraints.meal_budget_inr}."
            violations.append(Violation(rule=Rule.MEAL_BUDGET, item_id=None, message=message))
        warnings = [
            f"{dish.name} may contain {', '.join(a.value.replace('_', ' ') for a in hit)} (seen in its photo only)."
            for dish in dishes
            if (hit := sorted(constraints.excluded & set(dish.may_contain)))
        ]
        return ConstraintCheck(ok=not violations, violations=violations, warnings=warnings, totals=totals)
