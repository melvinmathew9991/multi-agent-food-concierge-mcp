"""Typed catalog records, validated from the CSV text in ``data/raw/``.

The validators accept CSV strings and already-typed values alike, so the same models describe a row
read from a file and a record built in code. ``MenuRow`` is the only place the hand-checked
``true_allergens`` column exists: ``to_item()`` drops it, so nothing downstream of the loader can
use the ground truth by accident.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator

from food_concierge.ingestion.normalize import MAX_SERVES, diet_problems, parse_serves
from food_concierge.ingestion.taxonomy import (
    DIET_ALIASES,
    Allergen,
    Category,
    Cuisine,
    Diet,
    missing_implied,
    parse_allergen_keys,
    parse_allergen_terms,
    slug,
)

ID_PATTERN = r"^[a-z][a-z0-9_-]{1,39}$"
MAX_INGREDIENTS = 40
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
NONE = "none"  # an explicit "no allergens", as opposed to a blank cell


def _clean_text(value: str) -> str:
    if _CONTROL.search(value):
        raise ValueError("contains a line break or control character")
    return " ".join(value.split())


Text = Annotated[str, AfterValidator(_clean_text)]


def _split(value: object) -> object:
    if isinstance(value, str):
        return tuple(part for part in (" ".join(p.split()) for p in value.split(";")) if part)
    return value


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Restaurant(_Record):
    restaurant_id: str = Field(pattern=ID_PATTERN)
    name: Text = Field(min_length=2, max_length=80)
    cuisines: tuple[Cuisine, ...] = Field(min_length=1)
    area: Text = Field(min_length=2, max_length=60)
    rating: float = Field(ge=0, le=5)

    @field_validator("cuisines", mode="before")
    @classmethod
    def _parse_cuisines(cls, value: object) -> object:
        split = _split(value)
        return tuple(slug(c) for c in split) if isinstance(split, tuple) else split


class MenuItem(_Record):
    """One dish as served at one restaurant."""

    item_id: str = Field(pattern=ID_PATTERN)
    restaurant_id: str = Field(pattern=ID_PATTERN)
    name: Text = Field(min_length=2, max_length=80)
    description: Text = Field(min_length=10, max_length=400)
    category: Category
    cuisine: Cuisine
    ingredients: tuple[Annotated[Text, Field(min_length=2, max_length=60)], ...] = Field(
        min_length=1, max_length=MAX_INGREDIENTS
    )
    diet: Diet
    contains_egg: bool
    serves: tuple[int, int]
    price_inr: int = Field(gt=0, le=100_000)
    kcal: int = Field(gt=0, le=5_000)
    protein_g: float = Field(ge=0, le=500)
    carbs_g: float = Field(ge=0, le=1_000)
    fat_g: float = Field(ge=0, le=500)
    rating: float = Field(ge=0, le=5)
    review_count: int = Field(ge=0)
    # What the restaurant declares. None: not labelled (a blank cell); empty: declared allergen-free ("none").
    label_allergens: frozenset[Allergen] | None

    @property
    def serves_min(self) -> int:
        return self.serves[0]

    @property
    def serves_max(self) -> int:
        return self.serves[1]

    @property
    def eggless(self) -> bool:
        return not self.contains_egg

    @field_validator("category", "cuisine", "diet", mode="before")
    @classmethod
    def _slug(cls, value: object) -> object:
        # Enum members are strings too, so typed values take the same path.
        if not isinstance(value, str):
            return value
        key = slug(value)
        return DIET_ALIASES.get(key, key)

    @field_validator("ingredients", mode="before")
    @classmethod
    def _parse_ingredients(cls, value: object) -> object:
        return _split(value)

    @field_validator("contains_egg", mode="before")
    @classmethod
    def _yes_no(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        answer = value.strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError("expected yes or no")
        return answer == "yes"

    @field_validator("serves", mode="before")
    @classmethod
    def _parse_serves(cls, value: object) -> object:
        return parse_serves(value) if isinstance(value, str) else value

    @field_validator("serves", mode="after")
    @classmethod
    def _serves_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        low, high = value
        if not 1 <= low <= high <= MAX_SERVES:
            raise ValueError(f"must be between 1 and {MAX_SERVES}, with the smaller number first")
        return value

    @field_validator("label_allergens", mode="before")
    @classmethod
    def _parse_label(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        if not value.strip():
            return None
        return frozenset() if value.strip().lower() == NONE else parse_allergen_terms(value)

    @model_validator(mode="after")
    def _diet_matches_ingredients(self) -> MenuItem:
        problems = diet_problems(self.diet, self.contains_egg, self.ingredients)
        if problems:
            raise ValueError("; ".join(problems))
        return self


class MenuRow(MenuItem):
    """A ``menu.csv`` row: a menu item plus its hand-checked allergens, used only by tests and evaluation."""

    true_allergens: frozenset[Allergen]

    @field_validator("true_allergens", mode="before")
    @classmethod
    def _parse_truth(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        if not value.strip():
            raise ValueError("blank: every dish needs reviewed allergens; write 'none' if it has none")
        return frozenset() if value.strip().lower() == NONE else parse_allergen_keys(value)

    @field_validator("true_allergens", mode="after")
    @classmethod
    def _complete(cls, value: frozenset[Allergen]) -> frozenset[Allergen]:
        missing = missing_implied(value)
        if missing:
            raise ValueError(f"also list {', '.join(sorted(missing))} (implied by the keys given)")
        return value

    def to_item(self) -> MenuItem:
        data: dict[str, Any] = self.model_dump(exclude={"true_allergens"})
        return MenuItem.model_validate(data)


RESTAURANT_COLUMNS: tuple[str, ...] = tuple(Restaurant.model_fields)
MENU_COLUMNS: tuple[str, ...] = tuple(MenuRow.model_fields)
