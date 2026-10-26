"""Dish lookups for the tools: one dish in full, or several as summaries (architecture §4).

Everything is read from the built catalog (``CatalogStore``). Nothing here decides whether a dish fits a diner's
constraints: those rules live once, in ``candidate_ids`` and ``Filters.admits``. Prices are in rupees, as in the
catalog.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

from food_concierge.errors import InputValidationError, NotFoundError
from food_concierge.ingestion.allergens import AllergenTag, Level, Source
from food_concierge.ingestion.photos import commons_thumb_url
from food_concierge.ingestion.taxonomy import Allergen, Category, Cuisine, Diet
from food_concierge.storage.catalog_db import CatalogStore

MAX_DISHES = 20  # dish ids in one request


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True)


class PhotoCredit(_Out):
    """What the photo's licence asks to be shown wherever the photo is: author, licence and source."""

    author: str
    licence: str
    licence_url: str
    file_page_url: str  # the file's page on Wikimedia Commons


class PhotoFile(PhotoCredit):
    """The credit, plus the original upload and the hash that pins it."""

    file_url: str  # the full-size original, often several MB; ``photo_url`` is the one to show
    sha256: str  # pins the file; names its thumbnail


class Tag(_Out):
    allergen: Allergen
    source: Source
    level: Level
    evidence: str  # the ingredient or description term behind the tag; "label" for a restaurant's label

    @classmethod
    def of(cls, tag: AllergenTag) -> Tag:
        return cls(allergen=tag.allergen, source=tag.source, level=tag.level, evidence=tag.evidence)


class DishSummary(_Out):
    """A dish as a result list shows it."""

    item_id: str
    name: str
    restaurant_id: str
    restaurant: str
    cuisine: Cuisine
    category: Category
    diet: Diet
    eggless: bool
    price_inr: int
    kcal: int
    contains: list[Allergen]
    may_contain: list[Allergen]  # from the photo only: a warning, never a filter (ADR-0007)
    unverified: bool  # an ingredient's recipe can't be checked, so allergens may be missing
    photo_url: str | None  # Commons' 500 px copy; show ``photo`` (the credit) with it
    photo: PhotoCredit | None


class DishDetail(DishSummary):
    """One dish in full."""

    dish_key: str  # the same dish at another restaurant shares it
    description: str
    ingredients: list[str]
    serves_min: int
    serves_max: int
    protein_g: float
    carbs_g: float
    fat_g: float
    rating: float
    review_count: int
    label_allergens: list[Allergen] | None  # what the restaurant declares; None when it declares nothing
    tags: list[Tag]
    restaurant_area: str
    restaurant_rating: float
    photo: PhotoFile | None


def dish_ids(values: Iterable[str], *, distinct: bool = True) -> list[str]:
    """The ids a caller gave, trimmed, lower-cased and in order, blanks dropped, and de-duplicated unless a repeat
    means something (a meal with two of a dish); refuses none or too many."""
    ids = [value.strip().lower() for value in values if value.strip()]
    if distinct:
        ids = list(dict.fromkeys(ids))
    if not ids:
        raise InputValidationError("Give at least one dish id.")
    if len(ids) > MAX_DISHES:
        raise InputValidationError(f"Give at most {MAX_DISHES} dish ids at a time.")
    return ids


def known_items(
    store: CatalogStore, values: Iterable[str], *, distinct: bool = True
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The ids asked for (see ``dish_ids``) and their rows; raises ``NotFoundError`` naming any unknown id."""
    ids = dish_ids(values, distinct=distinct)
    rows = store.items(ids)
    unknown = list(dict.fromkeys(item_id for item_id in ids if item_id not in rows))
    if unknown:
        raise NotFoundError(f"No dish has the id {', '.join(unknown)}.")
    return ids, rows


def _split(value: str | None) -> list[str]:
    return value.split(";") if value else []


class CatalogService:
    """Read-only dish lookups over one built catalog."""

    def __init__(self, store: CatalogStore) -> None:
        self._store = store

    def get_dish(self, dish_id: str) -> DishDetail:
        """One dish in full; raises ``NotFoundError`` for an unknown id."""
        return self.get_dishes([dish_id])[0]

    def get_dishes(self, values: Iterable[str]) -> list[DishDetail]:
        """These dishes in full, in the order asked; raises ``NotFoundError`` naming any unknown id."""
        ids, rows = known_items(self._store, values)
        return self._details(ids, rows)

    def summaries(self, values: Iterable[str]) -> list[DishSummary]:
        """These dishes as a result list shows them, in the order asked."""
        return [summary_of(detail) for detail in self.get_dishes(values)]

    def all_summaries(self) -> list[DishSummary]:
        """Every dish as a result list shows it, in id order: the whole menu, past the per-request limit."""
        ids = sorted(self._store.doc_texts())
        return [summary_of(detail) for detail in self._details(ids, self._store.items(ids))]

    def _details(self, ids: list[str], rows: dict[str, dict[str, Any]]) -> list[DishDetail]:
        tags = self._store.allergen_tags(ids)
        return [_detail(rows[item_id], tags.get(item_id, [])) for item_id in ids]


def summary_of(detail: DishDetail) -> DishSummary:
    return DishSummary.model_validate(detail.model_dump(include=set(DishSummary.model_fields)))


def _detail(row: dict[str, Any], tags: list[AllergenTag]) -> DishDetail:
    contains = sorted({tag.allergen for tag in tags if tag.level is Level.CONTAINS})
    may_contain = sorted({tag.allergen for tag in tags if tag.level is Level.MAY_CONTAIN} - set(contains))
    label = row["label_allergens"]
    photo = photo_url = None
    if row["file_url"] is not None:
        photo = PhotoFile(
            author=row["author"],
            licence=row["licence"],
            licence_url=row["licence_url"],
            file_page_url=row["file_page_url"],
            file_url=row["file_url"],
            sha256=row["photo_sha256"],
        )
        photo_url = commons_thumb_url(row["file_url"], row["photo_width"])
    return DishDetail(
        item_id=row["item_id"],
        name=row["name"],
        restaurant_id=row["restaurant_id"],
        restaurant=row["restaurant_name"],
        cuisine=row["cuisine"],
        category=row["category"],
        diet=row["diet"],
        eggless=not row["contains_egg"],
        price_inr=row["price_inr"],
        kcal=row["kcal"],
        contains=contains,
        may_contain=may_contain,
        unverified=bool(row["unverified"]),
        photo_url=photo_url,
        dish_key=row["dish_key"],
        description=row["description"],
        ingredients=_split(row["ingredients"]),
        serves_min=row["serves_min"],
        serves_max=row["serves_max"],
        protein_g=row["protein_g"],
        carbs_g=row["carbs_g"],
        fat_g=row["fat_g"],
        rating=row["rating"],
        review_count=row["review_count"],
        label_allergens=None if label is None else [Allergen(a) for a in _split(label)],
        tags=[Tag.of(tag) for tag in tags],
        restaurant_area=row["restaurant_area"],
        restaurant_rating=row["restaurant_rating"],
        photo=photo,
    )
