"""Load ``restaurants.csv`` and ``menu.csv`` into validated records.

Every row is checked before anything is raised, so one run lists every problem in the files with
its file, line and column. Contradictions (a diet that doesn't match the ingredients, an unknown
allergen term, a dish at a restaurant that doesn't exist) are errors. Calories that disagree with
the macros are warnings: the authored catalog must have none, and its tests assert that.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from food_concierge.config import get_settings
from food_concierge.errors import CatalogIssue, CatalogValidationError
from food_concierge.ingestion.normalize import ATWATER_TOLERANCE, atwater_deviation
from food_concierge.ingestion.schemas import (
    MENU_COLUMNS,
    PHOTO_COLUMNS,
    RESTAURANT_COLUMNS,
    MenuItem,
    MenuRow,
    Photo,
    Restaurant,
)
from food_concierge.ingestion.taxonomy import Allergen

logger = logging.getLogger(__name__)

RESTAURANTS_FILE = "restaurants.csv"
MENU_FILE = "menu.csv"
ATTRIBUTIONS_FILE = "attributions.csv"  # optional: a dish may have no suitable photo

_Model = TypeVar("_Model", bound=BaseModel)


@dataclass(frozen=True)
class Catalog:
    restaurants: tuple[Restaurant, ...]
    items: tuple[MenuItem, ...]
    photos: tuple[Photo, ...] = ()


@dataclass(frozen=True)
class CatalogLoad:
    catalog: Catalog
    # Hand-checked allergens per item_id, for tests and evaluation only. The system never reads them.
    true_allergens: Mapping[str, frozenset[Allergen]]
    warnings: tuple[CatalogIssue, ...]


def load_catalog(raw_dir: Path | None = None) -> CatalogLoad:
    """Read and validate the catalog in ``raw_dir`` (default: ``Settings.raw_dir``).

    Raises ``CatalogValidationError`` carrying every problem found.
    """
    raw_dir = raw_dir or get_settings().raw_dir
    issues: list[CatalogIssue] = []

    restaurants: dict[str, Restaurant] = {}
    for line, row in _read_rows(raw_dir / RESTAURANTS_FILE, RESTAURANT_COLUMNS, issues):
        restaurant = _validate(Restaurant, RESTAURANTS_FILE, line, row, issues)
        if restaurant is None:
            continue
        if restaurant.restaurant_id in restaurants:
            issues.append(_issue(RESTAURANTS_FILE, line, "restaurant_id", f"duplicate {restaurant.restaurant_id}"))
            continue
        restaurants[restaurant.restaurant_id] = restaurant

    items: dict[str, MenuItem] = {}
    truth: dict[str, frozenset[Allergen]] = {}
    names: set[tuple[str, str]] = set()
    warnings: list[CatalogIssue] = []
    for line, row in _read_rows(raw_dir / MENU_FILE, MENU_COLUMNS, issues):
        menu_row = _validate(MenuRow, MENU_FILE, line, row, issues)
        if menu_row is None:
            continue
        if menu_row.item_id in items:
            issues.append(_issue(MENU_FILE, line, "item_id", f"duplicate {menu_row.item_id}"))
            continue
        if menu_row.restaurant_id not in restaurants:
            issues.append(_issue(MENU_FILE, line, "restaurant_id", f"no restaurant {menu_row.restaurant_id}"))
        name_key = (menu_row.restaurant_id, menu_row.name.casefold())
        if name_key in names:
            issues.append(_issue(MENU_FILE, line, "name", f"{menu_row.name} is listed twice at this restaurant"))
        names.add(name_key)
        deviation = atwater_deviation(menu_row.kcal, menu_row.protein_g, menu_row.carbs_g, menu_row.fat_g)
        if deviation > ATWATER_TOLERANCE:
            message = f"calories differ from 4P+4C+9F by {deviation:.0%} (limit {ATWATER_TOLERANCE:.0%})"
            warnings.append(_issue(MENU_FILE, line, "kcal", message))
        items[menu_row.item_id] = menu_row.to_item()
        truth[menu_row.item_id] = menu_row.true_allergens

    photos: dict[str, Photo] = {}
    if (raw_dir / ATTRIBUTIONS_FILE).exists():
        for line, row in _read_rows(raw_dir / ATTRIBUTIONS_FILE, PHOTO_COLUMNS, issues):
            photo = _validate(Photo, ATTRIBUTIONS_FILE, line, row, issues)
            if photo is None:
                continue
            if photo.item_id not in items:
                issues.append(_issue(ATTRIBUTIONS_FILE, line, "item_id", f"no menu item {photo.item_id}"))
            elif photo.item_id in photos:
                issues.append(_issue(ATTRIBUTIONS_FILE, line, "item_id", f"{photo.item_id} has two photos"))
            else:
                photos[photo.item_id] = photo

    if issues:
        raise CatalogValidationError(issues)
    logger.info(
        "catalog loaded",
        extra={"restaurants": len(restaurants), "items": len(items), "photos": len(photos), "warnings": len(warnings)},
    )
    return CatalogLoad(
        catalog=Catalog(
            restaurants=tuple(restaurants.values()), items=tuple(items.values()), photos=tuple(photos.values())
        ),
        true_allergens=truth,
        warnings=tuple(warnings),
    )


def _issue(file: str, line: int | None, column: str | None, message: str) -> CatalogIssue:
    return CatalogIssue(file=file, line=line, column=column, message=message)


def _read_rows(path: Path, columns: tuple[str, ...], issues: list[CatalogIssue]) -> list[tuple[int, dict[str, str]]]:
    rows: list[tuple[int, dict[str, str]]] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            problem = _header_problem(header, columns)
            if problem:
                issues.append(_issue(path.name, 1, None, problem))
                return []
            for row in reader:
                # DictReader puts surplus cells under None and fills missing ones with None.
                if None in row or None in row.values():
                    issues.append(_issue(path.name, reader.line_num, None, f"expected {len(columns)} cells"))
                    continue
                rows.append((reader.line_num, row))
    except FileNotFoundError:
        issues.append(_issue(path.name, None, None, "file not found"))
    except UnicodeDecodeError:
        issues.append(_issue(path.name, None, None, "not valid UTF-8"))
    return rows


def _header_problem(header: list[str], columns: tuple[str, ...]) -> str | None:
    duplicated = sorted({name for name in header if header.count(name) > 1})
    missing = [name for name in columns if name not in header]
    unexpected = [name for name in header if name not in columns]
    parts = []
    if duplicated:
        parts.append(f"duplicate columns: {', '.join(duplicated)}")
    if missing:
        parts.append(f"missing columns: {', '.join(missing)}")
    if unexpected:
        parts.append(f"unexpected columns: {', '.join(unexpected)}")
    return "; ".join(parts) or None


def _validate(
    model: type[_Model], file: str, line: int, row: dict[str, str], issues: list[CatalogIssue]
) -> _Model | None:
    try:
        return model.model_validate(row)
    except ValidationError as exc:
        for error in exc.errors():
            column = str(error["loc"][0]) if error["loc"] else None
            issues.append(_issue(file, line, column, error["msg"].removeprefix("Value error, ")))
        return None
