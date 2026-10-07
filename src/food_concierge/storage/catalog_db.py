"""The SQLite catalog: validated records and allergen tags, built once offline and read by the services.

- ``dish_key`` groups the same dish across restaurants, so results can be de-duplicated (audit A13).
- ``doc_text`` is the semantic text that gets embedded: name, description, cuisine, category, ingredients, diet.
  Numbers (price, calories, macros) stay in SQL columns, where filters are exact (audit A14).
- Hard filters run as indexed SQL before any vector search (``candidate_ids``).
- The hand-checked ground truth never enters the database.

The build writes a temporary file and swaps it in atomically, so a reader never sees a half-built catalog.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from food_concierge.errors import NotReadyError
from food_concierge.ingestion.allergens import DishAllergens
from food_concierge.ingestion.loader import Catalog
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.ingestion.taxonomy import Allergen, Cuisine, Diet

DOC_TEXT_VERSION = 1
CATALOG_DB = "catalog.db"  # under Settings.processed_dir

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE restaurants (
    restaurant_id TEXT PRIMARY KEY, name TEXT NOT NULL, cuisines TEXT NOT NULL, area TEXT NOT NULL,
    rating REAL NOT NULL
);
CREATE TABLE dishes (dish_key TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE items (
    item_id TEXT PRIMARY KEY,
    dish_key TEXT NOT NULL REFERENCES dishes(dish_key),
    restaurant_id TEXT NOT NULL REFERENCES restaurants(restaurant_id),
    name TEXT NOT NULL, description TEXT NOT NULL, category TEXT NOT NULL, cuisine TEXT NOT NULL,
    ingredients TEXT NOT NULL, diet TEXT NOT NULL, contains_egg INTEGER NOT NULL,
    serves_min INTEGER NOT NULL, serves_max INTEGER NOT NULL, price_inr INTEGER NOT NULL, kcal INTEGER NOT NULL,
    protein_g REAL NOT NULL, carbs_g REAL NOT NULL, fat_g REAL NOT NULL, rating REAL NOT NULL,
    review_count INTEGER NOT NULL, label_allergens TEXT, unverified INTEGER NOT NULL, doc_text TEXT NOT NULL
);
CREATE TABLE allergen_tags (
    item_id TEXT NOT NULL REFERENCES items(item_id), allergen TEXT NOT NULL, source TEXT NOT NULL,
    level TEXT NOT NULL, evidence TEXT NOT NULL,
    PRIMARY KEY (item_id, allergen, source, evidence)
);
CREATE TABLE images (
    item_id TEXT PRIMARY KEY REFERENCES items(item_id), file_page_url TEXT NOT NULL, file_url TEXT NOT NULL,
    author TEXT NOT NULL, licence TEXT NOT NULL, licence_url TEXT NOT NULL, sha256 TEXT NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, bytes INTEGER NOT NULL
);
CREATE TABLE descriptions (
    sha256 TEXT NOT NULL, variant TEXT NOT NULL, model TEXT NOT NULL, text TEXT NOT NULL,
    PRIMARY KEY (sha256, variant)
);
CREATE INDEX items_dish ON items(dish_key);
CREATE INDEX items_diet ON items(diet, contains_egg);
CREATE INDEX items_kcal ON items(kcal);
CREATE INDEX items_price ON items(price_inr);
CREATE INDEX items_cuisine ON items(cuisine);
CREATE INDEX allergen_tags_allergen ON allergen_tags(allergen, item_id);
"""

# A requested diet admits these dish diets: vegetarian (lacto-ovo) diners can eat vegan dishes too.
ALLOWED_DIETS: dict[Diet, tuple[Diet, ...]] = {
    Diet.VEGAN: (Diet.VEGAN,),
    Diet.VEGETARIAN: (Diet.VEGAN, Diet.VEGETARIAN),
    Diet.NON_VEGETARIAN: tuple(Diet),
}


def dish_key(name: str) -> str:
    """'Paneer Butter Masala' → 'paneer-butter-masala'; accents folded, punctuation dropped."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def _label(value: str) -> str:
    return value.replace("_", " ")


def document_text(item: MenuItem) -> str:
    """The semantic text embedded for search (``DOC_TEXT_VERSION``); no prices, calories or macros."""
    diet = "non-vegetarian" if item.diet is Diet.NON_VEGETARIAN else item.diet.value
    return (
        f"{item.name}. {item.description} {_label(item.cuisine.value).title()} {item.category.value}. "
        f"Ingredients: {', '.join(item.ingredients)}. {diet.capitalize()}."
    )


def data_sha256(paths: Iterable[Path]) -> str:
    """One hash over the source files (names and bytes), binding a build to its inputs."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def build_catalog_db(
    catalog: Catalog,
    tags: Mapping[str, DishAllergens],
    path: Path,
    *,
    lexicon_version: int,
    source_sha256: str,
) -> Path:
    """Write the catalog database to ``path``, replacing any previous build atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.unlink(missing_ok=True)
    connection = sqlite3.connect(partial)
    try:
        with connection:
            connection.executescript(SCHEMA)
            connection.executemany(
                "INSERT INTO meta VALUES (?, ?)",
                [
                    ("doc_text_version", str(DOC_TEXT_VERSION)),
                    ("lexicon_version", str(lexicon_version)),
                    ("data_sha256", source_sha256),
                ],
            )
            connection.executemany(
                "INSERT INTO restaurants VALUES (?, ?, ?, ?, ?)",
                [(r.restaurant_id, r.name, ";".join(r.cuisines), r.area, r.rating) for r in catalog.restaurants],
            )
            dishes = {dish_key(item.name): item.name for item in reversed(catalog.items)}  # first name seen wins
            connection.executemany("INSERT INTO dishes VALUES (?, ?)", sorted(dishes.items()))
            connection.executemany(
                "INSERT INTO items VALUES (" + ", ".join("?" * 22) + ")",  # noqa: S608 - placeholders only
                [_item_row(item, tags[item.item_id]) for item in catalog.items],
            )
            connection.executemany(
                "INSERT INTO allergen_tags VALUES (?, ?, ?, ?, ?)",
                [
                    (item.item_id, tag.allergen.value, tag.source.value, tag.level.value, tag.evidence)
                    for item in catalog.items
                    for tag in tags[item.item_id].tags
                ],
            )
            connection.executemany(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [tuple(photo.model_dump().values()) for photo in catalog.photos],
            )
    finally:
        connection.close()
    os.replace(partial, path)
    return path


def _item_row(item: MenuItem, tags: DishAllergens) -> tuple[object, ...]:
    label = None if item.label_allergens is None else ";".join(sorted(item.label_allergens))
    return (
        item.item_id, dish_key(item.name), item.restaurant_id, item.name, item.description, item.category.value,
        item.cuisine.value, ";".join(item.ingredients), item.diet.value, int(item.contains_egg), item.serves_min,
        item.serves_max, item.price_inr, item.kcal, item.protein_g, item.carbs_g, item.fat_g, item.rating,
        item.review_count, label, int(tags.unverified), document_text(item),
    )  # fmt: skip


class Filters(BaseModel):
    """Hard constraints; every candidate satisfies all of them (PRD F4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    diet: Diet | None = None
    eggless: bool = False
    exclude_allergens: frozenset[Allergen] = frozenset()
    max_kcal: int | None = Field(default=None, gt=0)
    max_price_inr: int | None = Field(default=None, gt=0)
    cuisines: frozenset[Cuisine] = frozenset()
    # Unverified dishes have an ingredient whose recipe can't be checked; they are left out whenever allergens
    # matter, unless the caller will ask the user to confirm (Phase 5).
    include_unverified: bool = False


class CatalogStore:
    """Read-only access to a built catalog database."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise NotReadyError()
        self._connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)

    def close(self) -> None:
        self._connection.close()

    def meta(self) -> dict[str, str]:
        return dict(self._connection.execute("SELECT key, value FROM meta").fetchall())

    def doc_texts(self) -> dict[str, str]:
        rows = self._connection.execute("SELECT item_id, doc_text FROM items ORDER BY item_id")
        return dict(rows.fetchall())

    def dish_keys(self) -> dict[str, str]:
        return dict(self._connection.execute("SELECT item_id, dish_key FROM items").fetchall())

    def candidate_ids(self, filters: Filters) -> list[str]:
        """Item ids that satisfy every filter, in item-id order."""
        where: list[str] = []
        params: list[object] = []
        if filters.diet is not None:
            allowed = ALLOWED_DIETS[filters.diet]
            where.append(f"diet IN ({', '.join('?' * len(allowed))})")
            params += [diet.value for diet in allowed]
        excluded = set(filters.exclude_allergens)
        if filters.eggless:
            where.append("contains_egg = 0")
            excluded.add(Allergen.EGGS)
        if excluded:
            where.append(
                "NOT EXISTS (SELECT 1 FROM allergen_tags t WHERE t.item_id = items.item_id "  # noqa: S608
                f"AND t.allergen IN ({', '.join('?' * len(excluded))}))"
            )
            params += sorted(excluded)
            if not filters.include_unverified:
                where.append("unverified = 0")
        if filters.max_kcal is not None:
            where.append("kcal <= ?")
            params.append(filters.max_kcal)
        if filters.max_price_inr is not None:
            where.append("price_inr <= ?")
            params.append(filters.max_price_inr)
        if filters.cuisines:
            where.append(f"cuisine IN ({', '.join('?' * len(filters.cuisines))})")
            params += sorted(filters.cuisines)
        # Fixed fragments and "?" placeholders only; every value is a bound parameter.
        clause = " WHERE " + " AND ".join(where) if where else ""
        sql = "SELECT item_id FROM items" + clause + " ORDER BY item_id"  # noqa: S608
        return [row[0] for row in self._connection.execute(sql, params)]
