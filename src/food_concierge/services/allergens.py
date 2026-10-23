"""Which of a diner's allergens each dish has, for the ``check_allergens`` tool (architecture §4; PRD F5, F8).

- ``contains``: a label or an ingredient says so. Search never offers these dishes to someone avoiding them.
- ``may_contain``: only the photo's description suggests it: a warning, never proof (ADR-0007).
- ``unverified``: an ingredient's recipe can't be checked, so finding nothing is not the same as nothing being
  there; the agent must ask before recommending the dish to someone avoiding an allergen (PRD F8).

The answer comes from the same stored tags that search filters on, so the two can't disagree.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from pydantic import BaseModel, ConfigDict

from food_concierge.ingestion.allergens import Level
from food_concierge.ingestion.taxonomy import Allergen
from food_concierge.services.catalog import Tag, known_items
from food_concierge.storage.catalog_db import CatalogStore

DISCLAIMER = (
    "Allergen information is derived from restaurant labels, ingredient lists and photos of the dish. It can be "
    "incomplete and is not medical advice: anyone with a serious allergy should confirm with the restaurant."
)


class AllergenCheck(BaseModel):
    """One dish against the allergens asked about; ``clear`` only when none is found and the recipe is checked."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    name: str
    contains: list[Allergen]
    may_contain: list[Allergen]
    unverified: bool
    clear: bool
    evidence: list[Tag]  # the tags behind ``contains`` and ``may_contain``


class AllergenService:
    """Checks dishes against allergens, from the catalog's stored tags."""

    def __init__(self, store: CatalogStore) -> None:
        self._store = store

    def check(self, values: Iterable[str], allergens: Collection[Allergen] = ()) -> list[AllergenCheck]:
        """Each dish, in the order asked, against ``allergens`` (every allergen when none is given).

        Raises ``NotFoundError`` naming any unknown id.
        """
        ids, rows = known_items(self._store, values)
        asked = set(allergens) or set(Allergen)
        tags = self._store.allergen_tags(ids)
        checks = []
        for item_id in ids:
            relevant = [tag for tag in tags.get(item_id, []) if tag.allergen in asked]
            contains = sorted({tag.allergen for tag in relevant if tag.level is Level.CONTAINS})
            may_contain = sorted({tag.allergen for tag in relevant if tag.level is Level.MAY_CONTAIN} - set(contains))
            unverified = bool(rows[item_id]["unverified"])
            checks.append(
                AllergenCheck(
                    item_id=item_id,
                    name=rows[item_id]["name"],
                    contains=contains,
                    may_contain=may_contain,
                    unverified=unverified,
                    clear=not (contains or may_contain or unverified),
                    evidence=[Tag.of(tag) for tag in relevant],
                )
            )
        return checks
