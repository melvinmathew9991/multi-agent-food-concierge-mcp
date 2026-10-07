"""Normalisation and consistency checks for catalog rows.

The diet check is deliberately small and conservative. Its job is to catch an authoring mistake
(ghee in a vegan dish, chicken in a vegetarian one, an egg nobody flagged), not to tag allergens:
the versioned allergen lexicon does that. A term it misses is caught later by the allergen tests
against the hand-checked ground truth.
"""

from __future__ import annotations

import re

from food_concierge.ingestion.taxonomy import Diet

MAX_SERVES = 20
ATWATER_TOLERANCE = 0.25

# Not "goat": goat cheese is vegetarian.
_MEAT_TERMS = (
    "chicken", "mutton", "lamb", "beef", "pork", "bacon", "ham", "salami", "pepperoni", "sausage",
    "keema", "gosht", "murgh", "fish", "machli", "machhli", "prawn", "shrimp", "jhinga", "crab", "lobster",
    "squid", "octopus", "mussel", "clam", "oyster", "anchovy", "tuna", "salmon", "gelatin", "gelatine",
)  # fmt: skip
_EGG_TERMS = ("egg", "anda", "mayonnaise", "mayo", "meringue")
_DAIRY_TERMS = (
    "milk", "butter", "ghee", "cream", "cheese", "paneer", "curd", "yogurt", "yoghurt", "dahi", "khoya", "khoa",
    "mawa", "malai", "chhena", "buttermilk", "chaas", "lassi", "rabri", "whey",
)  # fmt: skip
_OTHER_ANIMAL_TERMS = ("honey",)

# Plant-based names that contain an animal word; removed before matching.
_PLANT_PHRASES = re.compile(
    r"\b(?:coconut|almond|oat|soy|soya|cashew|rice|peanut)\s+(?:milk|cream|butter|yogurt|yoghurt|curd)\b"
    r"|\bcocoa\s+butter\b|\bnut\s+butter\b|\beggplants?\b"
)
_PLANT_PREFIX = re.compile(r"^(?:vegan|eggless|egg-free|dairy-free|plant-based)\b")


def _pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(terms) + r")(?:s|es)?\b")


_MEAT = _pattern(_MEAT_TERMS)
_EGG = _pattern(_EGG_TERMS)
_NOT_VEGAN = _pattern(_DAIRY_TERMS + _EGG_TERMS + _OTHER_ANIMAL_TERMS)


def parse_serves(text: str) -> tuple[int, int]:
    """'2' → (2, 2); '2-3' → (2, 3). The range itself is checked by the schema."""
    match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", text)
    if match is None:
        raise ValueError("expected a number of people such as '2' or a range such as '2-3'")
    low = int(match.group(1))
    return low, int(match.group(2) or low)


def _animal_text(ingredient: str) -> str:
    text = ingredient.lower()
    if _PLANT_PREFIX.match(text):
        return ""
    return _PLANT_PHRASES.sub(" ", text)


def diet_problems(diet: Diet, contains_egg: bool, ingredients: tuple[str, ...]) -> list[str]:
    """Contradictions between the declared diet, the egg flag and the ingredient list (empty when consistent)."""
    problems: list[str] = []
    texts = [(ingredient, _animal_text(ingredient)) for ingredient in ingredients]
    if diet is not Diet.NON_VEGETARIAN:
        meat = [ingredient for ingredient, text in texts if _MEAT.search(text)]
        if meat:
            problems.append(f"{diet.value} dish lists meat or fish: {', '.join(meat)}")
    if diet is Diet.VEGAN:
        animal = [ingredient for ingredient, text in texts if _NOT_VEGAN.search(text)]
        if animal:
            problems.append(f"vegan dish lists animal products: {', '.join(animal)}")
        if contains_egg:
            problems.append("vegan dish is marked contains_egg")
    if not contains_egg:
        eggs = [ingredient for ingredient, text in texts if _EGG.search(text)]
        if eggs:
            problems.append(f"lists egg but contains_egg is no: {', '.join(eggs)}")
    return problems


def atwater_kcal(protein_g: float, carbs_g: float, fat_g: float) -> float:
    return 4 * protein_g + 4 * carbs_g + 9 * fat_g


def atwater_deviation(kcal: int, protein_g: float, carbs_g: float, fat_g: float) -> float:
    """Relative gap between stated calories and the 4/4/9 estimate from the macros (audit A13)."""
    return abs(atwater_kcal(protein_g, carbs_g, fat_g) - kcal) / kcal
