"""Allergen lexicon, three-source tagging, scoring, and the safety gate on the authored catalog."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from food_concierge.config import REPO_ROOT
from food_concierge.errors import ConfigError
from food_concierge.ingestion.allergens import (
    AllergenTag,
    DishAllergens,
    Level,
    Lexicon,
    Source,
    build_lexicon,
    evaluate,
    load_lexicon,
    tag_item,
)
from food_concierge.ingestion.loader import load_catalog
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.ingestion.taxonomy import Allergen

A = Allergen
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "catalog"


@pytest.fixture(scope="module")
def report_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("allergen_report", REPO_ROOT / "scripts" / "allergen_report.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["allergen_report"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return load_lexicon(REPO_ROOT / "data" / "lexicon" / "allergens.yaml")


def allergens_in(lexicon: Lexicon, text: str) -> set[Allergen]:
    return {allergen for hit in lexicon.match(text) for allergen in hit.allergens}


def item(**changes: Any) -> MenuItem:
    base: dict[str, Any] = {
        "item_id": "t1",
        "restaurant_id": "r1",
        "name": "Test Dish",
        "description": "A dish used only in tests.",
        "category": "main",
        "cuisine": "north_indian",
        "ingredients": ("rice",),
        "diet": "non_vegetarian",
        "contains_egg": False,
        "serves": (1, 1),
        "price_inr": 100,
        "kcal": 100,
        "protein_g": 5,
        "carbs_g": 15,
        "fat_g": 2,
        "rating": 4.0,
        "review_count": 1,
        "label_allergens": None,
    }
    return MenuItem.model_validate(base | changes)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("peanut butter", {A.PEANUTS}),  # the longest phrase wins: not milk
        ("almond butter", {A.TREE_NUTS}),
        ("coconut milk", set()),
        ("makki atta", set()),
        ("rice noodles", set()),
        ("glutinous rice", set()),
        ("eggplant", set()),
        ("eggless sponge", set()),
        ("prawns", {A.CRUSTACEANS}),  # plurals
        ("fishes", {A.FISH}),
        ("a wall of small tiles", set()),  # "es" only after a vowel or s, x, z, ch, sh: not "til" + "es"
        ("Cashews", {A.TREE_NUTS}),  # case
        ("soy-sauce", {A.SOY, A.WHEAT}),  # hyphenated phrase
        ("egg noodles", {A.EGGS, A.WHEAT}),
        ("hing", {A.WHEAT}),
        ("kimchi", {A.FISH, A.CRUSTACEANS}),
        ("dairy-free cheese", {A.MILK}),  # negations are ignored on purpose
        ("sesame burger bun", {A.SESAME, A.WHEAT}),
    ],
)
def test_lexicon_matching(lexicon: Lexicon, text: str, expected: set[Allergen]) -> None:
    assert allergens_in(lexicon, text) == expected


@pytest.mark.parametrize(
    ("text", "opaque"),
    [("chaat masala", True), ("garam masala", False), ("chilli sauce", True), ("soy sauce", False), ("farsan", True)],
)
def test_opaque_terms(lexicon: Lexicon, text: str, opaque: bool) -> None:
    assert any(hit.opaque for hit in lexicon.match(text)) is opaque


def test_three_sources_and_levels(lexicon: Lexicon) -> None:
    dish = item(ingredients=("paneer", "chaat masala", "atta"), label_allergens=frozenset({A.MILK}), diet="vegetarian")

    result = tag_item(dish, lexicon, description="Paneer cubes with a sprinkle of sesame and crushed peanuts")

    assert AllergenTag(A.MILK, Source.LABEL, Level.CONTAINS, "label") in result.tags
    assert AllergenTag(A.MILK, Source.LEXICON, Level.CONTAINS, "paneer") in result.tags
    assert AllergenTag(A.GLUTEN, Source.LEXICON, Level.CONTAINS, "atta") in result.tags  # wheat implies gluten
    assert AllergenTag(A.SESAME, Source.VISION, Level.MAY_CONTAIN, "sesame") in result.tags
    assert result.contains == {A.MILK, A.WHEAT, A.GLUTEN}
    assert result.may_contain == {A.SESAME, A.PEANUTS}  # milk is seen too, but already "contains"
    assert result.flagged == {A.MILK, A.WHEAT, A.GLUTEN, A.SESAME, A.PEANUTS}
    assert result.unverified
    assert result.opaque_ingredients == ("chaat masala",)


def test_label_keys_get_their_implied_keys(lexicon: Lexicon) -> None:
    result = tag_item(item(label_allergens=frozenset({A.WHEAT})), lexicon)

    assert result.contains == {A.WHEAT, A.GLUTEN}
    assert not result.unverified


def test_evaluate_counts_and_gate() -> None:
    results = [
        DishAllergens("d1", (AllergenTag(A.MILK, Source.LEXICON, Level.CONTAINS, "ghee"),), ()),
        DishAllergens("d2", (AllergenTag(A.SOY, Source.LABEL, Level.CONTAINS, "label"),), ()),
        DishAllergens("d3", (), ("chutney",)),
    ]
    truth = {"d1": frozenset({A.MILK, A.MUSTARD}), "d2": frozenset(), "d3": frozenset({A.PEANUTS})}

    report = evaluate(results, truth, lexicon_version=7)
    scores = {score.allergen: score for score in report.scores}

    assert (report.lexicon_version, report.dishes, report.unverified) == (7, 3, 1)
    assert report.missed == {"d1": {A.MUSTARD}}  # d3 is unverified, so its untagged peanuts pass the gate
    assert (scores[A.MILK].precision, scores[A.MILK].recall) == (1.0, 1.0)
    assert (scores[A.SOY].precision, scores[A.SOY].recall) == (0.0, None)
    assert (scores[A.MUSTARD].precision, scores[A.MUSTARD].recall) == (None, 0.0)
    assert scores[A.LUPIN].precision is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"version": 1, "allergens": {"milk": ["ghee", "ghee"]}, "neutral": [], "opaque": []}, "milk lists ghee twice"),
        ({"version": 1, "allergens": {"milk": ["Ghee"]}, "neutral": [], "opaque": []}, "lower-case words"),
        ({"version": 1, "allergens": {"milk": ["ghee"]}, "neutral": ["ghee"], "opaque": []}, "neutral terms also"),
        ({"version": 1, "allergens": {"dairy": ["ghee"]}, "neutral": [], "opaque": []}, "Input should be"),
        ({"version": 0, "allergens": {}, "neutral": [], "opaque": []}, "greater than or equal to 1"),
    ],
)
def test_invalid_lexicon_is_refused(tmp_path: Path, data: dict[str, Any], message: str) -> None:
    path = tmp_path / "allergens.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_lexicon(path)


def test_missing_or_malformed_lexicon_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not be loaded"):
        load_lexicon(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("version: [1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not be loaded"):
        load_lexicon(tmp_path / "bad.yaml")


def test_default_lexicon_path_comes_from_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "lexicon").mkdir()
    (tmp_path / "lexicon" / "allergens.yaml").write_text(
        "version: 3\nallergens: {milk: [ghee]}\nneutral: []\nopaque: []\n", encoding="utf-8"
    )
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    assert load_lexicon().version == 3
    assert build_lexicon({"version": 2, "allergens": {}, "neutral": [], "opaque": ["sauce"]}).version == 2


@pytest.mark.parametrize("raw_dir", [REPO_ROOT / "data" / "raw", FIXTURE], ids=["catalog", "fixture"])
def test_safety_gate(lexicon: Lexicon, raw_dir: Path) -> None:
    # Every hand-checked allergen of every dish is tagged, or the dish is unverified (Phase 2 DoD).
    loaded = load_catalog(raw_dir)

    report = evaluate((tag_item(i, lexicon) for i in loaded.catalog.items), loaded.true_allergens, lexicon.version)

    assert report.missed == {}
    # "Unverified" asks the user to confirm; it must stay the exception, not a way around the gate.
    assert report.unverified / report.dishes <= 0.25


def test_report_script(report_script: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert report_script.main() == 0
    out = capsys.readouterr().out
    assert "Safety gate: PASS" in out
    assert "| milk |" in out


def test_report_lists_misses(report_script: ModuleType) -> None:
    report = evaluate([DishAllergens("d1", (), ())], {"d1": frozenset({A.FISH})}, lexicon_version=1)

    text = report_script.render(report)

    assert "Safety gate: FAIL" in text
    assert "missed d1: fish" in text
