"""Catalog loading on the 12-dish fixture: what loads, and every way a file can be wrong."""

import csv
import shutil
from pathlib import Path

import pytest

from food_concierge.config import Settings
from food_concierge.errors import CatalogIssue, CatalogValidationError
from food_concierge.ingestion.loader import MENU_FILE, RESTAURANTS_FILE, load_catalog
from food_concierge.ingestion.schemas import MENU_COLUMNS, MenuItem, MenuRow, Restaurant
from food_concierge.ingestion.taxonomy import Allergen, Category, Cuisine, Diet

A = Allergen
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "catalog"


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return tmp_path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]], columns: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def edit_menu(raw_dir: Path, target: str, **changes: str) -> None:
    rows = read(raw_dir / MENU_FILE)
    for row in rows:
        if row["item_id"] == target:
            row.update(changes)
    write(raw_dir / MENU_FILE, rows)


def issues_of(raw_dir: Path) -> list[CatalogIssue]:
    with pytest.raises(CatalogValidationError) as caught:
        load_catalog(raw_dir)
    return list(caught.value.issues)


def test_fixture_loads_cleanly() -> None:
    loaded = load_catalog(FIXTURE)

    assert [r.restaurant_id for r in loaded.catalog.restaurants] == ["fx_dhaba", "fx_udupi", "fx_cafe", "fx_wok"]
    assert len(loaded.catalog.items) == 12
    assert loaded.warnings == ()
    assert set(loaded.true_allergens) == {item.item_id for item in loaded.catalog.items}


def test_rows_are_normalised() -> None:
    loaded = load_catalog(FIXTURE)
    items = {item.item_id: item for item in loaded.catalog.items}

    assert loaded.catalog.restaurants[0].cuisines == (Cuisine.NORTH_INDIAN, Cuisine.PUNJABI)
    naan, tikka, dal = items["fx002"], items["fx003"], items["fx004"]
    assert naan.diet is Diet.VEGETARIAN  # "veg"
    assert tikka.diet is Diet.NON_VEGETARIAN  # "non-veg"
    assert naan.category is Category.BREAD
    assert items["fx012"].cuisine is Cuisine.INDO_CHINESE  # "Indo-Chinese"
    assert (dal.serves_min, dal.serves_max) == (2, 3)
    assert naan.ingredients == ("maida", "butter", "yogurt", "yeast", "salt")
    assert items["fx009"].contains_egg
    assert not items["fx009"].eggless
    assert items["fx010"].eggless


def test_label_blank_none_and_terms_are_kept_apart() -> None:
    items = {item.item_id: item for item in load_catalog(FIXTURE).catalog.items}

    assert items["fx002"].label_allergens is None  # not labelled
    assert items["fx006"].label_allergens == frozenset()  # declared allergen-free
    assert items["fx008"].label_allergens == frozenset({A.MILK, A.PEANUTS, A.TREE_NUTS})  # "milk; nuts"
    assert items["fx012"].label_allergens == frozenset({A.CRUSTACEANS, A.MOLLUSCS, A.SOY, A.EGGS})


def test_ground_truth_stays_out_of_menu_items() -> None:
    loaded = load_catalog(FIXTURE)

    assert loaded.true_allergens["fx012"] == {A.CRUSTACEANS, A.EGGS, A.SOY, A.GLUTEN, A.WHEAT, A.SESAME}
    assert loaded.true_allergens["fx006"] == frozenset()
    assert all(type(item) is MenuItem for item in loaded.catalog.items)
    assert "true_allergens" not in MenuItem.model_fields


def test_default_directory_comes_from_settings(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (raw_dir / "raw").mkdir()
    for name in (RESTAURANTS_FILE, MENU_FILE):
        shutil.move(raw_dir / name, raw_dir / "raw" / name)
    monkeypatch.setenv("DATA_DIR", str(raw_dir))

    assert Settings().raw_dir == raw_dir / "raw"
    assert len(load_catalog().catalog.items) == 12


def test_every_bad_row_is_reported_with_line_and_column(raw_dir: Path) -> None:
    edit_menu(raw_dir, "fx002", kcal="lots")
    edit_menu(raw_dir, "fx005", price_inr="-10", label_allergens="kiwi")
    edit_menu(raw_dir, "fx012", true_allergens="")

    found = {(i.line, i.column) for i in issues_of(raw_dir)}

    assert found == {(3, "kcal"), (6, "price_inr"), (6, "label_allergens"), (13, "true_allergens")}


@pytest.mark.parametrize(
    ("changes", "column", "message"),
    [
        ({"protein_g": "nan"}, "protein_g", "finite number"),
        ({"serves": "3-2"}, "serves", "smaller number first"),
        ({"serves": "0"}, "serves", "between 1 and 20"),
        ({"serves": "a few"}, "serves", "number of people"),
        ({"diet": "pescatarian"}, "diet", "Input should be"),
        ({"cuisine": "Pizza"}, "cuisine", "Input should be"),
        ({"category": "Fast Food"}, "category", "Input should be"),
        ({"contains_egg": "maybe"}, "contains_egg", "expected yes or no"),
        ({"ingredients": " ; "}, "ingredients", "at least 1 item"),
        ({"description": "A tab\tinside the description"}, "description", "control character"),
        ({"item_id": "Bad ID"}, "item_id", "String should match pattern"),
        ({"rating": "5.5"}, "rating", "less than or equal to 5"),
        ({"label_allergens": "dairy; kiwi"}, "label_allergens", "unknown allergen term(s): kiwi"),
        ({"true_allergens": ""}, "true_allergens", "write 'none'"),
        ({"true_allergens": "dairy"}, "true_allergens", "not an allergen key: dairy"),
        ({"true_allergens": "wheat; milk"}, "true_allergens", "also list gluten"),
    ],
)
def test_bad_cells_are_rejected(raw_dir: Path, changes: dict[str, str], column: str, message: str) -> None:
    edit_menu(raw_dir, "fx002", **changes)

    [issue] = issues_of(raw_dir)

    assert (issue.file, issue.line, issue.column) == (MENU_FILE, 3, column)
    assert message in issue.message


@pytest.mark.parametrize(
    ("item_id", "changes", "message"),
    [
        ("fx005", {"ingredients": "rice; urad dal; ghee"}, "vegan dish lists animal products: ghee"),
        ("fx001", {"ingredients": "paneer; chicken"}, "vegetarian dish lists meat or fish: chicken"),
        ("fx009", {"contains_egg": "no"}, "lists egg but contains_egg is no: egg, mayonnaise"),
        ("fx006", {"contains_egg": "yes"}, "vegan dish is marked contains_egg"),
    ],
)
def test_diet_that_contradicts_ingredients_is_an_error(
    raw_dir: Path, item_id: str, changes: dict[str, str], message: str
) -> None:
    edit_menu(raw_dir, item_id, **changes)

    [issue] = issues_of(raw_dir)

    assert issue.column is None
    assert message in issue.message


def test_calories_far_from_macros_are_a_warning(raw_dir: Path) -> None:
    edit_menu(raw_dir, "fx011", kcal="300")  # macros give 464

    loaded = load_catalog(raw_dir)

    [warning] = loaded.warnings
    assert (warning.line, warning.column) == (12, "kcal")
    assert "55%" in warning.message


def test_bad_restaurant_rows_are_rejected(raw_dir: Path) -> None:
    rows = read(raw_dir / RESTAURANTS_FILE)
    rows[3].update(cuisines="Martian", rating="")
    write(raw_dir / RESTAURANTS_FILE, rows)

    found = {(i.file, i.line, i.column) for i in issues_of(raw_dir)}

    # fx_wok no longer exists, so its dish is reported too.
    assert found == {
        (RESTAURANTS_FILE, 5, "cuisines"),
        (RESTAURANTS_FILE, 5, "rating"),
        (MENU_FILE, 13, "restaurant_id"),
    }


def test_cross_row_problems(raw_dir: Path) -> None:
    restaurants = read(raw_dir / RESTAURANTS_FILE)
    write(raw_dir / RESTAURANTS_FILE, [*restaurants, dict(restaurants[0], name="Another Name")])
    menu = read(raw_dir / MENU_FILE)
    menu[1]["item_id"] = "fx001"  # duplicate id
    menu[2]["restaurant_id"] = "fx_nowhere"
    menu[3]["name"] = "paneer butter MASALA"  # same dish name twice at fx_dhaba
    write(raw_dir / MENU_FILE, menu)

    found = {(i.file, i.line, i.column, i.message) for i in issues_of(raw_dir)}

    assert found == {
        (RESTAURANTS_FILE, 6, "restaurant_id", "duplicate fx_dhaba"),
        (MENU_FILE, 3, "item_id", "duplicate fx001"),
        (MENU_FILE, 4, "restaurant_id", "no restaurant fx_nowhere"),
        (MENU_FILE, 5, "name", "paneer butter MASALA is listed twice at this restaurant"),
    }


def test_header_problems(raw_dir: Path) -> None:
    menu = read(raw_dir / MENU_FILE)
    columns = [c for c in MENU_COLUMNS if c != "kcal"] + ["calories"]
    write(raw_dir / MENU_FILE, [{("calories" if k == "kcal" else k): v for k, v in r.items()} for r in menu], columns)
    (raw_dir / RESTAURANTS_FILE).write_text("restaurant_id,name,name,cuisines,area,rating\n", encoding="utf-8")

    found = {(i.file, i.line, i.message) for i in issues_of(raw_dir)}

    assert found == {
        (RESTAURANTS_FILE, 1, "duplicate columns: name"),
        (MENU_FILE, 1, "missing columns: kcal; unexpected columns: calories"),
    }


def test_rows_with_the_wrong_number_of_cells(raw_dir: Path) -> None:
    text = (raw_dir / RESTAURANTS_FILE).read_text(encoding="utf-8")
    (raw_dir / RESTAURANTS_FILE).write_text(
        text + "fx_short,Short Row\nfx_long,Long Row,Thai,Area,4.0,extra\n", "utf-8"
    )

    found = {(i.line, i.message) for i in issues_of(raw_dir)}

    assert found == {(6, "expected 5 cells"), (7, "expected 5 cells")}


def test_missing_and_undecodable_files(raw_dir: Path) -> None:
    (raw_dir / RESTAURANTS_FILE).unlink()
    (raw_dir / MENU_FILE).write_bytes(b"\xff\xfe not utf-8")

    found = {(i.file, i.line, i.message) for i in issues_of(raw_dir)}

    assert found == {(RESTAURANTS_FILE, None, "file not found"), (MENU_FILE, None, "not valid UTF-8")}


def test_byte_order_mark_is_accepted(raw_dir: Path) -> None:
    path = raw_dir / RESTAURANTS_FILE
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    assert len(load_catalog(raw_dir).catalog.restaurants) == 4


def test_records_built_in_code_validate_the_same_way() -> None:
    restaurant = Restaurant(
        restaurant_id="fx_code", name="Code Kitchen", cuisines=(Cuisine.THAI,), area="Anywhere", rating=4.0
    )
    row = MenuRow.model_validate(
        {
            "item_id": "fx100",
            "restaurant_id": restaurant.restaurant_id,
            "name": "Green Curry",
            "description": "Vegetables in a coconut milk green curry.",
            "category": Category.MAIN,
            "cuisine": Cuisine.THAI,
            "ingredients": ("coconut milk", "green curry paste", "tofu"),
            "diet": Diet.VEGAN,
            "contains_egg": False,
            "serves": (1, 2),
            "price_inr": 360,
            "kcal": 450,
            "protein_g": 14,
            "carbs_g": 30,
            "fat_g": 30,
            "rating": 4.1,
            "review_count": 10,
            "label_allergens": None,
            "true_allergens": frozenset({A.SOY}),
        }
    )

    item = row.to_item()
    assert item.serves_max == 2
    assert item.label_allergens is None
    with pytest.raises(ValueError, match="between 1 and 20"):
        MenuItem.model_validate(item.model_dump() | {"serves": (0, 2)})
    with pytest.raises(ValueError, match="category"):
        MenuItem.model_validate(item.model_dump() | {"category": 3})
