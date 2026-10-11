"""Baseline C0, the original approach: parsers, the request in words, resumable collection and offline replay."""

from pathlib import Path

import pytest

from food_concierge.config import REPO_ROOT
from food_concierge.evaluation.dataset import TextQuery
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.original import (
    Judged,
    collect,
    constraint_phrase,
    dish_card,
    fixed_yes,
    missing,
    read_runs,
    relevance_prompt,
    replay,
    request_text,
    strict_yes,
)
from food_concierge.ingestion.loader import CatalogLoad, load_catalog
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.storage.catalog_db import Filters


@pytest.fixture(scope="module")
def loaded() -> CatalogLoad:
    return load_catalog(REPO_ROOT / "data" / "raw")


@pytest.fixture(scope="module")
def items(loaded: CatalogLoad) -> dict[str, MenuItem]:
    return {item.item_id: item for item in loaded.catalog.items}


def query(query_id: str, text: str, **fields: object) -> TextQuery:
    base: dict[str, object] = {"id": query_id, "text": text, "type": "constrained", "author": "drafted"}
    return TextQuery.model_validate(base | fields)


@pytest.mark.parametrize(
    ("reply", "strict", "fixed"),
    [
        ("yes", True, True),
        ("Yes", True, True),
        (" Yes ", True, True),
        ("Yes.", False, True),
        ("Yes\n", False, True),
        ("**Yes** - it is vegan.", False, True),
        ("No", False, False),
        ("No.", False, False),
        ("Yesterday's special", False, False),
        ("", False, False),
    ],
)
def test_the_strict_parser_drops_answers_the_fixed_one_reads(reply: str, strict: bool, fixed: bool) -> None:
    assert strict_yes(reply) is strict
    assert fixed_yes(reply) is fixed


def test_constraints_reach_the_request_only_as_words() -> None:
    filters = Filters.model_validate(
        {"diet": "vegan", "exclude_allergens": ["tree_nuts", "milk"], "max_kcal": 450, "eggless": True}
    )

    assert constraint_phrase(filters) == "vegan, eggless, no milk, tree nuts, under 450 kcal"
    assert request_text(query("t001", "a dessert", filters=filters)).startswith("a dessert, vegan, eggless")
    assert request_text(query("t002", "a dessert")) == "a dessert"
    assert constraint_phrase(Filters.model_validate({"max_price_inr": 100, "cuisines": ["kerala"]})) == (
        "under ₹100, kerala cuisine"
    )


def test_the_dish_card_shows_declared_allergens_only(items: dict[str, MenuItem]) -> None:
    unlabelled = next(item for item in items.values() if item.label_allergens is None)
    declared_free = next(item for item in items.values() if item.label_allergens == frozenset())

    assert "Allergens: \n" in dish_card(unlabelled)  # blank, as the original showed it
    assert "Allergens: none\n" in dish_card(declared_free)
    prompt = relevance_prompt("a dessert, no milk", unlabelled)
    assert "Request: a dessert, no milk" in prompt
    assert prompt.endswith("Answer Yes or No.")


def test_collect_saves_each_query_resumes_and_asks_a_changed_query_again(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    queries = [query("t001", "a dessert"), query("t002", "noodles")]
    asked: list[tuple[str, str]] = []

    def judge(request: str, item_id: str) -> Judged:
        asked.append((request, item_id))
        return Judged("Yes." if item_id == "k01" else "No", 150)

    runs = collect(queries, lambda _: ["k01", "k02"], judge, "groq-model", path, limit=1)

    assert list(runs) == ["t001"]
    assert runs["t001"].tokens == 300
    assert read_runs(path) == runs
    assert missing(queries, runs, "groq-model") == ["t002"]

    collect(queries, lambda _: ["k01", "k02"], judge, "groq-model", path)
    assert len(asked) == 4  # t001 was not asked again

    changed = [query("t001", "a dessert", filters={"exclude_allergens": ["milk"]}), queries[1]]
    again = collect(changed, lambda _: ["k03"], judge, "groq-model", path)
    assert asked[-1] == ("a dessert, no milk", "k03")
    assert again["t001"].retrieved == ("k03",)
    assert missing(changed, again, "other-model") == ["t001", "t002"]


def test_replay_scores_both_parsers_offline(tmp_path: Path, loaded: CatalogLoad, items: dict[str, MenuItem]) -> None:
    truth = GroundTruth.from_load(loaded)
    with_milk = sorted(item for item in items if "milk" in truth.allergens[item])[:2]
    no_milk = query("t001", "a curry", filters={"exclude_allergens": ["milk"]}, grades={with_milk[0]: 2})
    replies = {with_milk[0]: "Yes.", with_milk[1]: "yes"}
    runs = collect([no_milk], lambda _: with_milk, lambda _, item: Judged(replies[item], 1), "m", tmp_path / "v.jsonl")

    results = replay([no_milk], runs, truth)

    strict, fixed = results["C0_strict"][0], results["C0_fixed"][0]
    assert strict.ranked == (with_milk[1],)  # "Yes." dropped by the original parser
    assert fixed.ranked == tuple(with_milk)
    assert fixed.metrics["violation@3"] == 1.0  # constraints only in the prompt let milk through
    assert fixed.metrics["empty"] == 0.0
    assert strict.metrics["ndcg@3"] == 0.0
    assert fixed.metrics["ndcg@3"] > 0.0
