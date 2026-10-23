"""Scoring in the model profile script: the numbers in ADR-0006 are only as good as these functions."""

import importlib.util
import sys
from types import ModuleType

import pytest
import yaml

from food_concierge.config import REPO_ROOT


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_profile", REPO_ROOT / "scripts" / "model_profile.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_profile"] = module
    spec.loader.exec_module(module)
    return module


profile = _load()


def test_wilson_interval() -> None:
    assert profile.wilson(0, 0) == (0.0, 0.0)
    assert profile.wilson(20, 20) == (0.839, 1.0)
    assert profile.wilson(10, 20) == (0.299, 0.701)


def test_rates_and_percentiles() -> None:
    assert profile.rate([True, True, False, True]) == {"rate": 0.75, "k": 3, "n": 4, "ci95": profile.wilson(3, 4)}
    assert profile.rate([])["rate"] is None
    values = [float(v) for v in range(1, 21)]
    assert profile.percentile(values, 0.5) == 10
    assert profile.percentile(values, 0.95) == 19
    assert profile.percentile([], 0.5) is None


def _meal(**fields: object) -> object:
    base: dict[str, object] = {
        "diet": "any",
        "max_calories": None,
        "budget_inr": None,
        "exclude_allergens": [],
        "courses": None,
    }
    return profile.MealRequest(**{**base, **fields})


@pytest.mark.parametrize(
    ("fields", "correct"),
    [
        ({"diet": "vegan", "exclude_allergens": ["sesame", "peanuts"]}, True),
        ({"diet": "eggless", "exclude_allergens": ["peanuts", "sesame"]}, True),
        ({"diet": "vegetarian", "exclude_allergens": ["peanuts", "sesame"]}, False),
        ({"diet": "vegan", "exclude_allergens": ["peanuts"]}, False),
        ({"diet": "vegan", "exclude_allergens": ["peanuts", "sesame"], "courses": 2}, False),
    ],
    ids=["exact", "accepted-alternative", "wrong-diet", "missing-allergen", "invented-constraint"],
)
def test_extraction_scoring(fields: dict[str, object], correct: bool) -> None:
    expect = {
        "diet": ["vegan", "eggless"],
        "max_calories": None,
        "budget_inr": None,
        "exclude_allergens": ["peanuts", "sesame"],
        "courses": None,
    }

    assert profile.extraction_correct(_meal(**fields), expect) is correct


@pytest.mark.parametrize(
    ("args", "correct"),
    [
        ({"query": "vegan Biryani", "diet": "Vegan", "allergens": ["sesame", "peanuts"]}, True),
        ({"query": "pulao", "diet": "vegan", "allergens": ["peanuts", "sesame"]}, False),
        ({"query": "biryani", "diet": "vegetarian", "allergens": ["peanuts", "sesame"]}, False),
        ({"query": "biryani", "diet": "vegan", "allergens": "peanuts, sesame"}, False),
        ({"diet": "vegan", "allergens": ["peanuts", "sesame"]}, False),
    ],
    ids=["match", "wrong-query", "wrong-diet", "allergens-not-a-list", "missing-query"],
)
def test_tool_argument_scoring(args: dict[str, object], correct: bool) -> None:
    rules = {"query": {"contains": "biryani"}, "diet": {"equals": "vegan"}, "allergens": {"set": ["peanuts", "sesame"]}}

    assert profile.tool_args_correct(args, rules) is correct


@pytest.mark.parametrize(
    ("call", "scores"),
    [
        ({"name": "get_dish_details", "args": {"dish_id": "D-0600"}}, (True, True)),
        ({"name": "meal_totals", "args": {"dish_ids": ["d-0600"]}}, (True, True)),
        ({"name": "meal_totals", "args": {"dish_ids": ["d-0601"]}}, (True, False)),
        ({"name": "check_allergens", "args": {"dish_id": "d-0600", "allergens": []}}, (False, False)),
        (None, (False, False)),
    ],
    ids=["first-option", "second-option", "right-tool-wrong-args", "wrong-tool", "no-call"],
)
def test_tool_call_scoring_accepts_a_list_of_options(call: dict[str, object] | None, scores: tuple[bool, bool]) -> None:
    expect = [
        {"name": "get_dish_details", "args": {"dish_id": {"equals": "d-0600"}}},
        {"name": "meal_totals", "args": {"dish_ids": {"set": ["d-0600"]}}},
    ]

    assert profile.tool_call_correct(call, expect) == scores


def test_tool_call_scoring_with_one_expectation_accepts_only_that_tool() -> None:
    expect = {"name": "get_dish_details", "args": {"dish_id": {"equals": "d-0600"}}}

    details = {"name": "get_dish_details", "args": {"dish_id": "d-0600"}}
    totals = {"name": "meal_totals", "args": {"dish_ids": ["d-0600"]}}

    assert profile.tool_call_correct(details, expect) == (True, True)
    assert profile.tool_call_correct(totals, expect) == (False, False)


def test_dataset_is_well_formed() -> None:
    dataset = yaml.safe_load(profile.DATASET.read_text(encoding="utf-8"))
    tool_names = {tool.__name__ for tool in profile.TOOLS}
    routes = set(profile.Route.model_fields["route"].annotation.__args__)

    items = dataset["extraction"] + dataset["routing"] + dataset["tools"]
    assert len({item["id"] for item in items}) == len(items) == 74
    assert {item["split"] for item in items} == {"dev", "test"}
    for item in dataset["extraction"]:
        assert set(item["expect"]) == set(profile.MealRequest.model_fields)
    for item in dataset["routing"]:
        expect = item["expect"] if isinstance(item["expect"], list) else [item["expect"]]
        assert set(expect) <= routes
    for item in dataset["tools"]:
        for option in item["expect"] if isinstance(item["expect"], list) else [item["expect"]]:
            assert option["name"] in tool_names
            assert set(option["args"]) <= set(getattr(profile, option["name"]).model_fields)

    # #51: at least 30 each, and the test split covers every route and every tool.
    test = profile.select(dataset, "test")
    assert len(dataset["routing"]) >= 30
    assert len(dataset["tools"]) >= 30
    assert {item["expect"] for item in test["routing"] if isinstance(item["expect"], str)} == routes
    assert {item["expect"]["name"] for item in test["tools"] if isinstance(item["expect"], dict)} == tool_names


@pytest.mark.parametrize(
    ("route", "expect", "correct"),
    [
        ("recommend", "recommend", True),
        ("dish_info", "recommend", False),
        ("dish_info", ["dish_info", "recommend"], True),
        ("smalltalk", ["dish_info", "recommend"], False),
    ],
)
def test_route_scoring_accepts_a_list_for_ambiguous_messages(route: str, expect: object, correct: bool) -> None:
    assert profile.route_correct(route, expect) is correct


def test_select_keeps_only_the_split() -> None:
    dataset = {
        "extraction": [{"id": "ex01", "split": "dev"}],
        "routing": [{"id": "rt01", "split": "dev"}, {"id": "rt02", "split": "test"}],
        "tools": [{"id": "tl01", "split": "test"}],
    }

    assert profile.select(dataset, "test") == {
        "extraction": [],
        "routing": [{"id": "rt02", "split": "test"}],
        "tools": [{"id": "tl01", "split": "test"}],
    }


def test_pacing_wait_is_recorded_so_latency_can_exclude_it() -> None:
    import httpx2

    pacer = profile.Pacer(per_minute=1200)  # one request per 50 ms
    request = httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    pacer.on_request(request)
    pacer.on_request(request)
    pacer.on_response(httpx2.Response(429, headers={"Retry-After": "2", "x-ratelimit-remaining-requests": "9"}))

    assert 0.02 < pacer.waited_s < 0.5
    assert pacer.statuses == {"429": 1}
    assert pacer.headers == {"retry-after": "2", "x-ratelimit-remaining-requests": "9"}


def test_a_busy_refusal_is_retried_and_a_daily_one_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    from food_concierge.errors import BudgetExceededError

    monkeypatch.setattr(profile.time, "sleep", lambda _s: None)
    candidate = object.__new__(profile.Candidate)  # timed() needs only the pacer
    candidate.pacer = profile.Pacer(per_minute=None)
    outcomes: list[object] = [BudgetExceededError(), BudgetExceededError(), "parsed"]

    def call(_config: object) -> object:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert candidate.timed(call)[0::3] == ("parsed", None)

    calls = 0

    def refused(_config: object) -> object:
        nonlocal calls
        calls += 1
        raise BudgetExceededError

    assert candidate.timed(refused)[0::3] == (None, "budget_exceeded")
    assert calls == profile.BUSY_RETRIES + 1
