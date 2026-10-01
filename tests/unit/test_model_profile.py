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


def test_dataset_is_well_formed() -> None:
    dataset = yaml.safe_load(profile.DATASET.read_text(encoding="utf-8"))
    tool_names = {tool.__name__ for tool in profile.TOOLS}
    routes = set(profile.Route.model_fields["route"].annotation.__args__)

    items = dataset["extraction"] + dataset["routing"] + dataset["tools"]
    assert len(items) == 20
    assert len({item["id"] for item in items}) == 20
    for item in dataset["extraction"]:
        assert set(item["expect"]) == set(profile.MealRequest.model_fields)
    assert {item["expect"] for item in dataset["routing"]} <= routes
    for item in dataset["tools"]:
        assert item["expect"]["name"] in tool_names


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
