"""The MCP server, through in-memory MCP sessions on the real catalog, offline (hashing embedder, scripted vision)."""

import base64
import io
import json
import logging
import random
import runpy
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from mcp import Client, MCPError
from mcp.server.mcpserver import MCPServer
from PIL import Image

from food_concierge.config import Settings
from food_concierge.errors import InputValidationError, NotReadyError
from food_concierge.ingestion.taxonomy import Allergen, Cuisine, Diet
from food_concierge.mcp_server import server as server_module
from food_concierge.mcp_server import services as services_module
from food_concierge.mcp_server.server import create_server
from food_concierge.mcp_server.services import Services, build_services
from food_concierge.models.embeddings import HashingEmbedder
from food_concierge.models.fake import ScriptedChatModel
from food_concierge.services.allergens import DISCLAIMER
from food_concierge.services.catalog import CatalogService
from food_concierge.services.constraints import Constraints, dish_violations
from food_concierge.services.retrieval import Hit, RetrievalConfig, Retriever, SearchResult
from food_concierge.services.search import MAX_RESULTS
from food_concierge.services.vision import DESCRIBE_IN_WORDS, VisionService
from food_concierge.storage.catalog_db import CATALOG_DB, CatalogStore, Filters
from food_concierge.storage.indexes import INDEX_DIR, load_indexes

A = Allergen


@pytest.fixture(scope="module")
def store(built_data_dir: Path) -> Iterator[CatalogStore]:
    opened = CatalogStore(built_data_dir / "processed" / CATALOG_DB)
    yield opened
    opened.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None).model_copy(update={"photo_cache_dir": tmp_path / "photos"})


def make_services(store: CatalogStore, settings: Settings, **overrides: Any) -> Services:
    embedder = HashingEmbedder()
    indexes = load_indexes(store, embedder, store_dir(store) / INDEX_DIR)
    return Services.over(store, overrides.pop("searcher", Retriever(store, indexes, embedder)), settings, **overrides)


def store_dir(store: CatalogStore) -> Path:
    (path,) = [Path(row[2]) for row in store._connection.execute("PRAGMA database_list")]
    return path.parent


@pytest.fixture
def server(store: CatalogStore, settings: Settings) -> MCPServer:
    return create_server(make_services(store, settings))


async def call(server: MCPServer, tool: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
    """(is_error, the structured result, or the error text)."""
    async with Client(server) as client:
        result = await client.call_tool(tool, arguments)
    if result.is_error:
        return True, result.content[0].text  # type: ignore[union-attr]
    return False, result.structured_content


def names(result: Any) -> list[str]:
    return [dish["item_id"] for dish in result["dishes"]]


async def test_the_server_lists_its_tools_resources_and_prompts(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        resources = [str(r.uri) for r in (await client.list_resources()).resources]
        templates = [t.uri_template for t in (await client.list_resource_templates()).resource_templates]
        prompts = [p.name for p in (await client.list_prompts()).prompts]

    assert list(tools) == [
        "search_dishes",
        "find_similar",
        "get_dish",
        "check_allergens",
        "meal_totals",
        "describe_image",
    ]
    for tool in tools.values():
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is (tool.name == "describe_image")
        assert tool.output_schema is not None
    search = tools["search_dishes"].input_schema
    assert search["required"] == ["query"]
    assert search["properties"]["k"]["maximum"] == MAX_RESULTS
    assert search["$defs"]["Diet"]["enum"] == [d.value for d in Diet]
    assert resources == ["menu://catalog", "menu://policy/allergens"]
    assert templates == ["menu://dishes/{dish_id}", "menu://dishes/{dish_id}/image"]
    assert prompts == ["plan_meal", "find_similar_dish"]
    assert server.instructions is not None
    assert DISCLAIMER in server.instructions


def _random_arguments(rng: random.Random) -> dict[str, Any]:
    words = ["dairy", "nuts", "gluten", "shellfish", "egg", "soya", "sesame", "fish", "mustard"]
    return {
        "query": rng.choice(["spicy curry", "light breakfast", "sweet dessert", "rice dish", "street food snack"]),
        "diet": rng.choice([None, *(d.value for d in Diet)]),
        "eggless": rng.random() < 0.3,
        "exclude_allergens": rng.sample(words, rng.randint(0, 3)),
        "max_kcal": rng.choice([None, 400, 700]),
        "max_price_inr": rng.choice([None, 200, 400]),
        "cuisines": rng.sample([c.value for c in Cuisine], rng.choice([0, 0, 2])),
        "include_unverified": rng.random() < 0.3,
        "k": MAX_RESULTS,
    }


async def test_search_never_returns_a_dish_that_breaks_the_constraints(server: MCPServer, store: CatalogStore) -> None:
    rng = random.Random(110)
    catalog = CatalogService(store)
    returned = 0
    for _ in range(25):
        arguments = _random_arguments(rng)
        failed, result = await call(server, "search_dishes", arguments)
        assert not failed
        constraints = Constraints.model_validate({k: v for k, v in arguments.items() if k not in {"query", "k"}})
        ids = names(result)
        returned += len(ids)
        assert set(ids) <= set(store.candidate_ids(constraints.filters))
        if ids:
            assert not [v for dish in catalog.get_dishes(ids) for v in dish_violations(dish, constraints)]
    assert returned > 50  # the property was tested on real results, not on empty lists


async def test_search_skips_excluded_ids_and_explains_an_empty_result(server: MCPServer) -> None:
    _, first = await call(server, "search_dishes", {"query": "paneer", "k": 3})
    _, again = await call(server, "search_dishes", {"query": "paneer", "k": 3, "exclude_ids": names(first)})
    _, none = await call(server, "search_dishes", {"query": "paneer", "max_kcal": 1})

    assert len(names(first)) == 3
    assert not set(names(first)) & set(names(again))
    assert len(names(again)) == 3
    assert none == {"dishes": [], "note": "No dish meets all of these constraints; relaxing one may help."}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"query": "x" * 501}, "The query is longer than 500 characters."),
        ({"query": "curry", "exclude_allergens": ["kiwi"]}, "unknown allergen term(s): kiwi"),
        ({"query": "curry", "k": MAX_RESULTS + 1}, "k"),
        ({"query": "curry", "diet": "pescatarian"}, "diet"),
    ],
    ids=["long-query", "unknown-allergen", "k-too-large", "unknown-diet"],
)
async def test_bad_search_arguments_come_back_as_tool_errors(
    server: MCPServer, arguments: dict[str, Any], message: str
) -> None:
    failed, text = await call(server, "search_dishes", arguments)

    assert failed
    assert message in text


class Leaky:
    """A searcher that ignores the filters, as a broken one would."""

    def search(self, query: str, filters: Filters, config: RetrievalConfig) -> SearchResult:
        return SearchResult([Hit("s05", 1.0), Hit("t04", 0.5)])


async def test_a_result_that_breaks_the_constraints_is_dropped_and_logged(
    store: CatalogStore, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    server = create_server(make_services(store, settings, searcher=Leaky()))

    with caplog.at_level(logging.ERROR, logger="food_concierge.services.search"):
        failed, result = await call(server, "search_dishes", {"query": "anything", "exclude_allergens": ["dairy"]})

    assert not failed
    assert names(result) == ["t04"]  # Butter Chicken (milk) is gone
    assert [r.item_ids for r in caplog.records] == [["s05"]]  # type: ignore[attr-defined]


async def test_find_similar_leaves_out_the_same_dish_anywhere(server: MCPServer, store: CatalogStore) -> None:
    keys = store.dish_keys()
    twins = sorted(item_id for item_id, key in keys.items() if key == "paneer-butter-masala")

    failed, result = await call(server, "find_similar", {"dish_id": twins[0], "diet": "vegetarian", "k": 5})

    assert not failed
    assert len(names(result)) == 5
    assert not set(names(result)) & set(twins)
    failed, text = await call(server, "find_similar", {"dish_id": "zz99"})
    assert (failed, text) == (True, "Error executing tool find_similar: No dish has the id zz99.")


async def test_get_dish_returns_the_dish_in_full_or_names_an_unknown_id(server: MCPServer) -> None:
    failed, dish = await call(server, "get_dish", {"dish_id": "S05"})
    _, unknown = await call(server, "get_dish", {"dish_id": "zz99"})

    assert not failed
    assert (dish["name"], dish["price_inr"], dish["photo"]["licence"]) == ("Butter Chicken", 440, "CC BY-SA 4.0")
    assert unknown == "Error executing tool get_dish: No dish has the id zz99."


async def test_check_allergens_reads_plain_words_and_carries_the_disclaimer(server: MCPServer) -> None:
    failed, result = await call(server, "check_allergens", {"dish_ids": ["s05", "t04"], "allergens": ["dairy"]})
    _, everything = await call(server, "check_allergens", {"dish_ids": ["s05"]})

    assert not failed
    assert [(c["item_id"], c["contains"], c["clear"]) for c in result["checks"]] == [
        ("s05", ["milk"], False),
        ("t04", [], True),
    ]
    assert result["disclaimer"] == DISCLAIMER
    assert everything["checks"][0]["contains"] == ["milk", "tree_nuts"]


async def test_meal_totals_counts_a_repeated_dish_twice(server: MCPServer) -> None:
    failed, totals = await call(server, "meal_totals", {"dish_ids": ["s05", "s05"]})

    assert not failed
    assert (totals["kcal"], totals["price_inr"]) == (1020, 880)


def jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "orange").save(buffer, format="JPEG")
    return buffer.getvalue()


async def test_describe_image_describes_an_inline_photo(store: CatalogStore, settings: Settings) -> None:
    vision = VisionService(ScriptedChatModel(script=[AIMessage(content="An orange curry.")]), "vision", settings)
    server = create_server(make_services(store, settings, vision=vision))

    failed, result = await call(server, "describe_image", {"image_base64": base64.b64encode(jpeg()).decode()})

    assert not failed
    assert result["text"] == "An orange curry."


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ("not base64!", "The image is not valid base64."),
        ("A" * 8_000_000, "The image is larger than 5 MB."),
    ],
    ids=["not-base64", "too-large"],
)
async def test_describe_image_refuses_bad_input_before_decoding(
    store: CatalogStore, settings: Settings, image: str, message: str
) -> None:
    vision = VisionService(ScriptedChatModel(script=[]), "vision", settings)
    server = create_server(make_services(store, settings, vision=vision))

    failed, text = await call(server, "describe_image", {"image_base64": image})

    assert failed
    assert text.endswith(message)


async def test_without_a_vision_model_photos_are_answered_with_a_request_for_words(server: MCPServer) -> None:
    failed, text = await call(server, "describe_image", {"image_base64": base64.b64encode(jpeg()).decode()})

    assert failed
    assert text.endswith(DESCRIBE_IN_WORDS)


async def test_the_catalog_and_a_dish_are_readable_as_resources(server: MCPServer) -> None:
    async with Client(server) as client:
        catalog = json.loads((await client.read_resource("menu://catalog")).contents[0].text)  # type: ignore[union-attr]
        dish = json.loads((await client.read_resource("menu://dishes/s05")).contents[0].text)  # type: ignore[union-attr]
        with pytest.raises(MCPError, match="No dish has the id zz99"):
            await client.read_resource("menu://dishes/zz99")
        with pytest.raises(MCPError, match="Give at least one dish id"):
            await client.read_resource("menu://dishes/%20")

    assert (catalog["currency"], catalog["disclaimer"], len(catalog["dishes"])) == ("INR", DISCLAIMER, 150)
    assert "ingredients" not in catalog["dishes"][0]  # the short form
    assert (dish["name"], dish["ingredients"][0]) == ("Butter Chicken", "tandoori chicken")


async def test_a_dish_photo_is_served_from_the_thumbnail_cache(server: MCPServer, settings: Settings) -> None:
    async with Client(server) as client:
        with pytest.raises(MCPError, match="isn't stored on this server"):
            await client.read_resource("menu://dishes/s05/image")
        dish = json.loads((await client.read_resource("menu://dishes/s05")).contents[0].text)  # type: ignore[union-attr]
        thumbnail = settings.photo_cache_dir / "thumbs" / f"{dish['photo']['sha256']}.webp"
        thumbnail.parent.mkdir(parents=True)
        thumbnail.write_bytes(b"webp bytes")
        (content,) = (await client.read_resource("menu://dishes/s05/image")).contents
        with pytest.raises(MCPError, match="Dish a01 has no photo"):
            await client.read_resource("menu://dishes/a01/image")

    assert content.mime_type == "image/webp"
    assert base64.b64decode(content.blob) == b"webp bytes"  # type: ignore[union-attr]


async def test_the_allergen_policy_lists_every_allergen_with_its_words(server: MCPServer) -> None:
    async with Client(server) as client:
        policy = json.loads((await client.read_resource("menu://policy/allergens")).contents[0].text)  # type: ignore[union-attr]

    by_key = {entry["key"]: entry for entry in policy["allergens"]}
    assert list(by_key) == [a.value for a in Allergen]
    assert "dairy" in by_key["milk"]["words"]
    assert (by_key["wheat"]["eu_14"], by_key["wheat"]["us_big_9"]) == (False, True)
    assert set(policy["levels"]) == {"contains", "may_contain", "unverified"}
    assert policy["disclaimer"] == DISCLAIMER


async def test_the_prompts_carry_the_request_and_the_rules(server: MCPServer) -> None:
    async with Client(server) as client:
        plan = await client.get_prompt("plan_meal", {"constraints": "vegan, no peanuts, under ₹500"})
        similar = await client.get_prompt("find_similar_dish", {"description": "a crispy rice pancake"})

    plan_text = plan.messages[0].content.text  # type: ignore[union-attr]
    assert "vegan, no peanuts, under ₹500" in plan_text
    assert "search_dishes" in plan_text
    assert plan_text.endswith(DISCLAIMER)
    assert "a crispy rice pancake" in similar.messages[0].content.text  # type: ignore[union-attr]


def test_services_are_built_from_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build_data: Callable[..., None]
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, embed_provider="fake")
    build_data(tmp_path, settings.embed_query_prefix)

    built = build_services(settings)
    assert built.vision is None  # no vision model configured in tests
    built.store.close()

    sentinel = object()
    monkeypatch.setattr(services_module, "build_vision_service", lambda _settings: sentinel)
    with_vision = build_services(settings)
    assert with_vision.vision is sentinel
    with_vision.store.close()

    with pytest.raises(NotReadyError):
        build_services(Settings(_env_file=None, data_dir=tmp_path / "empty", embed_provider="fake"))


def test_the_entry_point_runs_the_server_on_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    class Stub:
        def run(self, transport: str) -> None:
            ran.append(transport)

    monkeypatch.setattr(services_module, "build_services", lambda _settings: object())
    monkeypatch.setattr(server_module, "create_server", lambda _services: Stub())
    monkeypatch.delitem(sys.modules, "food_concierge.mcp_server.__main__", raising=False)

    runpy.run_module("food_concierge.mcp_server", run_name="__main__")

    assert ran == ["stdio"]


@pytest.mark.parametrize("k", [0, MAX_RESULTS + 1])
def test_the_search_service_bounds_k_itself(store: CatalogStore, settings: Settings, k: int) -> None:
    # The tool and the API validate k already; the service checks again, for any other caller.
    with pytest.raises(InputValidationError, match=r"Ask for 1 to 10 dishes\."):
        make_services(store, settings).search.search("curry", Constraints(), k)
