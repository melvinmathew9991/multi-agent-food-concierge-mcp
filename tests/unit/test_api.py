"""The HTTP server, in-process: health, readiness, /v1/search, MCP over Streamable HTTP, tokens and rate limits."""

import base64
import io
import logging
import runpy
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from PIL import Image
from pydantic import SecretStr

from food_concierge.api import app as app_module
from food_concierge.api.app import caller_address, create_app
from food_concierge.config import Settings
from food_concierge.mcp_server import services as services_module
from food_concierge.mcp_server.services import Services
from food_concierge.models.embeddings import HashingEmbedder
from food_concierge.models.fake import ScriptedChatModel
from food_concierge.services.catalog import CatalogService
from food_concierge.services.constraints import Constraints, dish_violations
from food_concierge.services.retrieval import Retriever
from food_concierge.services.vision import VisionService
from food_concierge.storage.catalog_db import CATALOG_DB, CatalogStore
from food_concierge.storage.indexes import INDEX_DIR, load_indexes

HOST = "testserver"
AGENT_TOKEN = "agent-" + "a" * 40
ADMIN_TOKEN = "admin-" + "b" * 40


@pytest.fixture(scope="module")
def store(built_data_dir: Path) -> Iterator[CatalogStore]:
    opened = CatalogStore(built_data_dir / "processed" / CATALOG_DB)
    yield opened
    opened.close()


def make_app(store: CatalogStore, tmp_path: Path, **overrides: Any) -> FastAPI:
    settings = Settings(
        _env_file=None,
        api_token_agent=SecretStr(AGENT_TOKEN),
        api_token_admin=SecretStr(ADMIN_TOKEN),
        allowed_hosts=[HOST],
        photo_cache_dir=tmp_path,
        **overrides,
    )
    embedder = HashingEmbedder()
    retriever = Retriever(store, load_indexes(store, embedder, store_dir(store) / INDEX_DIR), embedder)
    vision = VisionService(ScriptedChatModel(script=[AIMessage(content="A golden fried pastry.")]), "v", settings)
    return create_app(Services.over(store, retriever, settings, vision=vision))


def store_dir(store: CatalogStore) -> Path:
    (path,) = [Path(row[2]) for row in store._connection.execute("PRAGMA database_list")]
    return path.parent


@asynccontextmanager
async def running(app: FastAPI) -> AsyncIterator[httpx2.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url=f"http://{HOST}") as http:
            yield http


@asynccontextmanager
async def mcp_client(app: FastAPI, token: str | None = None) -> AsyncIterator[Client]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        http = httpx2.AsyncClient(transport=transport, base_url=f"http://{HOST}", headers=headers)
        async with Client(streamable_http_client(f"http://{HOST}/mcp", http_client=http)) as client:
            yield client


async def test_health_and_readiness(store: CatalogStore, tmp_path: Path) -> None:
    async with running(make_app(store, tmp_path)) as http:
        health = await http.get("/health")
        ready = await http.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.json() == {"status": "ready", "dishes": 150}


async def test_v1_search_applies_the_constraints(store: CatalogStore, tmp_path: Path) -> None:
    body = {"query": "dosa", "diet": "vegan", "exclude_allergens": ["peanuts"], "max_price_inr": 200, "k": 5}
    async with running(make_app(store, tmp_path)) as http:
        response = await http.post("/v1/search", json=body)
        again = await http.post(
            "/v1/search", json={**body, "exclude_ids": [d["item_id"] for d in response.json()["dishes"]]}
        )

    assert response.status_code == 200
    ids = [dish["item_id"] for dish in response.json()["dishes"]]
    assert ids
    constraints = Constraints(diet="vegan", exclude_allergens=["peanuts"], max_price_inr=200)  # type: ignore[arg-type]
    assert not [v for dish in CatalogService(store).get_dishes(ids) for v in dish_violations(dish, constraints)]
    assert not set(ids) & {dish["item_id"] for dish in again.json()["dishes"]}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"query": "dosa", "k": 11}, "Check these fields: k."),
        ({"query": "dosa", "spicy": True}, "Check these fields: spicy."),
        ({"query": "dosa", "exclude_allergens": ["kiwi"]}, "unknown allergen term(s): kiwi"),
        ({"query": "   "}, "The query is empty."),
    ],
    ids=["k", "unknown-field", "unknown-allergen", "blank-query"],
)
async def test_v1_search_errors_share_one_shape(
    store: CatalogStore, tmp_path: Path, body: dict[str, Any], message: str
) -> None:
    async with running(make_app(store, tmp_path)) as http:
        response = await http.post("/v1/search", json=body)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_input"
    assert message in error["message"]
    assert error["request_id"] == response.headers["x-request-id"]


async def test_a_bad_token_is_refused_before_search_or_mcp(store: CatalogStore, tmp_path: Path) -> None:
    async with running(make_app(store, tmp_path)) as http:
        search = await http.post("/v1/search", json={"query": "dosa"}, headers={"Authorization": "Bearer guessed"})
        mcp = await http.post("/mcp", json={}, headers={"Authorization": "Basic abc"})
        health = await http.get("/health", headers={"Authorization": "Bearer guessed"})

    assert (search.status_code, search.headers["www-authenticate"]) == (401, "Bearer")
    assert search.json()["error"]["code"] == "unauthorized"
    assert mcp.status_code == 401
    assert health.status_code == 200  # health is not behind the gate


async def test_callers_over_their_rate_limit_are_told_when_to_retry(store: CatalogStore, tmp_path: Path) -> None:
    async with running(make_app(store, tmp_path, rate_limit_per_min=2)) as http:
        statuses = [(await http.post("/v1/search", json={"query": "dosa"})).status_code for _ in range(3)]
        limited = await http.post("/v1/search", json={"query": "dosa"})
        with_token = await http.post(
            "/v1/search", json={"query": "dosa"}, headers={"Authorization": f"Bearer {AGENT_TOKEN}"}
        )

    assert statuses == [200, 200, 429]
    assert limited.json()["error"]["code"] == "rate_limited"
    assert limited.headers["retry-after"] == "30"
    assert with_token.status_code == 200  # a token is its own caller


async def test_anonymous_mcp_callers_get_the_public_tools_only(
    store: CatalogStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="food_concierge.mcp_server.access"):
        async with mcp_client(make_app(store, tmp_path)) as client:
            tools = await client.list_tools()
            found = await client.call_tool("search_dishes", {"query": "idli", "k": 2})
            denied = await client.call_tool("describe_image", {"image_base64": photo()})

    assert len(tools.tools) == 6
    assert not found.is_error
    assert denied.is_error
    assert denied.content[0].text == "describe_image needs the agent scope; this request has public."  # type: ignore[union-attr]
    assert [(r.security_event, r.tool) for r in caplog.records] == [("forbidden", "describe_image")]  # type: ignore[attr-defined]


@pytest.mark.parametrize("token", [AGENT_TOKEN, ADMIN_TOKEN], ids=["agent", "admin"])
async def test_an_agent_or_admin_token_may_describe_photos(store: CatalogStore, tmp_path: Path, token: str) -> None:
    async with mcp_client(make_app(store, tmp_path), token) as client:
        described = await client.call_tool("describe_image", {"image_base64": photo()})

    assert not described.is_error
    assert described.structured_content["text"] == "A golden fried pastry."


async def test_mcp_answers_only_allowed_hosts(store: CatalogStore, tmp_path: Path) -> None:
    app = make_app(store, tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://evil.example") as http:
            response = await http.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Accept": "application/json, text/event-stream"},
            )

    assert response.status_code == 421


def photo() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "goldenrod").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_the_entry_point_serves_the_app_with_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[tuple[object, str, int]] = []
    sentinel = object()
    monkeypatch.setattr(services_module, "build_services", lambda _settings: object())
    monkeypatch.setattr(app_module, "create_app", lambda _services: sentinel)

    def run(app: object, host: str, port: int, log_config: None, proxy_headers: bool) -> None:
        assert proxy_headers is False  # X-Forwarded-For is read by AccessGate, per TRUSTED_PROXY_HOPS
        served.append((app, host, port))

    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.delitem(sys.modules, "food_concierge.api.__main__", raising=False)

    runpy.run_module("food_concierge.api", run_name="__main__")

    assert served == [(sentinel, "127.0.0.1", 8000)]


def forwarded(*hops: str) -> dict[str, str]:
    return {"X-Forwarded-For": ", ".join(hops)}


async def test_a_forged_forwarded_header_gets_no_fresh_allowance(store: CatalogStore, tmp_path: Path) -> None:
    # Audit H3: with no trusted proxy, a client that writes a new address per request must still be one caller.
    async with running(make_app(store, tmp_path, rate_limit_per_min=2)) as http:
        statuses = [
            (await http.post("/v1/search", json={"query": "dosa"}, headers=forwarded(f"10.0.0.{n}"))).status_code
            for n in range(3)
        ]

    assert statuses == [200, 200, 429]


async def test_behind_a_trusted_proxy_each_visitor_is_its_own_caller(store: CatalogStore, tmp_path: Path) -> None:
    app = make_app(store, tmp_path, rate_limit_per_min=1, trusted_proxy_hops=1)
    async with running(app) as http:
        first = await http.post("/v1/search", json={"query": "dosa"}, headers=forwarded("203.0.113.7"))
        other = await http.post("/v1/search", json={"query": "dosa"}, headers=forwarded("198.51.100.2"))
        # The visitor forges an earlier hop; the proxy still appends the address it saw, which is what counts.
        forged = await http.post("/v1/search", json={"query": "dosa"}, headers=forwarded("1.2.3.4", "203.0.113.7"))

    assert (first.status_code, other.status_code, forged.status_code) == (200, 200, 429)


@pytest.mark.parametrize(
    ("hops", "headers", "address"),
    [
        (0, [("x-forwarded-for", "10.0.0.1")], "ip:testclient"),
        (1, [("x-forwarded-for", "1.2.3.4, 10.0.0.1")], "ip:10.0.0.1"),
        (2, [("x-forwarded-for", "1.2.3.4, 10.0.0.1, 10.0.0.2")], "ip:10.0.0.1"),
        (1, [("x-forwarded-for", "1.2.3.4"), ("x-forwarded-for", "10.0.0.9")], "ip:10.0.0.9"),
        (2, [("x-forwarded-for", "10.0.0.1")], "ip:testclient"),  # fewer hops than proxies: not through them
        (1, [], "ip:testclient"),
    ],
)
def test_the_caller_address_is_the_one_a_trusted_proxy_recorded(
    hops: int, headers: list[tuple[str, str]], address: str
) -> None:
    scope = {
        "type": "http",
        "client": ("testclient", 50000),
        "headers": [(name.encode(), value.encode()) for name, value in headers],
    }
    assert caller_address(scope, hops) == address
    assert caller_address({"type": "http", "headers": []}, hops) == "ip:unknown"


async def test_anonymous_callers_cant_take_the_capacity_token_holders_need(store: CatalogStore, tmp_path: Path) -> None:
    app = make_app(
        store,
        tmp_path,
        rate_limit_per_min=5,
        rate_limit_global_per_min=10,
        rate_limit_anonymous_per_min=2,
        trusted_proxy_hops=1,
    )
    async with running(app) as http:
        visitors = [
            (await http.post("/v1/search", json={"query": "dosa"}, headers=forwarded(f"203.0.113.{n}"))).status_code
            for n in range(3)
        ]
        agent = await http.post(
            "/v1/search", json={"query": "dosa"}, headers={"Authorization": f"Bearer {AGENT_TOKEN}"}
        )

    assert visitors == [200, 200, 429]
    assert agent.status_code == 200


def test_production_without_an_agent_token_warns(
    store: CatalogStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(_env_file=None, environment="production", allowed_hosts=[HOST], photo_cache_dir=tmp_path)
    embedder = HashingEmbedder()
    retriever = Retriever(store, load_indexes(store, embedder, store_dir(store) / INDEX_DIR), embedder)
    with caplog.at_level(logging.WARNING, logger="food_concierge.api.app"):
        create_app(Services.over(store, retriever, settings))

    assert [r.message for r in caplog.records] == ["no API_TOKEN_AGENT set: describe_image can't be called over HTTP"]


async def test_wrong_tokens_spend_the_address_allowance_so_guessing_is_slow(
    store: CatalogStore, tmp_path: Path
) -> None:
    guess = {"Authorization": "Bearer " + "g" * 43}
    async with running(make_app(store, tmp_path, rate_limit_per_min=2)) as http:
        guesses = [(await http.post("/mcp", json={}, headers=guess)).status_code for _ in range(3)]
        anonymous = await http.post("/v1/search", json={"query": "dosa"})
        with_token = await http.post(
            "/v1/search", json={"query": "dosa"}, headers={"Authorization": f"Bearer {AGENT_TOKEN}"}
        )

    assert guesses == [401, 401, 429]
    assert anonymous.status_code == 429  # the same address, its allowance spent on guesses
    assert with_token.status_code == 200  # a real token has its own bucket
