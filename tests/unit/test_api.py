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
from food_concierge.api.app import create_app
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


@pytest.fixture(scope="module")
def store(built_data_dir: Path) -> Iterator[CatalogStore]:
    opened = CatalogStore(built_data_dir / "processed" / CATALOG_DB)
    yield opened
    opened.close()


def make_app(store: CatalogStore, tmp_path: Path, **overrides: Any) -> FastAPI:
    settings = Settings(
        _env_file=None,
        api_token_agent=SecretStr("agent-token"),
        api_token_admin=SecretStr("admin-token"),
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
            "/v1/search", json={"query": "dosa"}, headers={"Authorization": "Bearer agent-token"}
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


@pytest.mark.parametrize("token", ["agent-token", "admin-token"])
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
    monkeypatch.setattr("uvicorn.run", lambda app, host, port, log_config: served.append((app, host, port)))
    monkeypatch.delitem(sys.modules, "food_concierge.api.__main__", raising=False)

    runpy.run_module("food_concierge.api", run_name="__main__")

    assert served == [(sentinel, "127.0.0.1", 8000)]
