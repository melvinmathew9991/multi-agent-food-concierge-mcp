"""The deploy smoke test's checks, run against the real app in-process; the live run is against a container or the
Space."""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import httpx2
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from food_concierge.api.app import create_app
from food_concierge.config import REPO_ROOT, Settings
from food_concierge.mcp_server.services import Services
from food_concierge.models.embeddings import HashingEmbedder
from food_concierge.services.retrieval import Retriever
from food_concierge.storage.catalog_db import CATALOG_DB, CatalogStore
from food_concierge.storage.indexes import INDEX_DIR, load_indexes

HOST = "testserver"
AGENT_TOKEN = "agent-" + "a" * 40


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_deploy", REPO_ROOT / "scripts" / "smoke_deploy.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_deploy"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app(built_data_dir: Path) -> Iterator[FastAPI]:
    store = CatalogStore(built_data_dir / "processed" / CATALOG_DB)
    settings = Settings(_env_file=None, api_token_agent=SecretStr(AGENT_TOKEN), allowed_hosts=[HOST])
    embedder = HashingEmbedder()
    retriever = Retriever(store, load_indexes(store, embedder, built_data_dir / "processed" / INDEX_DIR), embedder)
    yield create_app(Services.over(store, retriever, settings))  # no vision: the agent check never reaches it
    store.close()


async def test_every_check_passes_against_the_app(smoke: ModuleType, app: FastAPI) -> None:
    def connect(headers: dict[str, str]) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), headers=headers)

    async with app.router.lifespan_context(app):
        checks = await smoke.run_checks(f"http://{HOST}/", connect, AGENT_TOKEN)

    assert [(c.name, c.passed) for c in checks] == [
        ("health", True),
        ("ready", True),
        ("v1 search", True),
        ("mcp tools", True),
        ("wrong token refused", True),
        ("public scope refused", True),
        ("agent scope accepted", True),
    ], checks


async def test_a_server_that_is_down_fails_the_http_checks(smoke: ModuleType) -> None:
    down = httpx2.MockTransport(lambda request: httpx2.Response(503, json={"error": {"code": "not_ready"}}))
    async with httpx2.AsyncClient(transport=down) as http:
        checks = await smoke._http_checks(f"http://{HOST}", http)

    assert not any(check.passed for check in checks[:4])


async def test_an_unreachable_mcp_endpoint_fails_its_checks(smoke: ModuleType) -> None:
    def connect(headers: dict[str, str]) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(502)))

    checks = await smoke.run_checks(f"http://{HOST}", connect, AGENT_TOKEN)

    assert {check.name for check in checks if check.passed} == set()
    assert checks[-1].detail.startswith("MCP failed")


def test_main_reads_the_token_from_the_named_variable(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[object] = []

    async def fake(base_url: str, connect: object, token: str | None) -> list[object]:
        seen.append((base_url, token))
        return [smoke.Check("health", True, "200"), smoke.Check("ready", False, "503")]

    monkeypatch.setattr(smoke, "run_checks", fake)
    monkeypatch.setenv("SMOKE_TOKEN", AGENT_TOKEN)

    assert smoke.main(["http://space.example", "--token-env", "SMOKE_TOKEN"]) == 1
    assert seen == [("http://space.example", AGENT_TOKEN)]
    assert capsys.readouterr().out.splitlines() == ["PASS health: 200", "FAIL ready: 503"]
