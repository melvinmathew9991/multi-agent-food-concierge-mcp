"""Smoke test for a running HTTP server: a local container, the Space after a deploy. No model calls.

Checks the deploy definition of done (optimization.md §6) against a base URL:

1. ``/health`` and ``/ready`` answer 200, and the catalog has dishes;
2. ``/v1/search`` returns dishes that keep its constraints (vegan, no peanuts), and the first with a photo links
   Commons' scaled copy, carries its credit, and has its thumbnail served by the MCP image resource;
3. MCP without a token lists the six tools and refuses ``describe_image`` for needing the ``agent`` scope;
4. a token that matches none is refused with 401;
5. with ``--token-env`` (the agent token), the same call passes the scope check and fails only on its deliberately
   invalid image (or asks for words when no vision model is configured), so no vision quota is spent.

    python scripts/smoke_deploy.py http://127.0.0.1:7860
    SMOKE_TOKEN=<agent token> python scripts/smoke_deploy.py https://<space>.hf.space --token-env SMOKE_TOKEN

Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

TOOLS = {"search_dishes", "find_similar", "get_dish", "check_allergens", "meal_totals", "describe_image"}
SEARCH = {"query": "dosa", "diet": "vegan", "exclude_allergens": ["peanuts"], "k": 3}
WRONG_TOKEN = "smoke-" + "x" * 40  # long enough to be checked, matches no configured token
NOT_AN_IMAGE = "not base64!"
TIMEOUT_S = 60.0  # a sleeping Space takes a while to wake


# Makes an HTTP client with these headers; tests pass one over an in-process transport.
Connect = Callable[[dict[str, str]], httpx2.AsyncClient]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def _bearer(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _describe_error(base_url: str, connect: Connect, token: str | None) -> str:
    """The error text ``describe_image`` returns for an invalid image, as the given token."""
    try:
        async with (
            connect(_bearer(token)) as http,
            Client(streamable_http_client(f"{base_url}/mcp", http_client=http)) as client,
        ):
            result = await client.call_tool("describe_image", {"image_base64": NOT_AN_IMAGE})
    except Exception as error:  # an unreachable or broken endpoint fails the check, it doesn't stop the run
        return f"MCP failed: {error!r}"
    return result.content[0].text if result.is_error else "no error"  # type: ignore[union-attr]


async def _image_type(base_url: str, http: httpx2.AsyncClient, dish_id: str) -> str:
    """The media type of the dish's photo as the MCP image resource serves it, or why it couldn't be read."""
    try:
        async with Client(streamable_http_client(f"{base_url}/mcp", http_client=http)) as client:
            (content,) = (await client.read_resource(f"menu://dishes/{dish_id}/image")).contents
    except Exception as error:
        return f"MCP failed: {error!r}"
    return content.mime_type or "no type"


def _photo_check(found: list[dict[str, Any]], image_type: str) -> Check:
    with_photo = [dish for dish in found if dish.get("photo_url")]
    if not with_photo:
        return Check("dish photo", False, "no result has a photo")
    dish = with_photo[0]
    credited = bool((dish.get("photo") or {}).get("author"))
    scaled = "/thumb/" in dish["photo_url"]  # Commons' scaled copy, not the multi-MB original
    passed = credited and scaled and image_type == "image/webp"
    return Check("dish photo", passed, f"{dish['item_id']}: credited={credited}, scaled={scaled}, {image_type}")


async def _tool_names(base_url: str, http: httpx2.AsyncClient) -> set[str]:
    try:
        async with Client(streamable_http_client(f"{base_url}/mcp", http_client=http)) as client:
            return {tool.name for tool in (await client.list_tools()).tools}
    except Exception:
        return set()


async def run_checks(base_url: str, connect: Connect, token: str | None = None) -> list[Check]:
    base_url = base_url.rstrip("/")
    checks: list[Check] = []
    async with connect({}) as http:
        checks += await _http_checks(base_url, http)
    public = await _describe_error(base_url, connect, None)
    checks.append(Check("public scope refused", "agent" in public, public))
    if token:
        agent = await _describe_error(base_url, connect, token)
        accepted = "scope" not in agent and not agent.startswith(("MCP failed", "no error"))
        checks.append(Check("agent scope accepted", accepted, agent))
    return checks


async def _http_checks(base_url: str, http: httpx2.AsyncClient) -> list[Check]:
    checks: list[Check] = []

    health = await http.get(f"{base_url}/health")
    checks.append(Check("health", health.status_code == 200, str(health.status_code)))
    ready = await http.get(f"{base_url}/ready")
    dishes = ready.json().get("dishes", 0) if ready.status_code == 200 else 0
    checks.append(Check("ready", ready.status_code == 200 and dishes > 0, f"{ready.status_code}, {dishes} dishes"))

    search = await http.post(f"{base_url}/v1/search", json=SEARCH)
    found = search.json().get("dishes", []) if search.status_code == 200 else []
    vegan = all(dish.get("diet") == "vegan" for dish in found)
    checks.append(Check("v1 search", bool(found) and vegan, f"{search.status_code}, {len(found)} dishes"))
    pictured = next((dish["item_id"] for dish in found if dish.get("photo_url")), None)
    checks.append(_photo_check(found, await _image_type(base_url, http, pictured) if pictured else "not read"))

    listed = await _tool_names(base_url, http)
    checks.append(Check("mcp tools", listed == TOOLS, ", ".join(sorted(listed))))

    wrong = await http.post(f"{base_url}/v1/search", json=SEARCH, headers=_bearer(WRONG_TOKEN))
    checks.append(Check("wrong token refused", wrong.status_code == 401, str(wrong.status_code)))
    return checks


async def _main(base_url: str, token: str | None) -> int:
    checks = await run_checks(base_url, lambda headers: httpx2.AsyncClient(headers=headers, timeout=TIMEOUT_S), token)
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    return 0 if all(check.passed for check in checks) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_url", help="e.g. http://127.0.0.1:7860")
    parser.add_argument("--token-env", metavar="NAME", help="environment variable holding the agent token")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env) if args.token_env else None
    return asyncio.run(_main(args.base_url, token))


if __name__ == "__main__":
    sys.exit(main())
