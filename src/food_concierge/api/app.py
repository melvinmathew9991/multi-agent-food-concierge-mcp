"""The HTTP server: health, readiness, retrieval-only search, and MCP over Streamable HTTP (architecture §8).

- ``GET /health``: the process is up. ``GET /ready``: the catalog and indexes are loaded and consistent (they are
  checked against each other when the services are built, so a server that started is ready).
- ``POST /v1/search``: search with no LLM, under the same constraints and the same safety re-check as the MCP tool.
- ``/mcp``: the MCP server over Streamable HTTP, stateless with JSON responses, so any replica can answer any
  request and free hosts need no sticky sessions. Host headers are checked against ``allowed_hosts`` (DNS
  rebinding protection), and the body limit fits a photo at ``max_image_mb`` in base64.

Before ``/mcp`` and ``/v1/``, ``AccessGate`` refuses a bad token (401) and a caller over its rate limit (429). An
anonymous caller is known by its address: the connecting one, or behind ``trusted_proxy_hops`` proxies the one the
outermost trusted proxy recorded in ``X-Forwarded-For``, never an entry the client could have written itself.
Which tools a scope may call is checked per call inside MCP (``mcp_server.access``). Errors share one shape:
``{"error": {"code", "message", "request_id"}}``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from food_concierge import __version__
from food_concierge.config import Settings
from food_concierge.errors import AppError, InputValidationError
from food_concierge.ingestion.taxonomy import Cuisine, Diet
from food_concierge.mcp_server.server import create_server
from food_concierge.mcp_server.services import Services
from food_concierge.security.access import UnauthorizedError, scope_for
from food_concierge.security.rate_limit import RateLimitedError, RateLimiter
from food_concierge.services.catalog import MAX_DISHES, dish_ids
from food_concierge.services.constraints import parse_constraints
from food_concierge.services.search import MAX_RESULTS, Dishes

logger = logging.getLogger(__name__)

PROTECTED = ("/mcp", "/v1/")
BODY_HEADROOM = 64 * 1024  # JSON-RPC framing around a base64 photo


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1)]
    diet: Diet | None = None
    eggless: bool = False
    exclude_allergens: Annotated[list[str], Field(max_length=20)] = []
    max_kcal: Annotated[int | None, Field(gt=0)] = None
    max_price_inr: Annotated[int | None, Field(gt=0)] = None
    cuisines: Annotated[list[Cuisine], Field(max_length=10)] = []
    include_unverified: bool = False
    exclude_ids: Annotated[list[str], Field(max_length=MAX_DISHES)] = []
    k: Annotated[int, Field(ge=1, le=MAX_RESULTS)] = 5


def caller_address(scope: Scope, trusted_hops: int) -> str:
    """The client's address: each trusted proxy appends the address it saw, so the ``trusted_hops``-th entry from
    the right was written by a proxy we trust; entries left of it may be forged. With too few entries the request
    didn't come through the proxies, so the connecting address is used."""
    client = scope.get("client")
    address = client[0] if client else "unknown"
    if trusted_hops:
        forwarded = ",".join(Headers(scope=scope).getlist("x-forwarded-for"))
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        if len(hops) >= trusted_hops:
            address = hops[-trusted_hops]
    return f"ip:{address}"


def error_response(error: AppError) -> JSONResponse:
    request_id = uuid.uuid4().hex
    headers = {"X-Request-ID": request_id}
    if isinstance(error, UnauthorizedError):
        headers["WWW-Authenticate"] = "Bearer"
    if isinstance(error, RateLimitedError):
        headers["Retry-After"] = str(max(1, round(error.retry_after_s)))
    body = {"error": {"code": error.code, "message": error.message, "request_id": request_id}}
    return JSONResponse(body, status_code=error.http_status, headers=headers)


class AccessGate:
    """Refuses bad tokens and over-limit callers before ``/mcp`` and ``/v1/``, as JSON errors."""

    def __init__(self, app: ASGIApp, settings: Settings, limiter: RateLimiter) -> None:
        self._app = app
        self._settings = settings
        self._limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope["path"]).startswith(PROTECTED):
            await self._app(scope, receive, send)
            return
        authorization = Headers(scope=scope).get("authorization")
        address = caller_address(scope, self._settings.trusted_proxy_hops)
        try:
            try:
                granted = scope_for(authorization, self._settings)
            except UnauthorizedError:
                # A wrong token spends the address's allowance, so tokens can't be guessed at speed.
                self._limiter.acquire(address, anonymous=True)
                raise
            if authorization:
                self._limiter.acquire(f"token:{granted.label}")
            else:
                self._limiter.acquire(address, anonymous=True)
        except AppError as error:
            logger.warning("request refused", extra={"security_event": error.code, "path": scope["path"]})
            await error_response(error)(scope, receive, send)
            return
        await self._app(scope, receive, send)


def create_app(services: Services) -> FastAPI:
    settings = services.settings
    mcp = create_server(services)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=(settings.max_image_bytes + 2) // 3 * 4 + BODY_HEADROOM,
        transport_security=TransportSecuritySettings(
            allowed_hosts=settings.allowed_hosts, allowed_origins=settings.cors_origins
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        async with mcp.session_manager.run():  # a mounted app's own lifespan doesn't run
            yield

    app = FastAPI(title="Food Concierge", version=__version__, lifespan=lifespan)
    limiter = RateLimiter(
        settings.rate_limit_per_min, settings.rate_limit_global_per_min, settings.rate_limit_anonymous_per_min
    )
    if settings.environment == "production" and settings.api_token_agent is None:
        logger.warning("no API_TOKEN_AGENT set: describe_image can't be called over HTTP")
    app.add_middleware(AccessGate, settings=settings, limiter=limiter)

    @app.exception_handler(AppError)
    async def app_error(_: Request, error: AppError) -> JSONResponse:
        return error_response(error)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(part) for part in e["loc"][1:]) or "body" for e in error.errors()})
        return error_response(InputValidationError(f"Check these fields: {', '.join(fields)}."))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        return {"status": "ready", "dishes": len(services.store.doc_texts())}

    @app.post("/v1/search")
    def search(request: SearchRequest) -> Dishes:
        constraints = parse_constraints(request.model_dump(exclude={"query", "exclude_ids", "k"}))
        skip = set(dish_ids(request.exclude_ids)) if request.exclude_ids else set()
        return services.search.search(request.query, constraints, request.k, skip)

    app.mount("/", mcp_app)  # serves /mcp; FastAPI's own routes match first
    return app
