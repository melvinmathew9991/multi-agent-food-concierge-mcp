"""Scope checks and audit spans for MCP tool calls, as server middleware.

Every ``tools/call`` is checked against the caller's scope before the tool runs. Over HTTP the scope comes from
the request's bearer token (``security.access``); over stdio there is no HTTP request, and the caller is the local
user, who has every scope. A denied call returns an ``isError`` result naming the scope it needs and is logged as
a security event; every call is traced as a span with the tool and the scope, never the arguments.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp_types import CallToolResult, TextContent

from food_concierge.config import Settings
from food_concierge.errors import AppError
from food_concierge.security.access import Scope, check, scope_for
from food_concierge.telemetry import get_telemetry

logger = logging.getLogger(__name__)


class ToolAccess:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def __call__(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        if ctx.method != "tools/call":
            return await call_next(ctx)
        tool = str((ctx.params or {}).get("name", ""))
        request = ctx.request
        try:
            scope = Scope.ADMIN if request is None else scope_for(request.headers.get("authorization"), self._settings)
            check(scope, tool)
        except AppError as error:
            logger.warning("tool call denied", extra={"security_event": error.code, "tool": tool})
            with get_telemetry().observe(f"mcp {tool}", meta={"tool": tool, "denied": error.code}) as span:
                span.update(level="WARNING", status_message=error.code)
            return CallToolResult(content=[TextContent(type="text", text=error.message)], is_error=True)
        with get_telemetry().observe(f"mcp {tool}", meta={"tool": tool, "scope": scope.label}):
            return await call_next(ctx)
