"""Who may call what over HTTP: bearer tokens map to scopes, and each tool needs one (architecture §16).

- ``public``: search and the read-only tools and resources. A request without a token is public, so anyone can
  try the demo; the rate limits keep that affordable.
- ``agent``: also ``describe_image``, which spends the hosted vision quota, and (from P5) agent runs.
- ``admin``: everything, including the ingestion flows when they are exposed.

Scopes are ordered: a higher one includes the lower ones. A token that matches no configured scope is refused,
not treated as public, so a mistyped secret fails loudly. Tokens are compared as SHA-256 digests in constant
time, so neither timing nor length tells a guess how close it came, and settings refuse tokens under 32 characters
and one token for two scopes. Wrong tokens count against the caller's rate limit (`api.app.AccessGate`), so
they can't be guessed at speed. Over stdio the server runs as the local user's own process, so every scope
applies.
"""

from __future__ import annotations

import hashlib
import hmac
from enum import IntEnum

from food_concierge.config import Settings
from food_concierge.errors import AppError


class Scope(IntEnum):
    PUBLIC = 1
    AGENT = 2
    ADMIN = 3

    @property
    def label(self) -> str:
        return self.name.lower()


# The scope each tool needs; a tool missing here needs admin, so a new tool is closed until it is listed.
TOOL_SCOPES: dict[str, Scope] = {
    "search_dishes": Scope.PUBLIC,
    "find_similar": Scope.PUBLIC,
    "get_dish": Scope.PUBLIC,
    "check_allergens": Scope.PUBLIC,
    "meal_totals": Scope.PUBLIC,
    "describe_image": Scope.AGENT,
}


class UnauthorizedError(AppError):
    code = "unauthorized"
    http_status = 401
    default_message = "The access token is not valid."


class ForbiddenError(AppError):
    code = "forbidden"
    http_status = 403
    default_message = "This needs a token with a higher scope."


def required_scope(tool: str) -> Scope:
    return TOOL_SCOPES.get(tool, Scope.ADMIN)


def scope_for(authorization: str | None, settings: Settings) -> Scope:
    """The scope an ``Authorization`` header grants: none is public; raises ``UnauthorizedError`` for a bad one."""
    if authorization is None or not authorization.strip():
        return Scope.PUBLIC
    scheme, _, token = authorization.strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedError("Send the token as 'Authorization: Bearer <token>'.")
    offered = hashlib.sha256(token.encode()).digest()
    granted: Scope | None = None
    for scope, secret in (
        (Scope.ADMIN, settings.api_token_admin),
        (Scope.AGENT, settings.api_token_agent),
        (Scope.PUBLIC, settings.api_token_public),
    ):
        # Every configured token is compared, so the time taken doesn't tell which one nearly matched.
        if secret is not None and secret.get_secret_value():
            matched = hmac.compare_digest(offered, hashlib.sha256(secret.get_secret_value().encode()).digest())
            if matched and granted is None:
                granted = scope
    if granted is None:
        raise UnauthorizedError()
    return granted


def check(scope: Scope, tool: str) -> None:
    """Raise ``ForbiddenError`` unless ``scope`` may call ``tool``."""
    needed = required_scope(tool)
    if scope < needed:
        raise ForbiddenError(f"{tool} needs the {needed.label} scope; this request has {scope.label}.")
