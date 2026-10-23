"""Access control: tokens to scopes, the scope each tool needs, and token-bucket rate limits."""

import pytest
from pydantic import SecretStr

from food_concierge.config import Settings
from food_concierge.security import rate_limit
from food_concierge.security.access import (
    TOOL_SCOPES,
    ForbiddenError,
    Scope,
    UnauthorizedError,
    check,
    required_scope,
    scope_for,
)
from food_concierge.security.rate_limit import RateLimitedError, RateLimiter


def settings(**tokens: str) -> Settings:
    update = {f"api_token_{scope}": SecretStr(value) for scope, value in tokens.items()}
    return Settings(_env_file=None).model_copy(update=update)


@pytest.mark.parametrize(
    ("authorization", "scope"),
    [
        (None, Scope.PUBLIC),
        ("  ", Scope.PUBLIC),
        ("Bearer pub-token", Scope.PUBLIC),
        ("Bearer agent-token", Scope.AGENT),
        ("bearer  admin-token ", Scope.ADMIN),
    ],
    ids=["none", "blank", "public", "agent", "admin-loose-spacing"],
)
def test_a_token_grants_its_scope_and_no_token_is_public(authorization: str | None, scope: Scope) -> None:
    configured = settings(public="pub-token", agent="agent-token", admin="admin-token")

    assert scope_for(authorization, configured) is scope


@pytest.mark.parametrize(
    ("authorization", "message"),
    [
        ("Bearer guessed", "The access token is not valid."),
        ("Basic dXNlcjpwYXNz", "Send the token as 'Authorization: Bearer <token>'."),
        ("Bearer", "Send the token as"),
    ],
    ids=["wrong-token", "wrong-scheme", "no-token"],
)
def test_a_bad_token_is_refused_not_treated_as_public(authorization: str, message: str) -> None:
    with pytest.raises(UnauthorizedError, match=message):
        scope_for(authorization, settings(agent="agent-token"))


def test_unset_or_blank_tokens_match_nothing() -> None:
    with pytest.raises(UnauthorizedError):
        scope_for("Bearer ", settings(agent=""))
    with pytest.raises(UnauthorizedError):
        scope_for("Bearer anything", Settings(_env_file=None))


def test_a_token_set_for_two_scopes_grants_the_higher() -> None:
    assert scope_for("Bearer same", settings(public="same", admin="same")) is Scope.ADMIN


def test_each_tool_needs_its_scope_and_an_unlisted_tool_needs_admin() -> None:
    assert required_scope("describe_image") is Scope.AGENT
    assert required_scope("search_dishes") is Scope.PUBLIC
    assert required_scope("run_ingestion") is Scope.ADMIN
    assert set(TOOL_SCOPES.values()) == {Scope.PUBLIC, Scope.AGENT}

    check(Scope.AGENT, "describe_image")
    check(Scope.ADMIN, "run_ingestion")
    with pytest.raises(ForbiddenError, match=r"describe_image needs the agent scope; this request has public\."):
        check(Scope.PUBLIC, "describe_image")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_each_caller_has_its_own_bucket_and_the_server_one_in_all() -> None:
    clock = Clock()
    limiter = RateLimiter(per_caller=2, overall=3, clock=clock)

    limiter.acquire("a")
    limiter.acquire("a")
    with pytest.raises(RateLimitedError, match="retry in 30 s") as refused:
        limiter.acquire("a")
    assert refused.value.retry_after_s == pytest.approx(30.0)
    limiter.acquire("b")  # a's empty bucket doesn't hold b back
    with pytest.raises(RateLimitedError):
        limiter.acquire("c")  # but the server's bucket is now empty
    clock.now = 20.0  # the server's bucket refills at 3 a minute: one request per 20 s
    limiter.acquire("c")


def test_a_refused_request_takes_nothing_from_either_bucket() -> None:
    clock = Clock()
    limiter = RateLimiter(per_caller=1, overall=10, clock=clock)

    limiter.acquire("a")
    for _ in range(5):
        with pytest.raises(RateLimitedError):
            limiter.acquire("a")
    for caller in "bcdefghij":
        limiter.acquire(caller)  # all nine fit: a's refusals spent none of the server's ten


def test_forgotten_callers_start_with_a_full_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "MAX_CALLERS", 2)
    limiter = RateLimiter(per_caller=1, overall=100, clock=Clock())

    limiter.acquire("a")
    limiter.acquire("b")
    limiter.acquire("c")  # "a" is the least recently used, so it is forgotten

    limiter.acquire("a")
    with pytest.raises(RateLimitedError):
        limiter.acquire("c")
