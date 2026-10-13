"""Call and token caps per model, and a daily photo budget, so the service stays inside free-tier quotas.

A free quota is shared by every user of the demo, and one script or a busy hour
can spend it, leaving everyone on the fallback or on retrieval-only answers.
Guarded models ask the process's ``UsageLedger`` before every attempt; at a cap
the attempt is refused with ``BudgetExceededError``, which moves the fallback
chain on instead of sending a request the provider would answer with a 429.

Each attempt counts against one or more budgets:

- its model's (``groq/<model>``): Groq sets its limits per model, so two models
  on one provider never share a count (ADR-0006, ADR-0007);
- for a vision call on a hosted provider, also the shared photo budget, so
  photos stop at a daily count of their own and cannot spend chat's quota.

Within a budget:

- Calls per UTC day and per rolling minute are counted when an attempt starts:
  failed attempts use the provider's quota too.
- Tokens per rolling minute (Groq's binding limit, ADR-0006) and per rolling
  24 hours (Groq's daily cap, found 2026-10-11, ADR-0007) are counted from the
  usage each reply reports. A call's size is unknown until it returns, so new
  calls stop once a window's tokens reach the cap; the call that crosses it may
  still get a 429, which falls back like any other. Streams report usage only
  when the endpoint is asked for it (``stream_usage``), which the router does
  not do yet, so streamed calls count toward call caps only.

Counts live in this process: they restart with it, and separate worker
processes count separately, so caps are per process. A persistent, shared
budget was dropped when paid providers left the default config (audit A10).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict

from food_concierge.config import ModelRole, ProviderName, Settings
from food_concierge.errors import BudgetExceededError

WINDOW_S = 60.0
DAY_S = 24 * 3600.0
PHOTOS = "photos"
BUSY = "The model service is busy. Please retry shortly."
PHOTO_LIMIT = "Today's photo limit has been reached. Please describe the dish in words instead."


class Caps(BaseModel):
    """One budget's limits; ``None`` is uncapped and 0 refuses every call."""

    model_config = ConfigDict(frozen=True)

    per_day: int | None = None
    per_minute: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None

    @classmethod
    def for_provider(cls, settings: Settings, provider: ProviderName) -> Caps:
        """The caps each of ``provider``'s models gets."""
        return cls(
            per_day=settings.daily_call_limit(provider),
            per_minute=settings.minute_call_limit(provider),
            tokens_per_minute=settings.minute_token_limit(provider),
            tokens_per_day=settings.daily_token_limit(provider),
        )


class Budget(BaseModel):
    """Caps under one ledger key; ``day_message`` replaces the default when its daily cap refuses a call."""

    model_config = ConfigDict(frozen=True)

    key: str
    caps: Caps = Caps()
    day_message: str | None = None


def budget_key(provider: str, model: str) -> str:
    return f"{provider}/{model}"


def budgets_for(settings: Settings, provider: ProviderName, role: ModelRole) -> tuple[Budget, ...]:
    """What one attempt at ``provider``'s model for ``role`` counts against."""
    own = Budget(
        key=budget_key(provider, settings.require_model_id(provider, role)),
        caps=Caps.for_provider(settings, provider),
    )
    # Hosted providers have a daily call cap; local ones cost nothing, so their photos are not budgeted.
    if role != "vision" or settings.daily_call_limit(provider) is None:
        return (own,)
    photos = Budget(key=PHOTOS, caps=Caps(per_day=settings.daily_photo_limit), day_message=PHOTO_LIMIT)
    return own, photos


class Usage(BaseModel):
    calls_today: int
    calls_last_minute: int
    tokens_last_minute: int
    tokens_last_day: int  # the rolling 24 hours, as Groq counts its daily cap
    tokens_total: int  # every reported token since start (or reset); never leaves a window


class UsageLedger:
    """Calls and tokens per budget key in this process. Thread-safe."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._days: dict[str, tuple[date, int]] = {}
        self._calls: dict[str, deque[float]] = {}
        self._tokens: dict[str, deque[tuple[float, int]]] = {}
        self._tokens_total: dict[str, int] = {}

    def acquire(self, provider: str, budgets: Sequence[Budget]) -> None:
        """Count one attempt at ``provider`` against every budget, or none if any cap is reached."""
        with self._lock:
            now = self._clock()
            today = datetime.fromtimestamp(now, UTC).date()
            usages = [(budget, self._usage(budget.key, now, today)) for budget in budgets]
            for budget, usage in usages:
                _check(provider, budget, usage)
            for budget, usage in usages:
                self._days[budget.key] = (today, usage.calls_today + 1)
                self._calls.setdefault(budget.key, deque()).append(now)

    def record_tokens(self, key: str, tokens: int) -> None:
        """Add the tokens one reply reported to the rolling windows and the running total."""
        if tokens <= 0:
            return
        with self._lock:
            self._tokens.setdefault(key, deque()).append((self._clock(), tokens))
            self._tokens_total[key] = self._tokens_total.get(key, 0) + tokens

    def usage(self, key: str) -> Usage:
        with self._lock:
            now = self._clock()
            return self._usage(key, now, datetime.fromtimestamp(now, UTC).date())

    def reset(self) -> None:
        with self._lock:
            self._days.clear()
            self._calls.clear()
            self._tokens.clear()
            self._tokens_total.clear()

    def _usage(self, key: str, now: float, today: date) -> Usage:
        # Caller holds the lock. Entries older than their window are dropped, so memory stays bounded by the rate.
        calls = self._calls.get(key, deque())
        while calls and calls[0] <= now - WINDOW_S:
            calls.popleft()
        tokens = self._tokens.get(key, deque())
        while tokens and tokens[0][0] <= now - DAY_S:
            tokens.popleft()
        day, count = self._days.get(key, (today, 0))
        return Usage(
            calls_today=count if day == today else 0,
            calls_last_minute=len(calls),
            tokens_last_minute=sum(n for t, n in tokens if t > now - WINDOW_S),
            tokens_last_day=sum(n for _, n in tokens),
            tokens_total=self._tokens_total.get(key, 0),
        )


def _check(provider: str, budget: Budget, usage: Usage) -> None:
    caps = budget.caps
    if (caps.per_day is not None and usage.calls_today >= caps.per_day) or (
        caps.tokens_per_day is not None and usage.tokens_last_day >= caps.tokens_per_day
    ):
        raise BudgetExceededError(budget.day_message, provider=provider)
    if (caps.per_minute is not None and usage.calls_last_minute >= caps.per_minute) or (
        caps.tokens_per_minute is not None and usage.tokens_last_minute >= caps.tokens_per_minute
    ):
        raise BudgetExceededError(BUSY, provider=provider)


_ledger = UsageLedger()


def get_usage_ledger() -> UsageLedger:
    """The ledger every guarded model in this process counts against."""
    return _ledger
