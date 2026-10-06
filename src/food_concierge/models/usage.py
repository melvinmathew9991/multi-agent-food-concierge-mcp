"""Per-provider call and token caps, so the service stays inside free-tier quotas.

A free quota is shared by every user of the demo, and one script or a busy hour
can spend it, leaving everyone on the fallback or on retrieval-only answers.
Guarded models ask the process's ``UsageLedger`` before every attempt; at a cap
the attempt is refused with ``BudgetExceededError``, which moves the fallback
chain on instead of sending a request the provider would answer with a 429.

- Calls per UTC day and per rolling minute are counted when an attempt starts:
  failed attempts use the provider's quota too.
- Tokens per rolling minute (Groq's binding limit, ADR-0006) are counted from
  the usage each reply reports. A call's size is unknown until it returns, so
  new calls stop once the last minute's tokens reach the cap; the call that
  crosses it may still get a 429, which falls back like any other. Streams
  report usage only when the endpoint is asked for it (``stream_usage``), which
  the router does not do yet, so streamed calls count toward call caps only.

Counts live in this process: they restart with it, and separate worker
processes count separately, so caps are per process. A persistent, shared
budget was dropped when paid providers left the default config (audit A10).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict

from food_concierge.config import ProviderName, Settings
from food_concierge.errors import BudgetExceededError

WINDOW_S = 60.0


class Caps(BaseModel):
    """One provider's limits; ``None`` is uncapped and 0 refuses every call."""

    model_config = ConfigDict(frozen=True)

    per_day: int | None = None
    per_minute: int | None = None
    tokens_per_minute: int | None = None

    @classmethod
    def for_provider(cls, settings: Settings, provider: ProviderName) -> Caps:
        return cls(
            per_day=settings.daily_call_limit(provider),
            per_minute=settings.minute_call_limit(provider),
            tokens_per_minute=settings.minute_token_limit(provider),
        )


class Usage(BaseModel):
    calls_today: int
    calls_last_minute: int
    tokens_last_minute: int


class UsageLedger:
    """Calls and tokens per provider in this process. Thread-safe."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._days: dict[str, tuple[date, int]] = {}
        self._calls: dict[str, deque[float]] = {}
        self._tokens: dict[str, deque[tuple[float, int]]] = {}

    def acquire(self, provider: str, caps: Caps) -> None:
        """Count one attempt at ``provider``, or raise ``BudgetExceededError`` if a cap is reached."""
        with self._lock:
            now = self._clock()
            today = datetime.fromtimestamp(now, UTC).date()
            usage = self._usage(provider, now, today)
            if caps.per_day is not None and usage.calls_today >= caps.per_day:
                raise BudgetExceededError(provider=provider)
            if caps.per_minute is not None and usage.calls_last_minute >= caps.per_minute:
                raise BudgetExceededError("The model service is busy. Please retry shortly.", provider=provider)
            if caps.tokens_per_minute is not None and usage.tokens_last_minute >= caps.tokens_per_minute:
                raise BudgetExceededError("The model service is busy. Please retry shortly.", provider=provider)
            self._days[provider] = (today, usage.calls_today + 1)
            self._calls.setdefault(provider, deque()).append(now)

    def record_tokens(self, provider: str, tokens: int) -> None:
        """Add the tokens one reply reported to the rolling minute."""
        if tokens <= 0:
            return
        with self._lock:
            self._tokens.setdefault(provider, deque()).append((self._clock(), tokens))

    def usage(self, provider: str) -> Usage:
        with self._lock:
            now = self._clock()
            return self._usage(provider, now, datetime.fromtimestamp(now, UTC).date())

    def reset(self) -> None:
        with self._lock:
            self._days.clear()
            self._calls.clear()
            self._tokens.clear()

    def _usage(self, provider: str, now: float, today: date) -> Usage:
        # Caller holds the lock. Entries older than the window are dropped, so memory stays bounded by the rate.
        cutoff = now - WINDOW_S
        calls = self._calls.get(provider, deque())
        while calls and calls[0] <= cutoff:
            calls.popleft()
        tokens = self._tokens.get(provider, deque())
        while tokens and tokens[0][0] <= cutoff:
            tokens.popleft()
        day, count = self._days.get(provider, (today, 0))
        return Usage(
            calls_today=count if day == today else 0,
            calls_last_minute=len(calls),
            tokens_last_minute=sum(n for _, n in tokens),
        )


_ledger = UsageLedger()


def get_usage_ledger() -> UsageLedger:
    """The ledger every guarded model in this process counts against."""
    return _ledger
