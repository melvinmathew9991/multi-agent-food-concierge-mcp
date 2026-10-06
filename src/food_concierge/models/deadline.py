"""A wall-clock deadline shared by every model call made inside it.

Attempt timeouts bound one HTTP request, not a call that makes several: typed
output can ask each provider twice (first try and repair), and an agent run
makes many calls. ``model_deadline`` sets one end time for everything inside
it, across the fallback chain and any thread or task started from it (it is a
context variable, which LangChain copies into its executors).

Guarded models ask ``attempt_timeout`` before each attempt. An attempt gets the
shorter of its own timeout and the time left, so a started request always ends
by the deadline; with no useful time left it is not started and
``DeadlineExceededError`` ends the call. A nested deadline never extends an outer one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

from food_concierge.errors import DeadlineExceededError

# Less time than this cannot complete a model request, so the attempt is not started.
MIN_ATTEMPT_S = 0.25

_clock: Callable[[], float] = time.monotonic
_deadline: ContextVar[float | None] = ContextVar("model_deadline", default=None)


@contextmanager
def model_deadline(seconds: float) -> Generator[None]:
    """Every model call inside the block ends within ``seconds`` from now (or the outer deadline, if sooner)."""
    end = _clock() + seconds
    current = _deadline.get()
    token = _deadline.set(end if current is None else min(current, end))
    try:
        yield
    finally:
        _deadline.reset(token)


def time_left() -> float | None:
    """Seconds until the current deadline; ``None`` outside any deadline."""
    end = _deadline.get()
    return None if end is None else end - _clock()


def attempt_timeout(timeout: float, provider: str) -> float:
    """The timeout for the next attempt at ``provider``: ``timeout``, cut to the time left."""
    left = time_left()
    if left is None:
        return timeout
    if left < MIN_ATTEMPT_S:
        raise DeadlineExceededError(provider=provider)
    return min(timeout, left)
