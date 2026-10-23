"""Token-bucket rate limits: one bucket per caller and one for the whole server (architecture §16).

A caller is its token's scope when it sends one, otherwise its client address, so anonymous callers don't share
one bucket. The buckets live in this process only, and the least recently used are forgotten past
``MAX_CALLERS``, so a flood of addresses can't grow memory without bound.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from food_concierge.errors import AppError

MAX_CALLERS = 10_000


class RateLimitedError(AppError):
    code = "rate_limited"
    http_status = 429
    default_message = "Too many requests. Please retry shortly."

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"Too many requests. Please retry in {max(1, round(retry_after_s))} s.")
        self.retry_after_s = retry_after_s


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Allows ``per_caller`` requests a minute to each caller and ``overall`` a minute in all; thread-safe."""

    def __init__(self, per_caller: int, overall: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._per_caller = per_caller
        self._overall = overall
        self._clock = clock
        self._lock = threading.Lock()
        self._callers: OrderedDict[str, _Bucket] = OrderedDict()
        self._all = _Bucket(float(overall), clock())

    def acquire(self, caller: str) -> None:
        """Take one request from ``caller``'s bucket and the overall one; raises ``RateLimitedError`` when either is
        empty, taking from neither."""
        with self._lock:
            now = self._clock()
            bucket = self._callers.pop(caller, None) or _Bucket(float(self._per_caller), now)
            self._callers[caller] = bucket
            while len(self._callers) > MAX_CALLERS:
                self._callers.popitem(last=False)
            waits = [
                _wait(bucket, self._per_caller, now),
                _wait(self._all, self._overall, now),
            ]
            if max(waits) > 0:
                raise RateLimitedError(max(waits))
            bucket.tokens -= 1
            self._all.tokens -= 1


def _wait(bucket: _Bucket, per_minute: int, now: float) -> float:
    """Refill ``bucket`` to ``now``; the seconds until it holds a whole request (0 when it does)."""
    rate = per_minute / 60.0
    bucket.tokens = min(float(per_minute), bucket.tokens + (now - bucket.updated) * rate)
    bucket.updated = now
    return 0.0 if bucket.tokens >= 1 else (1 - bucket.tokens) / rate
