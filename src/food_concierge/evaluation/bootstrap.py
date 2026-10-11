"""Seeded percentile bootstrap over queries (audit A18): a mean with its interval, and paired differences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def bootstrap_mean(
    values: Sequence[float], seed: int, resamples: int = RESAMPLES, confidence: float = 0.95
) -> Interval:
    """Mean of ``values`` with a percentile interval from resampling queries with replacement."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("no values to resample")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return Interval(float(data.mean()), float(low), float(high))


def paired_bootstrap(
    baseline: Sequence[float], candidate: Sequence[float], seed: int, resamples: int = RESAMPLES
) -> Interval:
    """Mean of ``candidate - baseline`` per query; the same queries are resampled for both."""
    if len(baseline) != len(candidate):
        raise ValueError("paired scores need one value per query on both sides")
    differences = [c - b for b, c in zip(baseline, candidate, strict=True)]
    return bootstrap_mean(differences, seed, resamples)
