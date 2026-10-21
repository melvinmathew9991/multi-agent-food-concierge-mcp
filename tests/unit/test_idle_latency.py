"""Offline tests for the idle-latency script's decisions; the live run happens only by hand (#50)."""

import importlib.util
import sys
from types import ModuleType

import pytest

from food_concierge.config import REPO_ROOT, Settings


@pytest.fixture(scope="module")
def idle() -> ModuleType:
    spec = importlib.util.spec_from_file_location("idle_latency", REPO_ROOT / "scripts" / "idle_latency.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["idle_latency"] = module
    spec.loader.exec_module(module)
    return module


def _sample(idle: ModuleType, kind: str, ms: int, error: str | None = None, calls: int = 1) -> object:
    return idle.Sample(gap_min=5, kind=kind, started="t", latency_ms=ms, model_calls=calls, error=error)


def test_gaps_parse_as_minutes(idle: ModuleType) -> None:
    assert idle.parse_gaps("0, 1.5,60") == [0.0, 1.5, 60.0]


@pytest.mark.parametrize("text", ["", "5,-1"])
def test_empty_or_negative_gaps_are_refused(idle: ModuleType, text: str) -> None:
    with pytest.raises(ValueError, match="gaps"):
        idle.parse_gaps(text)


def test_the_percentile_is_a_value_that_was_seen(idle: ModuleType) -> None:
    assert idle.percentile([300, 100, 200], 0.95) == 300
    assert idle.percentile([100, 200, 300, 400], 0.5) == 200


def test_the_summary_counts_slow_calls_against_timeout_and_fallback_budget(idle: ModuleType) -> None:
    samples = [
        _sample(idle, "cold", 5600),
        _sample(idle, "cold", 4800, calls=2),
        _sample(idle, "cold", 30000, error="provider_timeout"),
        _sample(idle, "warm", 900),
        _sample(idle, "warm", 1100),
    ]

    summary = idle.summarize(samples, attempt_timeout_s=5.0, fallback_budget_s=4.5)

    cold = summary["cold"]
    assert (cold["n"], cold["answered"], cold["errors"]) == (3, 2, {"provider_timeout": 1})
    assert (cold["p50_ms"], cold["p95_ms"], cold["max_ms"]) == (5200, 5600, 5600)
    assert (cold["over_attempt_timeout"], cold["over_fallback_budget"], cold["repaired"]) == (1, 2, 1)
    assert summary["warm"]["over_fallback_budget"] == 0


def test_a_kind_with_no_answers_has_no_latency(idle: ModuleType) -> None:
    summary = idle.summarize([_sample(idle, "cold", 30000, error="provider_rate_limited")], 5.0, 5.0)

    assert summary["cold"]["p50_ms"] is None
    assert summary["warm"]["n"] == 0


def test_measuring_relaxes_only_the_measured_provider_and_drops_the_chain(idle: ModuleType) -> None:
    production = Settings(_env_file=None)

    measured = idle.measuring_settings(production, "gemini")

    assert measured.provider_chain == ["gemini"]
    assert measured.gemini_timeout_s == idle.MEASURE_TIMEOUT_S
    assert measured.groq_timeout_s == production.groq_timeout_s
    assert measured.provider_max_retries == 0
