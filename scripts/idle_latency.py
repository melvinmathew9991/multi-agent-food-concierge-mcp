"""Gemini latency after idle: does the fallback fit its attempt timeout on a cold call? (#50, never in CI)

On 2026-10-06 Gemini timed out at its 5 s attempt limit after an idle stretch, so a forced fallback failed.
This script waits through a list of idle gaps; after each gap it makes one cold call and then a few warm
calls paced under the per-minute cap, through the production structured-output path (one typed router
call, with the repair if the model needs it). The attempt timeout is relaxed so true latency is measured;
calls slower than the production timeout, or slower than what is left of the 8 s deadline after a Groq
timeout, are counted instead. The production call caps apply.

    python scripts/idle_latency.py                      # gaps 0,1,2,5,10,20,30,45,60 min: about 3 h, 27 calls
    python scripts/idle_latency.py --gaps 0,5 --warm 1  # a quick check

Results go to ``eval/results/idle_latency_<provider>_<UTC time>.json``; conclusions go to ADR-0006.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from food_concierge.config import REPO_ROOT, ProviderName, Settings
from food_concierge.errors import AppError, ConfigError
from food_concierge.models.structured import get_structured_model
from food_concierge.telemetry import ModelCallRecorder

RESULTS = REPO_ROOT / "eval" / "results"
MEASURE_TIMEOUT_S = 60.0
DEFAULT_GAPS_MIN = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0)
# A typical router extraction: the call a Groq outage hands to Gemini.
MESSAGE = "Something vegetarian for two under 800 rupees, no peanuts, and keep it under 700 calories each"


class IdleRequest(BaseModel):
    """What the diner asked for. Leave a field null when the message does not say it."""

    diet: Literal["any", "vegetarian", "vegan", "eggless"] = Field(description="'any' unless a diet is stated")
    max_calories: int | None = Field(description="Upper calorie limit per person, in kcal")
    budget_inr: int | None = Field(description="Upper price limit for the whole order, in rupees")
    exclude_allergens: list[str] = Field(description="Allergens to avoid, lowercase; [] if none")


class Sample(BaseModel):
    gap_min: float
    kind: Literal["cold", "warm"]
    started: str
    latency_ms: int
    model_calls: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Pure helpers (tested offline)
# ---------------------------------------------------------------------------


def parse_gaps(text: str) -> list[float]:
    gaps = [float(part) for part in text.split(",") if part.strip()]
    if not gaps or any(g < 0 for g in gaps):
        raise ValueError("gaps must be a comma-separated list of minutes, none negative")
    return gaps


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile: with few samples, a value that was actually seen."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(samples: list[Sample], attempt_timeout_s: float, fallback_budget_s: float) -> dict[str, Any]:
    """Per kind: latency of the answered calls, errors, and calls over the timeout or the fallback budget."""
    summary: dict[str, Any] = {}
    for kind in ("cold", "warm"):
        group = [s for s in samples if s.kind == kind]
        answered = [s.latency_ms for s in group if s.error is None]
        errors: dict[str, int] = {}
        for s in group:
            if s.error is not None:
                errors[s.error] = errors.get(s.error, 0) + 1
        summary[kind] = {
            "n": len(group),
            "answered": len(answered),
            "errors": errors,
            "p50_ms": round(statistics.median(answered)) if answered else None,
            "p95_ms": percentile(answered, 0.95) if answered else None,
            "max_ms": max(answered) if answered else None,
            "over_attempt_timeout": sum(1 for ms in answered if ms > attempt_timeout_s * 1000),
            "over_fallback_budget": sum(1 for ms in answered if ms > fallback_budget_s * 1000),
            "repaired": sum(1 for s in group if s.error is None and s.model_calls > 1),
        }
    return summary


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------


def measuring_settings(settings: Settings, provider: ProviderName) -> Settings:
    # model_copy skips validation on purpose: the relaxed timeout would not fit the production deadline.
    return settings.model_copy(
        update={
            "chat_provider": provider,
            "fallback_providers": [],
            "provider_max_retries": 0,
            f"{provider}_timeout_s": MEASURE_TIMEOUT_S,
            "request_deadline_s": MEASURE_TIMEOUT_S,
        }
    )


def call_once(settings: Settings, gap_min: float, kind: Literal["cold", "warm"]) -> Sample:
    # The same build as production; any pooled connection is the one a long-running server would hold.
    model = get_structured_model(IdleRequest, "router", settings)
    recorder = ModelCallRecorder()
    started = datetime.now(UTC)
    start = time.perf_counter()
    try:
        model.invoke(MESSAGE, config={"callbacks": [recorder]})
        error = None
    except AppError as exc:
        error = exc.code
    latency = round((time.perf_counter() - start) * 1000)
    return Sample(
        gap_min=gap_min,
        kind=kind,
        started=started.isoformat(timespec="seconds"),
        latency_ms=latency,
        model_calls=len(recorder.attempts),
        error=error,
    )


def run(settings: Settings, gaps: list[float], warm: int, pace_s: float) -> list[Sample]:
    samples: list[Sample] = []
    for gap in gaps:
        if gap:
            print(f"idle {gap:g} min ...", flush=True)
            time.sleep(gap * 60)
        for index in range(1 + warm):
            if index:
                time.sleep(pace_s)
            sample = call_once(settings, gap, "cold" if index == 0 else "warm")
            samples.append(sample)
            print(f"  gap {gap:g} min {sample.kind:4} {sample.latency_ms:6} ms  {sample.error or 'ok'}", flush=True)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq"])
    parser.add_argument("--gaps", default=",".join(f"{g:g}" for g in DEFAULT_GAPS_MIN), help="idle minutes")
    parser.add_argument("--warm", type=int, default=2, help="warm calls after each cold call")
    parser.add_argument("--pace", type=float, default=10.0, help="seconds between warm calls")
    parser.add_argument("--out", type=Path, help="results file")
    args = parser.parse_args(argv)
    try:
        gaps = parse_gaps(args.gaps)
    except ValueError as exc:
        parser.error(str(exc))

    production = Settings()
    provider: Literal["gemini", "groq"] = args.provider
    try:
        production.require_api_key(provider)
    except ConfigError as exc:
        print(f"cannot run: {exc}", file=sys.stderr)
        return 2
    settings = measuring_settings(production, provider)
    attempt_timeout = production.timeout_for(provider)
    # A Groq attempt that times out leaves the fallback only what is left of the deadline.
    fallback_budget = production.request_deadline_s - production.timeout_for(production.chat_provider)

    started = datetime.now(UTC)
    samples = run(settings, gaps, args.warm, args.pace)
    summary = summarize(samples, attempt_timeout, fallback_budget)
    results = {
        "run": {
            "started": started.isoformat(timespec="seconds"),
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": provider,
            "model": production.require_model_id(provider, "router"),
            "gaps_min": gaps,
            "warm_per_gap": args.warm,
            "pace_s": args.pace,
            "attempt_timeout_s": attempt_timeout,
            "fallback_budget_s": fallback_budget,
            "measure_timeout_s": MEASURE_TIMEOUT_S,
        },
        "summary": summary,
        "samples": [s.model_dump() for s in samples],
    }
    out = args.out or RESULTS / f"idle_latency_{provider}_{started:%Y-%m-%dT%H%MZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
