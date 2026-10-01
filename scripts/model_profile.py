"""Phase 1 model profile: what each free model reliably does, measured with real calls (never in CI).

Each candidate runs the 20 fixed prompts in ``eval/datasets/model_profile.yaml`` through the production
code path (router, structured output with one repair, tool binding) and production settings, except that
attempt timeouts are relaxed so true latency is measured; calls slower than the production timeout are
counted instead. Requests are paced under the free-tier per-minute caps in settings.

    python scripts/model_profile.py groq:openai/gpt-oss-20b gemini:gemini-3.5-flash-lite --probe
    python scripts/model_profile.py groq:openai/gpt-oss-20b gemini:gemini-3.5-flash-lite

Results go to ``eval/results/model_profile_<date>.json``; model choices go to ADR-0006.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import httpx2
import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from food_concierge.config import REPO_ROOT, ModelRole, ProviderName, Settings
from food_concierge.errors import AppError
from food_concierge.models.capabilities import CAPABILITIES
from food_concierge.models.router import build_chat_model
from food_concierge.models.structured import structured_runnable
from food_concierge.telemetry import ModelCallRecorder

DATASET = REPO_ROOT / "eval" / "datasets" / "model_profile.yaml"
RESULTS = REPO_ROOT / "eval" / "results"
PROFILE_TIMEOUT_S = 30.0

Allergen = Literal[
    "peanuts", "tree_nuts", "milk", "eggs", "gluten", "soy", "sesame", "fish", "shellfish", "mustard", "celery",
    "lupin", "molluscs", "sulphites",
]  # fmt: skip


class MealRequest(BaseModel):
    """What the diner asked for. Leave a field null when the message does not say it."""

    diet: Literal["any", "vegetarian", "vegan", "eggless"] = Field(description="'any' unless a diet is stated")
    max_calories: int | None = Field(description="Upper calorie limit for the whole meal, in kcal")
    budget_inr: int | None = Field(description="Upper price limit for the whole meal, in rupees")
    exclude_allergens: list[Allergen] = Field(description="Allergens the diner must avoid; [] if none")
    courses: int | None = Field(description="Number of courses, if stated")


class Route(BaseModel):
    """Which specialist should handle the message."""

    route: Literal["recommend", "meal_plan", "dish_info", "smalltalk", "out_of_scope"] = Field(
        description=(
            "recommend: suggest dishes; meal_plan: build a multi-course meal under limits; "
            "dish_info: a question about one specific dish; smalltalk: greetings or thanks; "
            "out_of_scope: anything not about food from the catalog"
        )
    )


# Lowercase on purpose: the class name is the tool name the model sees.
class search_dishes(BaseModel):
    """Search the catalog for dishes."""

    query: str = Field(description="What the diner is looking for, in a few words")
    diet: Literal["vegetarian", "vegan", "eggless"] | None = Field(default=None, description="Diet filter")
    max_price_inr: int | None = Field(default=None, description="Maximum price per dish, in rupees")


class get_dish_details(BaseModel):
    """Full details of one dish: ingredients, nutrition, price, restaurant."""

    dish_id: str


class check_allergens(BaseModel):
    """Check whether a dish contains, or may contain, the given allergens."""

    dish_id: str
    allergens: list[str]


class meal_totals(BaseModel):
    """Total calories, macros and price of a set of dishes."""

    dish_ids: list[str]


TOOLS = [search_dishes, get_dish_details, check_allergens, meal_totals]
EXTRACT_SYSTEM = "Extract the diner's constraints from their message. Do not invent constraints they did not state."
ROUTE_SYSTEM = "You route messages for a restaurant food concierge. Pick the one specialist for this message."
TOOLS_SYSTEM = "You are a food concierge. Use exactly one tool to answer. Dish IDs look like d-0042."


# ---------------------------------------------------------------------------
# Pacing and rate-limit headers
# ---------------------------------------------------------------------------


class Pacer:
    """Spaces requests under a per-minute cap; records the time spent waiting, statuses and rate-limit headers.

    The wait happens inside the timed call, so ``waited_s`` is subtracted from measured latency.
    """

    def __init__(self, per_minute: int | None) -> None:
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self.next_slot = 0.0
        self.waited_s = 0.0
        self.headers: dict[str, str] = {}
        self.statuses: dict[str, int] = {}
        self.lock = threading.Lock()

    def on_request(self, request: httpx2.Request) -> None:
        with self.lock:
            wait = self.next_slot - time.monotonic()
            if wait > 0:
                time.sleep(wait)
                self.waited_s += wait
            self.next_slot = time.monotonic() + self.interval

    def on_response(self, response: httpx2.Response) -> None:
        status = str(response.status_code)
        self.statuses[status] = self.statuses.get(status, 0) + 1
        for key, value in response.headers.items():
            if key.lower().startswith(("x-ratelimit", "retry-after")):
                self.headers[key.lower()] = value


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; honest at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def rate(flags: list[bool]) -> dict[str, Any]:
    hits = sum(flags)
    return {
        "rate": round(hits / len(flags), 3) if flags else None,
        "k": hits,
        "n": len(flags),
        "ci95": wilson(hits, len(flags)),
    }


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


def _norm(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def extraction_correct(parsed: MealRequest, expect: dict[str, Any]) -> bool:
    for key, wanted in expect.items():
        got = getattr(parsed, key)
        if key == "exclude_allergens":
            if {_norm(a) for a in got} != {_norm(a) for a in wanted}:
                return False
        elif isinstance(wanted, list):
            if _norm(got) not in [_norm(w) for w in wanted]:
                return False
        elif _norm(got) != _norm(wanted):
            return False
    return True


def tool_args_correct(args: dict[str, Any], expect: dict[str, dict[str, Any]]) -> bool:
    for key, rule in expect.items():
        got = args.get(key)
        if "equals" in rule and _norm(got) != _norm(rule["equals"]):
            return False
        if "contains" in rule and (not isinstance(got, str) or rule["contains"] not in got.lower()):
            return False
        if "set" in rule and (not isinstance(got, list) or {_norm(g) for g in got} != {_norm(s) for s in rule["set"]}):
            return False
    return True


# ---------------------------------------------------------------------------
# Running a candidate
# ---------------------------------------------------------------------------


def candidate_settings(provider: ProviderName, model: str) -> Settings:
    names = {"groq": "groq_chat_model", "gemini": "gemini_chat_model", "ollama": "ollama_chat_model"}
    overrides: dict[str, Any] = {
        "chat_provider": provider,
        "fallback_providers": [],
        names[provider]: model,
        "request_deadline_s": PROFILE_TIMEOUT_S * 2,
        "provider_timeout_s": PROFILE_TIMEOUT_S,
        "groq_timeout_s": PROFILE_TIMEOUT_S,
        "gemini_timeout_s": PROFILE_TIMEOUT_S,
    }
    return Settings(**overrides)


class Candidate:
    def __init__(self, provider: ProviderName, model: str) -> None:
        self.provider = provider
        self.model = model
        self.settings = candidate_settings(provider, model)
        self.production_timeout = Settings(_env_file=None).timeout_for(provider)
        self.pacer = Pacer(self.settings.minute_call_limit(provider))
        self.client = httpx2.Client(
            event_hooks={"request": [self.pacer.on_request], "response": [self.pacer.on_response]},
        )

    def model_for(self, role: ModelRole) -> Any:
        return build_chat_model(self.provider, role, self.settings, http_client=self.client)

    def timed(self, call: Any) -> tuple[Any, float, ModelCallRecorder, str | None]:
        recorder = ModelCallRecorder()
        waited_before = self.pacer.waited_s
        start = time.perf_counter()
        try:
            result, error = call({"callbacks": [recorder]}), None
        except AppError as exc:
            result, error = None, exc.code
        elapsed = time.perf_counter() - start - (self.pacer.waited_s - waited_before)
        return result, elapsed * 1000, recorder, error


def probe(candidate: Candidate) -> dict[str, Any]:
    model = candidate.model_for("chat")
    reply, latency, recorder, error = candidate.timed(
        lambda config: model.invoke("Reply with the single word: ok", config=config)
    )
    text = reply.text.strip()[:40] if isinstance(reply, BaseMessage) else None
    return {"ok": error is None, "error": error, "latency_ms": round(latency), "reply": text,
            "output_tokens": recorder.output_tokens}  # fmt: skip


def run_structured(candidate: Candidate, task: str, item: dict[str, Any]) -> dict[str, Any]:
    schema: type[BaseModel] = MealRequest if task == "extraction" else Route
    system = EXTRACT_SYSTEM if task == "extraction" else ROUTE_SYSTEM
    runnable = structured_runnable(
        candidate.model_for("router"),
        schema,
        provider=candidate.provider,
        method=CAPABILITIES[candidate.provider].structured_method,
    )
    messages = [SystemMessage(system), HumanMessage(item["text"])]
    parsed, latency, recorder, error = candidate.timed(lambda config: runnable.invoke(messages, config=config))
    model_calls = len(recorder.attempts)
    if parsed is None:
        correct = False
    elif task == "extraction":
        correct = extraction_correct(cast(MealRequest, parsed), item["expect"])
    else:
        correct = cast(Route, parsed).route == item["expect"]
    return {
        "id": item["id"], "task": task, "valid_first": parsed is not None and model_calls == 1,
        "valid_final": parsed is not None, "correct": correct, "model_calls": model_calls,
        "latency_ms": round(latency), "input_tokens": recorder.input_tokens,
        "output_tokens": recorder.output_tokens, "error": error,
        "answer": parsed.model_dump() if parsed is not None else None,
    }  # fmt: skip


def run_tool(candidate: Candidate, item: dict[str, Any]) -> dict[str, Any]:
    model = candidate.model_for("chat").bind_tools(TOOLS)
    messages = [SystemMessage(TOOLS_SYSTEM), HumanMessage(item["text"])]
    reply, latency, recorder, error = candidate.timed(lambda config: model.invoke(messages, config=config))
    calls = reply.tool_calls if isinstance(reply, AIMessage) else []
    first = calls[0] if calls else None
    right_tool = first is not None and first["name"] == item["expect"]["name"]
    return {
        "id": item["id"], "task": "tools", "called": first is not None, "right_tool": right_tool,
        "correct": right_tool and first is not None and tool_args_correct(first["args"], item["expect"]["args"]),
        "latency_ms": round(latency), "input_tokens": recorder.input_tokens,
        "output_tokens": recorder.output_tokens, "error": error,
        "answer": {"name": first["name"], "args": first["args"]} if first else None,
    }  # fmt: skip


def summarise(candidate: Candidate, items: list[dict[str, Any]]) -> dict[str, Any]:
    structured = [i for i in items if i["task"] != "tools"]
    tools = [i for i in items if i["task"] == "tools"]
    latencies = [float(i["latency_ms"]) for i in items if i["error"] is None]
    errors: dict[str, int] = {}
    for item in items:
        if item["error"]:
            errors[item["error"]] = errors.get(item["error"], 0) + 1
    return {
        "structured_valid_first_try": rate([i["valid_first"] for i in structured]),
        "structured_valid_after_repair": rate([i["valid_final"] for i in structured]),
        "extraction_correct": rate([i["correct"] for i in structured if i["task"] == "extraction"]),
        "routing_correct": rate([i["correct"] for i in structured if i["task"] == "routing"]),
        "tool_called": rate([i["called"] for i in tools]),
        "tool_correct": rate([i["correct"] for i in tools]),
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95), "n": len(latencies)},
        "over_production_timeout": sum(1 for v in latencies if v > candidate.production_timeout * 1000),
        "production_timeout_s": candidate.production_timeout,
        "mean_tokens": {
            "input": round(statistics.mean(i["input_tokens"] for i in items)) if items else 0,
            "output": round(statistics.mean(i["output_tokens"] for i in items)) if items else 0,
        },
        "errors": errors,
    }


def profile(candidate: Candidate, dataset: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for task in ("extraction", "routing"):
        for item in dataset[task]:
            items.append(run_structured(candidate, task, item))
            print(f"  {item['id']}: {'ok' if items[-1]['correct'] else items[-1]['error'] or 'wrong'}", flush=True)
    for item in dataset["tools"]:
        items.append(run_tool(candidate, item))
        print(f"  {item['id']}: {'ok' if items[-1]['correct'] else items[-1]['error'] or 'wrong'}", flush=True)
    return {"metrics": summarise(candidate, items), "items": items}


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)  # noqa: S607
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", nargs="+", help="provider:model, e.g. groq:openai/gpt-oss-20b")
    parser.add_argument("--probe", action="store_true", help="one short call per candidate, no profile")
    parser.add_argument("--out", type=Path, help="results file (default eval/results/model_profile_<date>.json)")
    args = parser.parse_args(argv)

    dataset_bytes = DATASET.read_bytes()
    dataset = yaml.safe_load(dataset_bytes)
    settings = Settings()
    results: dict[str, Any] = {
        "run": {
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "commit": _git_commit(),
            "dataset": str(DATASET.relative_to(REPO_ROOT)).replace("\\", "/"),
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "dataset_version": dataset["version"],
            "max_output_tokens": settings.max_output_tokens,
            "reasoning_effort": {"chat": settings.reasoning_effort_chat, "router": settings.reasoning_effort_router},
            "chat_temperature": settings.chat_temperature,
            "model_seed": settings.model_seed,
            "attempt_timeout_s": PROFILE_TIMEOUT_S,
        },
        "candidates": [],
    }
    for spec in args.candidates:
        provider_name, _, model = spec.partition(":")
        if provider_name not in ("groq", "gemini", "ollama") or not model:
            parser.error(f"bad candidate '{spec}'; use groq:<model>, gemini:<model> or ollama:<model>")
        candidate = Candidate(cast(ProviderName, provider_name), model)
        print(f"{spec}", flush=True)
        entry: dict[str, Any] = {"provider": candidate.provider, "model": model, "probe": probe(candidate)}
        print(f"  probe: {entry['probe']}", flush=True)
        if not args.probe and entry["probe"]["ok"]:
            entry.update(profile(candidate, dataset))
        entry["rate_limit_headers"] = candidate.pacer.headers
        entry["http_statuses"] = candidate.pacer.statuses
        results["candidates"].append(entry)

    if args.probe:
        return 0
    out = args.out or RESULTS / f"model_profile_{datetime.now(UTC):%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
