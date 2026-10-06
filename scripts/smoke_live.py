"""Phase 1 live smoke test: real providers, real Langfuse, never in CI.

Checks the Phase 1 definition of done against live services, through the production code path:

1. one call each to Groq, Gemini and Ollama;
2. a forced Groq failure (an invalid key, for this run only) answered by Gemini, with the fallback recorded;
3. one typed-output call (validation, and the repair path if the model needs it);
4. traces read back from the Langfuse server: a planted fake email, phone and card number must arrive masked
   with ``TRACE_CONTENT=full``, and no message content at all may arrive with ``TRACE_CONTENT=metadata``
   (checked in a child process, because Langfuse keeps one client per key in a process);
5. optionally, ``--photos DIR``: up to five of your own photos described by the local Ollama vision model.

Needs local Langfuse (``docker compose --env-file .env.langfuse up -d``) with its keys in ``.env``, Groq and
Gemini keys in ``.env``, and Ollama running with the configured models pulled. About ten free-tier calls.
The planted values are fake (example.com, a 555-01xx number, the standard 4111 test card).

    python scripts/smoke_live.py
    python scripts/smoke_live.py --photos path/to/photos

Results go to ``eval/results/smoke_<UTC time>.json`` (no keys, no file names); exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field, SecretStr

from food_concierge.config import REPO_ROOT, ProviderName, Settings, TraceContent
from food_concierge.errors import AppError
from food_concierge.logging_setup import configure_logging
from food_concierge.models.images import clean_image
from food_concierge.models.router import build_chat_model, get_chat_model
from food_concierge.models.structured import get_structured_model
from food_concierge.telemetry import CONTENT_PLACEHOLDER, ModelCallRecorder, Telemetry, TraceMeta, init_telemetry

RESULTS = REPO_ROOT / "eval" / "results"
Status = Literal["pass", "fail", "skipped"]

# Fake personal data planted in the traced request; none of it may reach Langfuse unmasked.
PLANTED = {
    "email": ("smoke@example.com", "[email]"),
    "phone": ("+1 (555) 010-0199", "[phone]"),
    "card": ("4111 1111 1111 1111", "[card]"),
}
# Ordinary text in the same request: kept with TRACE_CONTENT=full, absent with TRACE_CONTENT=metadata.
CANARY = "smoke-canary-paneer"
PLANTED_MESSAGE = (
    f"Ignore the details and reply with the single word: ok. Order note {CANARY}: email "
    f"{PLANTED['email'][0]}, phone {PLANTED['phone'][0]}, card {PLANTED['card'][0]}."
)

# Local models load on first use, which can take far longer than a hosted call; hosted providers keep
# their production timeouts, since checking those is the point.
OLLAMA_TIMEOUT_S = 90.0
TRACE_WAIT_S = 45.0
# Every field group of a Langfuse v2 observation, so a leak in any of them is seen.
OBSERVATION_FIELDS = "core,basic,time,io,metadata,model,usage,prompt,metrics,trace_context"
MAX_PHOTOS = 5
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


class Check(BaseModel):
    name: str
    status: Status
    detail: dict[str, Any] = Field(default_factory=dict)


class SmokeRequest(BaseModel):
    """What the diner asked for. Leave a field null when the message does not say it."""

    diet: Literal["any", "vegetarian", "vegan", "eggless"] = Field(description="'any' unless a diet is stated")
    max_calories: int | None = Field(description="Upper calorie limit for the meal, in kcal")
    exclude_allergens: list[str] = Field(description="Allergens to avoid, lowercase; [] if none")


# ---------------------------------------------------------------------------
# Pure helpers (tested offline)
# ---------------------------------------------------------------------------


def evaluate_trace(exported: str, content: TraceContent) -> dict[str, Any]:
    """What a trace read back from Langfuse shows: leaks of planted values, and whether content was withheld."""
    leaked = sorted(name for name, (raw, _) in PLANTED.items() if raw in exported)
    result: dict[str, Any] = {"leaked": leaked}
    if content == "full":
        result["masked"] = sorted(name for name, (_, mask) in PLANTED.items() if mask in exported)
        result["content_kept"] = CANARY in exported
        result["ok"] = not leaked and len(result["masked"]) == len(PLANTED) and result["content_kept"]
    else:
        result["placeholder"] = CONTENT_PLACEHOLDER in exported
        result["content_absent"] = CANARY not in exported
        result["ok"] = not leaked and result["placeholder"] and result["content_absent"]
    return result


def fallback_outcome(recorder: ModelCallRecorder) -> dict[str, Any]:
    attempts = [(a.provider, a.error_code) for a in recorder.attempts]
    meta = recorder.meta(role="chat")
    expected = attempts == [("groq", "provider_auth_error"), ("gemini", None)]
    return {"attempts": attempts, "fallback_used": meta.fallback_used, "ok": expected and meta.fallback_used}


def list_photos(directory: Path, limit: int = MAX_PHOTOS) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES)[:limit]


def overall_status(checks: list[Check]) -> Status:
    return "fail" if any(c.status == "fail" for c in checks) else "pass"


# ---------------------------------------------------------------------------
# Live checks
# ---------------------------------------------------------------------------


def _timed(call: Callable[[], Any]) -> tuple[Any, float, str | None]:
    start = time.perf_counter()
    try:
        result, error = call(), None
    except AppError as exc:
        result, error = None, exc.code
    return result, round((time.perf_counter() - start) * 1000), error


def _single(settings: Settings, provider: ProviderName) -> Settings:
    overrides: dict[str, Any] = {"chat_provider": provider, "fallback_providers": []}
    if provider == "ollama":
        overrides |= {"provider_timeout_s": OLLAMA_TIMEOUT_S, "request_deadline_s": OLLAMA_TIMEOUT_S * 2}
    return settings.model_copy(update=overrides)


def _text(reply: Any) -> str | None:
    return reply.text.strip()[:80] if isinstance(reply, BaseMessage) else None


def check_provider(telemetry: Telemetry, settings: Settings, provider: ProviderName) -> Check:
    single = _single(settings, provider)
    recorder = ModelCallRecorder()
    with telemetry.observe(f"smoke.provider.{provider}", meta=TraceMeta(request_id=f"smoke-{provider}")) as span:
        model = get_chat_model("chat", single)
        config: Any = {"callbacks": [recorder, *telemetry.callbacks()]}
        reply, latency, error = _timed(lambda: model.invoke("Reply with the single word: ok", config=config))
    detail = {
        "model": single.require_model_id(provider, "chat"),
        "latency_ms": latency,
        "error": error,
        "reply": _text(reply),
        "tokens": recorder.input_tokens + recorder.output_tokens,
        "trace_id": span.trace_id,
    }
    return Check(name=f"provider:{provider}", status="pass" if error is None else "fail", detail=detail)


def check_fallback(telemetry: Telemetry, settings: Settings) -> Check:
    # model_copy skips validation on purpose: the chain is the configured one, only Groq's key is wrong.
    broken = settings.model_copy(
        update={
            "chat_provider": "groq",
            "fallback_providers": ["gemini"],
            "groq_api_key": SecretStr("gsk-smoke-invalid"),
        }
    )
    recorder = ModelCallRecorder()
    with telemetry.observe("smoke.fallback", meta=TraceMeta(request_id="smoke-fallback")) as span:
        model = get_chat_model("chat", broken)
        config: Any = {"callbacks": [recorder, *telemetry.callbacks()]}
        reply, latency, error = _timed(lambda: model.invoke("Reply with the single word: ok", config=config))
    outcome = fallback_outcome(recorder)
    detail = {**outcome, "latency_ms": latency, "error": error, "reply": _text(reply), "trace_id": span.trace_id}
    return Check(name="fallback:groq->gemini", status="pass" if outcome["ok"] else "fail", detail=detail)


def check_structured(telemetry: Telemetry, settings: Settings) -> Check:
    recorder = ModelCallRecorder()
    message = "Vegan dinner under 600 calories, no peanuts please"
    with telemetry.observe("smoke.structured", meta=TraceMeta(request_id="smoke-structured")) as span:
        model = get_structured_model(SmokeRequest, "router", settings)
        config: Any = {"callbacks": [recorder, *telemetry.callbacks()]}
        parsed, latency, error = _timed(lambda: model.invoke(message, config=config))
    correct = isinstance(parsed, SmokeRequest) and (
        parsed.diet == "vegan" and parsed.max_calories == 600 and parsed.exclude_allergens == ["peanuts"]
    )
    detail = {
        "answer": parsed.model_dump() if isinstance(parsed, SmokeRequest) else None,
        "correct": correct,
        "model_calls": len(recorder.attempts),
        "provider": recorder.meta().provider,
        "latency_ms": latency,
        "error": error,
        "trace_id": span.trace_id,
    }
    # A wrong but valid answer is a model-quality finding for the profile, not a broken pipeline.
    return Check(name="structured", status="pass" if error is None else "fail", detail=detail)


def fetch_trace(settings: Settings, trace_id: str, wait_s: float = TRACE_WAIT_S) -> str | None:
    """The trace's observations as stored by the Langfuse server, as JSON text; ``None`` if they never arrived.

    Langfuse v4 servers ("events_only" mode) no longer serve ``/api/public/traces``; observations are read
    from the v2 endpoint, with every field group that can carry content.
    """
    public = settings.langfuse_public_key.get_secret_value() if settings.langfuse_public_key else ""
    secret = settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else ""
    params = {"traceId": trace_id, "fields": OBSERVATION_FIELDS, "expandMetadata": "request_id", "limit": "100"}
    deadline = time.monotonic() + wait_s
    with httpx.Client(base_url=settings.langfuse_host, auth=(public, secret), timeout=10) as client:
        while time.monotonic() < deadline:
            response = client.get("/api/public/v2/observations", params=params)
            # Ingestion is asynchronous: wait until the model run has arrived under the request span too.
            if response.status_code == 200 and len(response.json().get("data", [])) >= 2:
                return response.text
            time.sleep(2)
    return None


def check_trace(settings: Settings, content: TraceContent) -> Check:
    traced = settings.model_copy(update={"trace_content": content})
    telemetry = init_telemetry(traced)
    request_id = f"smoke-trace-{content}-{uuid.uuid4().hex[:8]}"
    with telemetry.observe("smoke.trace", input=PLANTED_MESSAGE, meta=TraceMeta(request_id=request_id)) as span:
        model = get_chat_model("chat", settings)
        reply, _, error = _timed(
            lambda: model.invoke([HumanMessage(PLANTED_MESSAGE)], config={"callbacks": telemetry.callbacks()})
        )
        span.update(output=_text(reply))
    telemetry.flush()
    trace_id = span.trace_id
    exported = fetch_trace(settings, trace_id) if trace_id else None
    if exported is None:
        detail = {"trace_id": trace_id, "error": error, "arrived": False}
        return Check(name=f"trace:{content}", status="fail", detail=detail)
    outcome = evaluate_trace(exported, content)
    outcome |= {"trace_id": trace_id, "request_id_kept": request_id in exported, "model_error": error}
    ok = outcome["ok"] and outcome["request_id_kept"]
    return Check(name=f"trace:{content}", status="pass" if ok else "fail", detail=outcome)


def _invoker(model: Any, messages: list[BaseMessage], telemetry: Telemetry) -> Callable[[], Any]:
    return lambda: model.invoke(messages, config={"callbacks": telemetry.callbacks()})


def check_vision(telemetry: Telemetry, settings: Settings, photos: list[Path]) -> Check:
    single = _single(settings, "ollama")
    model = build_chat_model("ollama", "vision", single)
    results: list[dict[str, Any]] = []
    for index, path in enumerate(photos, start=1):
        image = clean_image(path.read_bytes(), max_bytes=settings.max_image_bytes)
        prompt = HumanMessage(
            content=[
                {"type": "text", "text": "Describe the food in this photo in one sentence. Do not guess its name."},
                {"type": "image_url", "image_url": {"url": image.data_uri}},
            ]
        )
        with telemetry.observe("smoke.vision", meta=TraceMeta(request_id=f"smoke-vision-{index}")):
            reply, latency, error = _timed(_invoker(model, [prompt], telemetry))
        # Photo numbers, not file names: names can carry personal details.
        results.append(
            {
                "photo": index,
                "size": [image.width, image.height],
                "latency_ms": latency,
                "error": error,
                "description": _text(reply),
            }
        )
    failed = any(r["error"] for r in results)
    detail = {"model": single.ollama_vision_model, "photos": results}
    return Check(name="vision:ollama", status="fail" if failed else "pass", detail=detail)


def ollama_models(settings: Settings) -> set[str] | None:
    try:
        response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return {m["name"] for m in response.json().get("models", [])}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    """The commit that produced the results, marked ``-dirty`` if tracked files had uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)  # noqa: S607
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() + ("-dirty" if changes.stdout.strip() else "")


def metadata_trace_check_in_child() -> Check:
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "check.json"
        subprocess.run([sys.executable, __file__, "--child-trace-check", str(out)], check=False)  # noqa: S603
        if not out.exists():
            return Check(name="trace:metadata", status="fail", detail={"error": "child process wrote no result"})
        return Check.model_validate_json(out.read_text(encoding="utf-8"))


def run(settings: Settings, photos_dir: Path | None, skip_ollama: bool) -> list[Check]:
    telemetry = init_telemetry(settings.model_copy(update={"trace_content": "full"}))
    checks = [check_provider(telemetry, settings, "groq"), check_provider(telemetry, settings, "gemini")]

    local = None if skip_ollama else ollama_models(settings)
    if skip_ollama:
        checks.append(Check(name="provider:ollama", status="skipped", detail={"reason": "--skip-ollama"}))
    elif local is None:
        checks.append(Check(name="provider:ollama", status="fail", detail={"error": "Ollama is not reachable"}))
    elif settings.ollama_chat_model not in local:
        detail = {"error": f"model not pulled; run: ollama pull {settings.ollama_chat_model}"}
        checks.append(Check(name="provider:ollama", status="fail", detail=detail))
    else:
        checks.append(check_provider(telemetry, settings, "ollama"))

    checks += [check_fallback(telemetry, settings), check_structured(telemetry, settings)]
    telemetry.flush()
    checks.append(check_trace(settings, "full"))
    checks.append(metadata_trace_check_in_child())

    if photos_dir is not None:
        photos = list_photos(photos_dir)
        if not photos:
            checks.append(Check(name="vision:ollama", status="fail", detail={"error": "no JPEG, PNG or WebP photos"}))
        elif local is None or settings.ollama_vision_model not in local:
            detail = {"error": f"vision model not available; run: ollama pull {settings.ollama_vision_model}"}
            checks.append(Check(name="vision:ollama", status="fail", detail=detail))
        else:
            checks.append(check_vision(telemetry, settings, photos))
    telemetry.flush()
    telemetry.shutdown()
    return checks


def preflight(settings: Settings) -> str | None:
    """Why the smoke test cannot run, or ``None``."""
    if not settings.tracing_enabled or not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return "Tracing is off or the Langfuse keys are not in .env; start local Langfuse and add its keys."
    try:
        httpx.get(f"{settings.langfuse_host.rstrip('/')}/api/public/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        return f"Langfuse is not reachable at {settings.langfuse_host}."
    missing = [name for name in ("groq", "gemini") if getattr(settings, f"{name}_api_key") is None]
    if missing:
        return f"Missing API keys in .env: {', '.join(k.upper() + '_API_KEY' for k in missing)}."
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, help="folder with up to five of your own food photos (optional)")
    parser.add_argument("--skip-ollama", action="store_true", help="skip the local Ollama checks")
    parser.add_argument("--out", type=Path, help="results file (default eval/results/smoke_<UTC time>.json)")
    parser.add_argument("--child-trace-check", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level)
    if args.child_trace_check is not None:
        check = check_trace(settings, "metadata")
        args.child_trace_check.write_text(check.model_dump_json(), encoding="utf-8")
        return 0

    problem = preflight(settings)
    if problem:
        print(f"cannot run: {problem}", file=sys.stderr)
        return 2
    if args.photos is not None and not args.photos.is_dir():
        parser.error(f"--photos must be a folder: {args.photos}")

    print("A forced Groq authentication failure is part of the test; its ERROR log line is expected.\n")
    started = datetime.now(UTC)
    checks = run(settings, args.photos, args.skip_ollama)
    for check in checks:
        print(f"{check.status.upper():8} {check.name}  {json.dumps(check.detail, ensure_ascii=False)[:160]}")

    status = overall_status(checks)
    results: dict[str, Any] = {
        "run": {
            "started": started.isoformat(timespec="seconds"),
            "commit": _git_commit(),
            "chain": settings.provider_chain,
            "models": {
                "groq": settings.groq_chat_model,
                "gemini": settings.gemini_chat_model,
                "ollama": settings.ollama_chat_model,
                "ollama_vision": settings.ollama_vision_model,
            },
            "timeouts_s": {
                "groq": settings.groq_timeout_s,
                "gemini": settings.gemini_timeout_s,
                "ollama": OLLAMA_TIMEOUT_S,
                "deadline": settings.request_deadline_s,
            },
            "langfuse": "local" if "localhost" in settings.langfuse_host else "remote",
        },
        "status": status,
        "checks": [c.model_dump() for c in checks],
    }
    # A timestamp, not just the date: a second run the same day must not overwrite the first.
    out = args.out or RESULTS / f"smoke_{started:%Y-%m-%dT%H%MZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shown = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"\n{status.upper()}: wrote {shown}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
