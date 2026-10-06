"""Offline tests for the live smoke script's decisions; the live checks themselves run only by hand."""

import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import respx
from pydantic import SecretStr

from food_concierge.config import REPO_ROOT, Settings
from food_concierge.telemetry import CONTENT_PLACEHOLDER, ModelCallRecorder


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_live", REPO_ROOT / "scripts" / "smoke_live.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_live"] = module
    spec.loader.exec_module(module)
    return module


def _masked_trace(smoke: ModuleType) -> str:
    masks = " ".join(mask for _, mask in smoke.PLANTED.values())
    return f'{{"input": "Order note {smoke.CANARY}: {masks}", "metadata": {{"request_id": "r"}}}}'


def test_a_full_trace_passes_when_everything_planted_is_masked(smoke: ModuleType) -> None:
    result = smoke.evaluate_trace(_masked_trace(smoke), "full")

    assert result == {"leaked": [], "masked": ["card", "email", "phone"], "content_kept": True, "ok": True}


def test_a_full_trace_fails_on_any_leak(smoke: ModuleType) -> None:
    leaked = _masked_trace(smoke).replace("[card]", smoke.PLANTED["card"][0])

    result = smoke.evaluate_trace(leaked, "full")

    assert result["leaked"] == ["card"]
    assert not result["ok"]


def test_a_full_trace_fails_when_content_went_missing(smoke: ModuleType) -> None:
    # Masking must not swallow ordinary text: full mode is for debugging.
    assert not smoke.evaluate_trace(_masked_trace(smoke).replace(smoke.CANARY, "x"), "full")["ok"]


def test_a_metadata_trace_must_withhold_all_content(smoke: ModuleType) -> None:
    withheld = f'{{"input": "\\"{CONTENT_PLACEHOLDER}\\"", "metadata": {{"request_id": "r"}}}}'

    assert smoke.evaluate_trace(withheld, "metadata") == {
        "leaked": [],
        "placeholder": True,
        "content_absent": True,
        "ok": True,
    }
    # Masked content is still content: metadata mode fails if any of it arrives.
    assert not smoke.evaluate_trace(_masked_trace(smoke) + CONTENT_PLACEHOLDER, "metadata")["ok"]


def _recorder(*attempts: tuple[str, str | None]) -> ModelCallRecorder:
    recorder = ModelCallRecorder()
    for provider, error in attempts:
        run_id = uuid.uuid4()
        recorder.on_chat_model_start({}, [[]], run_id=run_id, metadata={"ls_provider": provider})
        if error:
            recorder.on_llm_error(Exception(), run_id=run_id)
            recorder.attempts[-1].error_code = error
    return recorder


def test_the_fallback_must_go_from_a_groq_auth_error_to_gemini(smoke: ModuleType) -> None:
    good = smoke.fallback_outcome(_recorder(("groq", "provider_auth_error"), ("gemini", None)))
    no_fallback = smoke.fallback_outcome(_recorder(("groq", None)))
    wrong_cause = smoke.fallback_outcome(_recorder(("groq", "provider_rate_limited"), ("gemini", None)))

    assert good["ok"]
    assert good["attempts"] == [("groq", "provider_auth_error"), ("gemini", None)]
    assert not no_fallback["ok"]
    assert not wrong_cause["ok"]


def test_photos_are_images_only_sorted_and_limited(smoke: ModuleType, tmp_path: Path) -> None:
    for name in ["c.JPG", "a.png", "b.webp", "d.jpeg", "e.jpg", "f.png", "notes.txt", "clip.gif"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "folder.jpg").mkdir()

    assert [p.name for p in smoke.list_photos(tmp_path)] == ["a.png", "b.webp", "c.JPG", "d.jpeg", "e.jpg"]


def test_overall_status_fails_on_any_failure_and_ignores_skips(smoke: ModuleType) -> None:
    check = smoke.Check

    assert smoke.overall_status([check(name="a", status="pass"), check(name="b", status="skipped")]) == "pass"
    assert smoke.overall_status([check(name="a", status="pass"), check(name="b", status="fail")]) == "fail"


def _live_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "langfuse_public_key": SecretStr("pk-lf"),
        "langfuse_secret_key": SecretStr("sk-lf"),
        "groq_api_key": SecretStr("gsk"),
        "gemini_api_key": SecretStr("gem"),
    }
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


@respx.mock
def test_preflight_names_what_is_missing(smoke: ModuleType) -> None:
    health = respx.get("http://localhost:3000/api/public/health")

    assert "Langfuse keys" in smoke.preflight(Settings(_env_file=None))
    assert "Langfuse keys" in smoke.preflight(_live_settings(tracing_enabled=False))

    health.mock(side_effect=httpx.ConnectError("refused"))
    assert "not reachable" in smoke.preflight(_live_settings())

    health.mock(return_value=httpx.Response(200, json={"status": "OK"}))
    assert "GEMINI_API_KEY" in smoke.preflight(_live_settings(gemini_api_key=None))
    assert smoke.preflight(_live_settings()) is None
