"""The run record written into every committed result (rules §9.3), so a number can be traced to its inputs.

The machine is described by CPU, core count and OS family only: no hostname, user or path (rules §8.4).
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from food_concierge.errors import EvaluationError
from food_concierge.models.embeddings import EmbedderFingerprint, RerankerFingerprint

LOCK_FILE = "requirements.lock"


class Machine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu: str
    cores: int
    os: str


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    git_sha: str
    lock_sha256: str
    data_sha256: str
    embedders: dict[str, EmbedderFingerprint]
    reranker: RerankerFingerprint | None
    models: dict[str, str]  # role → model name, for runs that call a hosted model
    prompt_versions: dict[str, int]
    seeds: dict[str, int]
    machine: Machine
    created_at: datetime


def current_machine() -> Machine:
    return Machine(cpu=platform.processor() or platform.machine(), cores=os.cpu_count() or 1, os=platform.system())


def git_sha(repo_root: Path) -> str:
    """The commit being evaluated; a run outside a git checkout is refused rather than recorded as unknown."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH, as in CI
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvaluationError("The git commit could not be read; run evaluations from a git checkout.") from exc
    return completed.stdout.strip()


def make_record(
    repo_root: Path,
    data_sha256: str,
    embedders: dict[str, EmbedderFingerprint],
    reranker: RerankerFingerprint | None,
    seeds: dict[str, int],
    models: dict[str, str] | None = None,
    prompt_versions: dict[str, int] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RunRecord:
    return RunRecord(
        git_sha=git_sha(repo_root),
        lock_sha256=hashlib.sha256((repo_root / LOCK_FILE).read_bytes()).hexdigest(),
        data_sha256=data_sha256,
        embedders=embedders,
        reranker=reranker,
        models=models or {},
        prompt_versions=prompt_versions or {},
        seeds=seeds,
        machine=current_machine(),
        created_at=now(),
    )
