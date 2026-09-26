"""Safe provenance for repeated Agent evaluations."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from modules.vehicle_ai.evaluation.models import EvaluationCase


@dataclass(frozen=True)
class BatchProvenance:
    source_revision: str
    source_tree_clean: bool
    manifest_sha256: str
    case_set_sha256: str
    rubric_set_sha256: str
    agent_config_sha256: str
    case_count: int
    repetitions: int
    split: str
    temperature: float
    timeout_seconds: float
    max_tool_rounds: int
    turn_timeout_seconds: float
    max_tool_calls: int
    max_task_trace_events: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_batch_provenance(
    repo_root: Path,
    dataset_root: Path,
    agent_config: Path,
    cases: Sequence[EvaluationCase],
    *,
    repetitions: int,
    split: str,
    temperature: float,
    timeout_seconds: float,
    max_tool_rounds: int,
    turn_timeout_seconds: float,
    max_tool_calls: int,
    max_task_trace_events: int,
) -> BatchProvenance:
    """Bind a run to source, exact frozen inputs and runtime budgets, never secrets."""
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in revision.lower()
    ):
        raise ValueError("git returned an invalid source revision")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not cases or repetitions < 1:
        raise ValueError("provenance requires cases and positive repetitions")

    manifest_sha256 = _sha256((dataset_root / "manifest.yaml").read_bytes())
    case_digest = hashlib.sha256()
    rubric_digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: item.id):
        case_digest.update(case.id.encode())
        case_digest.update(b"\0")
        case_digest.update(case.sha256.encode())
        case_digest.update(b"\n")
        rubric_digest.update(case.id.encode())
        rubric_digest.update(b"\0")
        rubric_digest.update(
            (dataset_root / "rubrics" / f"{case.id}.yaml").read_bytes()
        )
        rubric_digest.update(b"\0")

    return BatchProvenance(
        source_revision=revision,
        source_tree_clean=not bool(status.strip()),
        manifest_sha256=manifest_sha256,
        case_set_sha256=case_digest.hexdigest(),
        rubric_set_sha256=rubric_digest.hexdigest(),
        agent_config_sha256=_sha256(agent_config.read_bytes()),
        case_count=len(cases),
        repetitions=repetitions,
        split=split,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_tool_rounds=max_tool_rounds,
        turn_timeout_seconds=turn_timeout_seconds,
        max_tool_calls=max_tool_calls,
        max_task_trace_events=max_task_trace_events,
    )
