"""Small online pilot with a tool-call preflight and isolated case trials."""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yaml

from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.report import write_report
from modules.vehicle_ai.evaluation.runner import TrialResult, run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient


PILOT_IDS = ("C01", "R02", "X01", "X03", "T02", "T03", "T05", "M01")


def load_pilot_cases(directory: Path) -> tuple[EvaluationCase, ...]:
    cases = tuple(
        EvaluationCase.from_mapping(
            yaml.safe_load((directory / f"{case_id}.yaml").read_text(encoding="utf-8"))
        )
        for case_id in PILOT_IDS
    )
    for expected_id, case in zip(PILOT_IDS, cases, strict=True):
        if case.id != expected_id or case.split != "dev":
            raise ValueError(f"pilot case identity/split mismatch: {expected_id}")
        if case.review_status != "candidate":
            raise ValueError(f"pilot case should remain candidate: {expected_id}")
        if not all("at_ms" in step for step in case.steps):
            raise ValueError(f"pilot case step lacks at_ms: {expected_id}")
    return cases


def _preflight_case() -> EvaluationCase:
    return EvaluationCase.from_mapping({
        "id": "P00", "split": "dev", "category": "preflight",
        "review_status": "candidate",
        "steps": [{"user_text": (
            "这是接口预检。请先调用 get_climate_status 工具查询当前空调状态，"
            "再根据工具结果用中文回复。"
        )}],
        "expected": {
            "tools": [{"name": "get_climate_status", "arguments": {}}],
            "final_vehicle": {}, "required_facts": [], "forbidden_phrases": [],
        },
    })


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _endpoint_host(client: BaseLLMClient) -> str | None:
    raw = getattr(client, "base_url", None)
    return urlparse(str(raw)).hostname if raw else None


def _client_version(provider: str) -> str | None:
    package = {"qwen": "openai", "glm": "zhipuai"}.get(provider)
    if package is None:
        return None
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _trial_entry(case: EvaluationCase, trial: TrialResult, destination: Path) -> dict:
    grade = grade_trial(case, trial)
    write_report(destination, case, trial, grade)
    return {
        "case_id": case.id,
        "case_sha256": case.sha256,
        "status": grade["status"],
        "error": trial.error,
        "request_count": trial.request_count,
        "latency_ms": trial.latency_ms,
        "trace": str(destination / "trial.json"),
    }


def _write_manifest(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_pilot(
    provider: str,
    client: BaseLLMClient,
    cases: tuple[EvaluationCase, ...],
    *,
    output_root: Path,
    model: str,
) -> dict:
    """Run a fresh runtime per trial and never silently continue past failed preflight."""
    output_root.mkdir(parents=True, exist_ok=False)
    case_entries: list[dict] = []
    result: dict = {
        "run_type": "candidate_pilot",
        "status": "started",
        "provider": provider,
        "requested_model": model,
        "endpoint_host": _endpoint_host(client),
        "client_version": _client_version(provider),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "preflight": None,
        "cases": case_entries,
    }
    _write_manifest(output_root / "pilot.json", result)
    preflight_case = _preflight_case()
    preflight_trial = run_trial(
        preflight_case, client, provider=provider, model=model, trial_index=1,
    )
    preflight_grade = grade_trial(preflight_case, preflight_trial)
    preflight_passed = (
        preflight_trial.error is None
        and preflight_grade["tool_selection"]
        and preflight_grade["argument_match"]
        and len(preflight_trial.tool_calls) == 1
        and preflight_trial.tool_calls[0]["success"]
    )
    write_report(
        output_root / "preflight", preflight_case,
        preflight_trial, preflight_grade,
    )
    result["preflight"] = {
        "passed": preflight_passed,
        "trial": asdict(preflight_trial),
        "error": preflight_trial.error,
    }
    result["status"] = "preflight_passed" if preflight_passed else "preflight_failed"
    _write_manifest(output_root / "pilot.json", result)
    if preflight_passed:
        for case in cases:
            trial = run_trial(case, client, provider=provider, model=model, trial_index=1)
            case_entries.append(
                _trial_entry(case, trial, output_root / case.id)
            )
            _write_manifest(output_root / "pilot.json", result)
        result["status"] = "completed"
        _write_manifest(output_root / "pilot.json", result)
    return result
