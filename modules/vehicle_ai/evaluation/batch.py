"""Hash-checked repeated trials for the internally reviewed Agent set."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.report import write_report
from modules.vehicle_ai.evaluation.rubric import load_rubric
from modules.vehicle_ai.evaluation.runner import TrialResult, run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient


def load_frozen_cases(root: Path) -> tuple[EvaluationCase, ...]:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    entries = manifest.get("cases", [])
    if len(entries) != 40 or manifest.get("independent_human_review") is not False:
        raise ValueError("expected a complete AI-self-reviewed 40-case manifest")
    cases: list[EvaluationCase] = []
    for item in entries:
        case_id = item["id"]
        case = EvaluationCase.from_mapping(
            yaml.safe_load(
                (root / "cases" / f"{case_id}.yaml").read_text(encoding="utf-8")
            )
        )
        if (
            case.id != case_id
            or case.split != item["split"]
            or case.sha256 != item["case_sha256"]
        ):
            raise ValueError(f"frozen case hash/identity mismatch: {case_id}")
        if case.review_status != "ai_reviewed":
            raise ValueError(f"case is not AI reviewed: {case_id}")
        rubric_path = root / "rubrics" / f"{case_id}.yaml"
        rubric_text = rubric_path.read_text(encoding="utf-8")
        if hashlib.sha256(rubric_text.encode()).hexdigest() != item["rubric_sha256"]:
            raise ValueError(f"frozen rubric hash mismatch: {case_id}")
        load_rubric(rubric_path, case)
        cases.append(case)
    if len({case.id for case in cases}) != 40:
        raise ValueError("duplicate case in frozen manifest")
    return tuple(cases)


def _trial_record(
    case: EvaluationCase,
    trial: TrialResult,
    grade: dict,
    output: Path,
    trace_sha256: str,
) -> dict:
    return {
        "case_id": case.id,
        "case_sha256": case.sha256,
        "split": case.split,
        "category": case.category,
        "trial_index": trial.trial_index,
        "trace": str(output / "trial.json"),
        "trace_sha256": trace_sha256,
        "mechanical_status": grade["status"],
        "semantic_status": "needs_review",
        "tool_selection": grade["tool_selection"],
        "argument_match": grade["argument_match"],
        "final_state": grade["final_state"],
        "error": trial.error,
        "request_count": trial.request_count,
        "latency_ms": trial.latency_ms,
        "usage": _total_usage(trial.model_responses),
    }


def _total_usage(responses: Sequence[dict[str, Any]]) -> dict[str, int | None]:
    def total(field: str) -> int | None:
        provider_field = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
        }[field]
        values = [
            (response.get("usage") or {}).get(
                field, (response.get("usage") or {}).get(provider_field)
            )
            for response in responses
        ]
        if not values or any(value is None for value in values):
            return None
        return sum(int(value) for value in values if value is not None)

    return {
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
    }


def reconcile_usage_from_traces(run_root: Path) -> dict:
    """Correct metadata from hash-verified immutable trial traces."""
    run_path = run_root / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "completed":
        raise ValueError("cannot reconcile an incomplete run")
    for item in run["trials"]:
        trace_bytes = (run_root / item["trace"]).read_bytes()
        if hashlib.sha256(trace_bytes).hexdigest() != item["trace_sha256"]:
            raise ValueError(f"trace hash mismatch: {item['case_id']}")
        trace = json.loads(trace_bytes)
        item["usage"] = _total_usage(trace["trial"]["model_responses"])
    run["usage_reconciled_from_traces"] = True
    temporary = run_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(run_path)
    (run_root / "summary.md").write_text(_summary(run), encoding="utf-8")
    return run


def _summary(result: dict[str, Any]) -> str:
    trials = result["trials"]
    total = len(trials)
    successes = sum(bool(item["tool_selection"]) for item in trials)
    arguments = sum(bool(item["argument_match"]) for item in trials)
    states = sum(bool(item["final_state"]) for item in trials)
    errors = sum(item["error"] is not None for item in trials)
    latencies = [item["latency_ms"] for item in trials]
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = sorted(latencies)[min(total - 1, int(total * 0.95))] if latencies else 0.0

    def usage_total(field: str) -> str:
        values = [item["usage"][field] for item in trials]
        return (
            "未完整报告"
            if any(value is None for value in values)
            else str(sum(int(value) for value in values if value is not None))
        )

    return (
        "# 在线 Agent 内部评测运行摘要\n\n"
        f"- 模型：{result['provider']} / {result['model']}\n"
        f"- 运行：{total} 个 case-trial；开发集/保留集按 trace 可追溯。\n"
        f"- Tool Selection（工具选择）：{successes}/{total}\n"
        f"- Argument Match（参数匹配）：{arguments}/{total}\n"
        f"- Final State（最终状态）：{states}/{total}\n"
        f"- 运行异常：{errors}/{total}\n"
        f"- 延迟 p50/p95：{p50:.1f}/{p95:.1f} ms\n"
        f"- 模型请求数：{sum(item['request_count'] for item in trials)}\n"
        f"- 输入 token：{usage_total('prompt_tokens')}；输出 token：{usage_total('completion_tokens')}\n"
        "- 回答语义仍待逐条复核；本摘要不将 needs_review 计为成功。\n"
        "- 数据为 Codex AI 自审内部基准，不是独立人工标注或公开 Benchmark。\n"
    )


def run_batch(
    cases: Sequence[EvaluationCase],
    *,
    provider: str,
    model: str,
    client_factory: Callable[[], BaseLLMClient],
    repetitions: int,
    output: Path,
    max_tool_rounds: int = 5,
    turn_timeout_seconds: float = 90.0,
    max_tool_calls: int = 10,
    max_task_trace_events: int = 200,
) -> dict:
    if repetitions < 1 or not cases:
        raise ValueError("batch requires cases and positive repetitions")
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "provider": provider,
        "model": model,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": "AI-self-reviewed internal cases; answer semantics pending",
        "status": "running",
        "trials": [],
    }

    def persist() -> None:
        temporary = output / "run.json.tmp"
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output / "run.json")

    persist()
    for case in cases:
        for trial_index in range(1, repetitions + 1):
            trial = run_trial(
                case,
                client_factory(),
                provider=provider,
                model=model,
                trial_index=trial_index,
                max_tool_rounds=max_tool_rounds,
                turn_timeout_seconds=turn_timeout_seconds,
                max_tool_calls=max_tool_calls,
                max_task_trace_events=max_task_trace_events,
            )
            grade = grade_trial(case, trial)
            relative = Path(case.id) / f"trial-{trial_index}"
            write_report(output / relative, case, trial, grade)
            trace_sha256 = hashlib.sha256(
                (output / relative / "trial.json").read_bytes()
            ).hexdigest()
            result["trials"].append(
                _trial_record(case, trial, grade, relative, trace_sha256)
            )
            persist()
    result["status"] = "completed"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist()
    (output / "summary.md").write_text(_summary(result), encoding="utf-8")
    return result
