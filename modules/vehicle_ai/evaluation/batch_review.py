"""Attach explicit AI-authored semantic decisions to a complete batch run."""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REVIEWER = "Codex AI self-review"


def review_batch(run_root: Path, decisions: list[dict]) -> dict:
    if (run_root / "reviewed.json").exists():
        raise FileExistsError(run_root / "reviewed.json")
    run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "completed":
        raise ValueError("cannot review an incomplete run")
    trials = run["trials"]
    keys = {(item["case_id"], item["trial_index"]) for item in trials}
    decision_keys = {
        (item.get("case_id"), item.get("trial_index")) for item in decisions
    }
    if len(decisions) != len(keys) or decision_keys != keys:
        raise ValueError("semantic decisions must cover every trial exactly once")
    by_key = {(item["case_id"], item["trial_index"]): item for item in decisions}
    reviewed: list[dict] = []
    for trial in trials:
        key = (trial["case_id"], trial["trial_index"])
        decision = by_key[key]
        if set(decision) != {"case_id", "trial_index", "verdict", "evidence"}:
            raise ValueError(f"invalid semantic decision fields: {key}")
        if decision["verdict"] not in {"pass", "fail"}:
            raise ValueError(f"semantic decision must be pass or fail: {key}")
        if (
            not isinstance(decision["evidence"], str)
            or not decision["evidence"].strip()
        ):
            raise ValueError(f"semantic decision needs evidence: {key}")
        trace = run_root / trial["trace"]
        trace_bytes = trace.read_bytes()
        if hashlib.sha256(trace_bytes).hexdigest() != trial.get("trace_sha256"):
            raise ValueError(f"trial trace hash mismatch: {key}")
        payload = json.loads(trace_bytes)
        if payload["trial"]["case_sha256"] != trial["case_sha256"]:
            raise ValueError(f"trial trace identity mismatch: {key}")
        if trial.get("mechanical_status") not in {"fail", "needs_review"}:
            raise ValueError(f"invalid mechanical status: {key}")
        if payload["grade"].get("status") != trial["mechanical_status"]:
            raise ValueError(f"trial grade mismatch: {key}")
        for metric in ("tool_selection", "argument_match", "final_state"):
            if payload["grade"].get(metric) != trial.get(metric):
                raise ValueError(f"trial grade mismatch: {key} / {metric}")
        passed = (
            trial["mechanical_status"] == "needs_review"
            and decision["verdict"] == "pass"
        )
        reviewed.append(
            {
                **trial,
                "semantic_verdict": decision["verdict"],
                "semantic_evidence": decision["evidence"],
                "task_pass": passed,
            }
        )
    passed = sum(item["task_pass"] for item in reviewed)
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "total": 0}
    )
    for item in reviewed:
        for bucket, name in (
            (by_split, item["split"]),
            (by_category, item["category"]),
        ):
            bucket[name]["total"] += 1
            bucket[name]["passed"] += item["task_pass"]
    result = {
        "schema_version": 1,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": REVIEWER,
        "independent_human_review": False,
        "provider": run["provider"],
        "model": run["model"],
        "task_success": {"passed": passed, "total": len(reviewed)},
        "by_split": dict(by_split),
        "by_category": dict(by_category),
        "trials": reviewed,
    }
    (run_root / "reviewed.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 在线 Agent 内部评测语义复核",
        "",
        f"- 模型：{run['provider']} / {run['model']}",
        f"- Task Success（端到端任务成功）：{passed}/{len(reviewed)}",
        "- 审核者：Codex AI 自审；不是独立人工标注。",
        "- 每条 verdict 和依据见 `reviewed.json`；失败与成功均保留原始 trace。",
        "",
        "## 分层结果",
        "",
    ]
    for split, stats in sorted(by_split.items()):
        lines.append(f"- {split}: {stats['passed']}/{stats['total']}")
    lines.extend(["", "## 类别结果", ""])
    for category, stats in sorted(by_category.items()):
        lines.append(f"- {category}: {stats['passed']}/{stats['total']}")
    (run_root / "reviewed_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result
