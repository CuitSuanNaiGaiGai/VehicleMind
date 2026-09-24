"""Apply traceable Codex corrections to AI-judge decisions without rewriting them."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.batch_review import review_batch


def audit_batch(run_root: Path, override_path: Path) -> dict:
    judged = json.loads((run_root / "reviewed.json").read_text(encoding="utf-8"))
    progress = json.loads(
        (run_root / "judge_progress.json").read_text(encoding="utf-8")
    )
    if progress.get("source_sha256") != judged.get("source_sha256"):
        raise ValueError("judge and reviewed source mismatch")
    overrides = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    if not isinstance(overrides, list):
        raise ValueError("audit overrides must be a list")
    decisions = [dict(item) for item in progress["decisions"]]
    by_key = {(item["case_id"], item["trial_index"]): item for item in decisions}
    if len(decisions) != judged["task_success"]["total"]:
        raise ValueError("judge decision count mismatch")
    audit_log = []
    seen = set()
    for item in overrides:
        if not isinstance(item, dict) or set(item) != {
            "case_id",
            "trial_index",
            "verdict",
            "evidence",
        }:
            raise ValueError("invalid audit override fields")
        key = (item["case_id"], item["trial_index"])
        if key not in by_key or key in seen:
            raise ValueError(f"unknown or duplicate audit override: {key}")
        if item["verdict"] not in {"pass", "fail"} or not str(item["evidence"]).strip():
            raise ValueError(f"invalid audit verdict/evidence: {key}")
        original = by_key[key]
        if original["verdict"] == item["verdict"]:
            raise ValueError(f"audit override must change verdict: {key}")
        audit_log.append({**item, "previous_verdict": original["verdict"]})
        original["verdict"] = item["verdict"]
        original["evidence"] = item["evidence"]
        seen.add(key)
    return review_batch(
        run_root,
        decisions,
        reviewer=f"Codex audit of {judged['reviewer']}",
        output_stem="audited",
        audit_overrides=audit_log,
        source_sha256=judged.get("source_sha256"),
    )
