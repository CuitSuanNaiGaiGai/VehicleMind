"""AI-assisted semantic review of completed internal Agent runs.

This is a convenience reviewer, not independent human adjudication.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from modules.vehicle_ai.evaluation.batch import _total_usage, load_frozen_cases
from modules.vehicle_ai.evaluation.batch_review import review_batch
from modules.vehicle_ai.evaluation.judge_prompt import (
    FAIL_CLOSED_EVIDENCE,
    JUDGE_SYSTEM_MESSAGE,
    build_judge_prompt,
    parse_judge_decisions,
)
from modules.vehicle_ai.llm.base import BaseLLMClient


JUDGE = "Online AI model-assisted semantic review"
PROTOCOL_VERSION = 4


def _source_sha256(run: dict, golden_root: Path) -> str:
    """Bind reviewer progress to one run's traces and frozen case/rubric set."""
    manifest_hash = hashlib.sha256(
        (golden_root / "manifest.yaml").read_bytes()
    ).hexdigest()
    source = {
        "provider": run["provider"],
        "model": run["model"],
        "started_at_utc": run["started_at_utc"],
        "golden_manifest_sha256": manifest_hash,
        "trials": sorted(
            (
                {
                    key: item[key]
                    for key in ("case_id", "case_sha256", "trial_index", "trace_sha256")
                }
                for item in run["trials"]
            ),
            key=lambda item: (item["case_id"], item["trial_index"]),
        ),
    }
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _save_progress(path: Path, progress: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def judge_batch(
    run_root: Path, golden_root: Path, client: BaseLLMClient, *, judge_model: str
) -> dict:
    """Review all replies against frozen rubrics, persisting after each case."""
    run: dict[str, Any] = json.loads(
        (run_root / "run.json").read_text(encoding="utf-8")
    )
    if run.get("status") != "completed":
        raise ValueError("cannot judge an incomplete run")
    frozen = {case.id: case for case in load_frozen_cases(golden_root)}
    source_sha256 = _source_sha256(run, golden_root)
    progress_path = run_root / "judge_progress.json"
    progress: dict[str, Any] = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {
            "judge": JUDGE,
            "protocol_version": PROTOCOL_VERSION,
            "source_sha256": source_sha256,
            "decisions": [],
            "raw_responses": {},
            "usage_by_case": {},
            "review_errors": {},
        }
    )
    if progress.get("judge") not in {JUDGE, "AI model assisted, Codex self-review"}:
        raise ValueError("judge provenance mismatch")
    if progress.get("judge_model") not in {None, judge_model}:
        raise ValueError("judge model mismatch")
    if progress.get("protocol_version") not in {None, PROTOCOL_VERSION}:
        raise ValueError("judge protocol version mismatch")
    if progress.get("source_sha256") != source_sha256:
        raise ValueError("judge progress source mismatch")
    progress["judge"] = JUDGE
    progress["judge_model"] = judge_model
    progress["protocol_version"] = PROTOCOL_VERSION
    progress.setdefault("usage_by_case", {})
    progress.setdefault("review_errors", {})
    progress.setdefault("raw_responses", {})
    unaccounted = progress.setdefault("unaccounted_attempts_by_case", {})
    completed = {item["case_id"] for item in progress["decisions"]}
    case_ids = list(dict.fromkeys(item["case_id"] for item in run["trials"]))
    for case_id in case_ids:
        if case_id in completed:
            continue
        if case_id not in frozen:
            raise ValueError(f"unknown frozen case: {case_id}")
        trials = [item for item in run["trials"] if item["case_id"] == case_id]
        if any(item["case_sha256"] != frozen[case_id].sha256 for item in trials):
            raise ValueError(f"case hash mismatch: {case_id}")
        rubric = (golden_root / "rubrics" / f"{case_id}.yaml").read_text(
            encoding="utf-8"
        )
        replies = []
        for item in trials:
            trace = json.loads((run_root / item["trace"]).read_text(encoding="utf-8"))
            first_messages = trace["trial"]["requests"][0]["messages"]
            context_messages = [
                message["content"]
                for message in first_messages
                if isinstance(message.get("content"), str)
                and "CURRENT RELEVANT VEHICLE CONTEXT" in message["content"]
            ]
            replies.append(
                {
                    "trial_index": item["trial_index"],
                    "replies": trace["trial"]["replies"],
                    "context_shown_to_agent": context_messages,
                    "tool_results": trace["trial"]["tool_calls"],
                }
            )
        prompt = build_judge_prompt(frozen[case_id], rubric, replies)
        if case_id in progress["raw_responses"]:
            content = progress["raw_responses"][case_id]
            if not isinstance(content, str):
                raise ValueError(f"invalid persisted judge response: {case_id}")
        else:
            # Persist before dispatch: exceptions and process interruption may still
            # represent billed requests whose usage cannot be reconstructed.
            unaccounted[case_id] = unaccounted.get(case_id, 0) + 1
            _save_progress(progress_path, progress)
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_MESSAGE,
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            content = response.content
            if not isinstance(content, str):
                raise ValueError(f"invalid judge response type: {case_id}")
            progress["raw_responses"][case_id] = content
            progress["usage_by_case"][case_id] = _total_usage(
                [{"usage": response.usage}]
            )
            unaccounted[case_id] -= 1
            _save_progress(progress_path, progress)
        try:
            parsed = parse_judge_decisions(content, {item["trial_index"] for item in trials})
        except ValueError as exc:
            response_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            progress["review_errors"][case_id] = {
                "error_type": "invalid_judge_output",
                "response_sha256": response_sha256,
                "reason": str(exc)[:240],
            }
            parsed = [
                {
                    "trial_index": item["trial_index"],
                    "verdict": "fail",
                    "evidence": FAIL_CLOSED_EVIDENCE,
                }
                for item in trials
            ]
        for item in parsed:
            item["case_id"] = case_id
        progress["decisions"].extend(parsed)
        _save_progress(progress_path, progress)
    latest_run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if _source_sha256(latest_run, golden_root) != source_sha256:
        raise ValueError("judge source changed during review")
    return review_batch(
        run_root,
        progress["decisions"],
        reviewer=f"{judge_model} AI-assisted review",
        source_sha256=source_sha256,
    )
