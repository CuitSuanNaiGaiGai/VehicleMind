from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import TrialResult
from modules.vehicle_ai.evaluation.showcase import render_showcase


def write_report(
    destination: Path,
    case: EvaluationCase,
    trial: TrialResult,
    grade: dict,
) -> None:
    """Write one trace and a deliberately provisional Chinese result summary."""
    destination.mkdir(parents=True, exist_ok=False)
    payload = {
        "case": {
            "id": case.id,
            "split": case.split,
            "category": case.category,
            "review_status": case.review_status,
            "steps": case.steps,
            "expected": case.expected,
        },
        "trial": asdict(trial),
        "grade": grade,
    }
    (destination / "trial.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "report.html").write_text(
        render_showcase(case, trial, grade), encoding="utf-8"
    )
    (destination / "report.md").write_text(
        f"# Agent 评估单次报告：{case.id}\n\n"
        f"- 模型：{trial.provider} / {trial.model}\n"
        f"- 数据状态：{case.review_status}"
        f"（{'候选数据不可称为正式金标' if case.review_status == 'candidate' else 'AI 自审仅为内部基准，非独立人工标注'}）\n"
        f"- 判定：{grade['status']}\n"
        f"- 工具选择：{grade['tool_selection']}\n"
        f"- 参数匹配：{grade['argument_match']}\n"
        f"- 最终状态：{grade['final_state']}\n"
        f"- 请求次数：{trial.request_count}\n"
        f"- 总耗时：{trial.latency_ms:.1f} ms\n"
        f"- 错误类别：{trial.error or '无'}\n\n"
        "详细轨迹见 `trial.json`；`needs_review` 不计为成功。\n",
        encoding="utf-8",
    )
