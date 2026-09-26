"""Post-hoc descriptive slices of already validated reviewed trials."""

from __future__ import annotations

from html import escape
from typing import Any, Sequence

CONTEXT_CASE_IDS = tuple(
    [f"C{i:02}" for i in range(1, 7)]
    + [f"R{i:02}" for i in range(1, 7)]
    + [f"X{i:02}" for i in range(1, 11)]
)
QUALITY_CASE_IDS = ("C04", "C05", "R06", "X06", "X07", "X08")
NOTE = (
    "事后描述性子集：分子为完整 rubric 的语义 pass，不是逐事实或人工准确率；"
    "子集重叠，不能相加。完整三次批次分母为 66/18。"
    "逐次 Task Success（任务成功）仅描述重复波动，不是独立实验的置信区间。"
)
HEADERS = (
    "模型",
    "对照",
    "Context Grounding Correctness（上下文依据子集语义通过率）",
    "stale/UNKNOWN Handling（过期/未知子集语义通过率）",
    "逐次 Task Success（任务成功）波动",
)


def summarize_subsets(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count actual intersections; caller validates identities and complete batches."""

    def count(rows: Sequence[dict[str, Any]], *, semantic: bool) -> dict[str, Any]:
        return {
            "passed": sum(
                row["semantic_verdict"] == "pass"
                if semantic
                else row["task_pass"] is True
                for row in rows
            ),
            "total": len(rows),
            "case_ids": sorted({row["case_id"] for row in rows}),
        }

    return {
        "context_grounding": count(
            [row for row in items if row["case_id"] in CONTEXT_CASE_IDS], semantic=True
        ),
        "stale_unknown": count(
            [row for row in items if row["case_id"] in QUALITY_CASE_IDS], semantic=True
        ),
        "per_trial_task_success": {
            str(index): count(
                [row for row in items if row["trial_index"] == index], semantic=False
            )
            for index in sorted({row["trial_index"] for row in items})
        },
    }


def _rate(value: dict[str, Any]) -> str:
    if not value or not value["total"]:
        return "N/A"
    return (
        f"{value['passed']}/{value['total']} ({value['passed'] / value['total']:.1%})"
    )


def _rows(comparison: dict[str, Any]) -> list[list[str]]:
    rows = []
    for provider in ("qwen", "glm"):
        for key, label in (("before", "优化前"), ("after", "A5 后")):
            data = comparison["providers"][provider][key].get("descriptive_subsets", {})
            metrics = []
            for subset in ("context_grounding", "stale_unknown"):
                value = data.get(subset, {})
                ids = ", ".join(value.get("case_ids", [])) or "无"
                metrics.append(f"{_rate(value)}；case IDs：{ids}")
            trials = data.get("per_trial_task_success", {})
            variation = (
                "；".join(
                    f"第 {index} 次：{_rate(trials[index])}"
                    for index in sorted(trials, key=int)
                )
                or "N/A"
            )
            rows.append([provider.upper(), label, *metrics, variation])
    return rows


def render_subset_markdown(comparison: dict[str, Any]) -> str:
    return "\n".join(
        [
            "",
            "## 事后描述性子集与重复波动",
            "",
            NOTE,
            "",
            "| " + " | ".join(HEADERS) + " |",
            "|---|---|---|---|---|",
            *("| " + " | ".join(row) + " |" for row in _rows(comparison)),
            "",
        ]
    )


def render_subset_html(comparison: dict[str, Any]) -> str:
    headers = "".join(f"<th>{escape(value)}</th>" for value in HEADERS)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in _rows(comparison)
    )
    return (
        '<section class="panel"><h2>事后描述性子集与重复波动</h2>'
        f'<p>{escape(NOTE)}</p><div class="scroll"><table><thead><tr>{headers}'
        f"</tr></thead><tbody>{rows}</tbody></table></div></section>"
    )
