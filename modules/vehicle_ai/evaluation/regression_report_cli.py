"""Build the A6 Chinese regression report from complete, hash-checked batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modules.vehicle_ai.evaluation.batch import load_frozen_cases
from modules.vehicle_ai.evaluation.regression_report import (
    aggregate_batch,
    build_comparison,
    write_report,
)


def _stage_evidence() -> list[dict[str, str]]:
    return [
        {
            "stage": "A1",
            "name": "上下文/异常可靠性（Reliability）",
            "value": "在线失败逐条保留；无总体 A1 单分",
            "scope": "A1 验收与历史回归，AI 自审合成场景",
        },
        {
            "stage": "A2",
            "name": "Recommendation Appropriateness（建议适配率）",
            "value": "1/2；Confirmation Compliance（确认合规率）1/1；未经确认敏感执行 0",
            "scope": "Qwen，3 个策略场景单次在线运行；字面规则评分，见 A2 报告",
        },
        {
            "stage": "A3",
            "name": "Recall@5 / Citation Support（证据召回/引用支持）",
            "value": "20/20；69/74；范围弃答定向复测 5/5；来源越界 0",
            "scope": "Qwen，30 题主评测；范围修正后 5 题单独复测，AI 辅助审查",
        },
        {
            "stage": "A4",
            "name": "Event Retrieval Accuracy / Temporal Confusion（事件检索/时间混淆）",
            "value": "8/8 字段（4/4 查询）；注入检查 0/1",
            "scope": "结构化合成行程事件，不是在线自然语言金标",
        },
        {
            "stage": "A5",
            "name": "Recovery Success / Safe Stop / Budget（恢复/停止/预算）",
            "value": "1/1；4/4；7/7；固定场景断言 7/7",
            "scope": "确定性故障注入、模拟 POI 与模拟车机，不是在线模型成绩",
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 A5 后在线 Agent 回归报告")
    for name in ("before-qwen", "before-glm", "after-qwen", "after-glm"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--golden", type=Path, default=Path("scenarios/agent_eval/golden")
    )
    parser.add_argument(
        "--pricing",
        type=Path,
        default=Path("docs/reports/2026-09-26-model-pricing-snapshot.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reports/a6-post-a5-regression"),
    )
    args = parser.parse_args(argv)
    cases = load_frozen_cases(args.golden)
    case_ids = tuple(case.id for case in cases)
    case_hashes = {case.id: case.sha256 for case in cases}
    manifest_sha256 = hashlib.sha256(
        (args.golden / "manifest.yaml").read_bytes()
    ).hexdigest()
    before = {
        "qwen": aggregate_batch(
            args.before_qwen,
            expected_case_ids=case_ids,
            expected_case_hashes=case_hashes,
            expected_manifest_sha256=manifest_sha256,
            repetitions=3,
        ),
        "glm": aggregate_batch(
            args.before_glm,
            expected_case_ids=case_ids,
            expected_case_hashes=case_hashes,
            expected_manifest_sha256=manifest_sha256,
            repetitions=3,
        ),
    }
    after = {
        "qwen": aggregate_batch(
            args.after_qwen,
            expected_case_ids=case_ids,
            expected_case_hashes=case_hashes,
            expected_manifest_sha256=manifest_sha256,
            repetitions=3,
            require_clean_source=True,
        ),
        "glm": aggregate_batch(
            args.after_glm,
            expected_case_ids=case_ids,
            expected_case_hashes=case_hashes,
            expected_manifest_sha256=manifest_sha256,
            repetitions=3,
            require_clean_source=True,
        ),
    }
    pricing = json.loads(args.pricing.read_text(encoding="utf-8"))
    report = build_comparison(
        before,
        after,
        stage_evidence=_stage_evidence(),
        pricing=pricing,
    )
    report["golden_manifest_sha256"] = manifest_sha256
    write_report(report, args.output_dir)
    print(f"已生成中文回归证据：{args.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
