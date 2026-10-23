"""Export frozen evaluation suites into deterministic promptfoo configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from modules.vehicle_ai.evaluation.batch import load_frozen_cases
from modules.vehicle_ai.evaluation.harness.injection import (
    DEFAULT_SECURITY_SUITE,
    load_security_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_PATH = "modules/vehicle_ai/evaluation/harness/provider.py"
ASSERTIONS_PATH = "modules/vehicle_ai/evaluation/harness/assertions.py"
DEFAULT_OUTPUT_DIR = Path("eval/promptfoo")
DEFAULT_DATASET_ROOT = Path("scenarios/agent_eval/golden")
DEFAULT_BACKENDS = ("qwen", "glm")


def _file_link(target: str, function: str | None = None) -> str:
    """file:// paths resolve relative to the promptfoo config location."""
    link = f"file://../../{target}"
    return f"{link}:{function}" if function else link


def _expected_json(expected: dict[str, Any]) -> str:
    return json.dumps(expected, ensure_ascii=False, sort_keys=True)


def build_injection_config() -> dict[str, Any]:
    suite = load_security_suite(REPO_ROOT / DEFAULT_SECURITY_SUITE)
    tests = [
        {
            "description": f"{case['id']} ({case['source']})",
            "vars": {
                "case_id": case["id"],
                "expected_json": _expected_json(case["expected"]),
            },
        }
        for case in suite["cases"]
    ]
    return {
        "description": (
            "VehicleMind 注入门禁：对抗载荷不得绕过执行层确认门"
            "（确定性、离线、零 API 成本）"
        ),
        "prompts": ["{{case_id}}"],
        "providers": [
            {
                "id": _file_link(PROVIDER_PATH),
                "label": "injection-gate",
                "config": {
                    "mode": "injection_gate",
                    "security_suite": DEFAULT_SECURITY_SUITE,
                },
            }
        ],
        "defaultTest": {
            "assert": [
                {
                    "type": "python",
                    "value": _file_link(ASSERTIONS_PATH, "assert_injection_gate"),
                    "timeout": 120,
                }
            ]
        },
        "tests": tests,
    }


def build_agent_eval_config(
    dataset_root: Path,
    backends: tuple[str, ...] = DEFAULT_BACKENDS,
    *,
    judged: bool = False,
) -> dict[str, Any]:
    cases = load_frozen_cases(dataset_root)
    relative_root = dataset_root.resolve().relative_to(REPO_ROOT).as_posix()
    tests = [
        {
            "description": f"{case.id} · {case.category} · {case.split}",
            "vars": {
                "case_id": case.id,
                "expected_json": _expected_json(case.expected),
                **({"dataset_root": relative_root} if judged else {}),
            },
        }
        for case in cases
    ]
    providers = [
        {
            "id": _file_link(PROVIDER_PATH),
            "label": backend,
            "config": {
                "mode": "agent_eval",
                "backend": backend,
                "dataset_root": relative_root,
            },
        }
        for backend in backends
    ]
    asserts = [
        {
            "type": "python",
            "value": _file_link(ASSERTIONS_PATH, "assert_agent_case"),
            "timeout": 300,
        }
    ]
    if judged:
        asserts.append(
            {
                "type": "python",
                "value": _file_link(ASSERTIONS_PATH, "assert_rubric_semantics"),
                "timeout": 600,
            }
        )
    if judged:
        description = (
            "VehicleMind 金标集在线评测（含 rubric 语义判分）：判分与 batch_judge "
            "共用同一协议；judge 调用产生额外真实 API 费用"
        )
    else:
        description = (
            "VehicleMind 金标集在线评测：机械断言通过不等于任务成功，"
            "语义仍需逐条复核（产生真实 API 费用）"
        )
    return {
        "description": description,
        "prompts": ["{{case_id}}"],
        "providers": providers,
        "defaultTest": {"assert": asserts},
        "tests": tests,
    }


def _render(config: dict[str, Any]) -> str:
    return yaml.dump(
        config,
        allow_unicode=True,
        sort_keys=False,
        width=4096,
        default_flow_style=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 promptfoo 评测配置")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="校验已生成配置与冻结场景一致，不做写入",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    dataset_root = (
        args.dataset_root
        if args.dataset_root.is_absolute()
        else REPO_ROOT / args.dataset_root
    )
    configs = {
        "injection_gate.yaml": build_injection_config(),
        "agent_eval.yaml": build_agent_eval_config(dataset_root),
        "agent_eval_rubric.yaml": build_agent_eval_config(dataset_root, judged=True),
    }

    stale: list[Path] = []
    for name, config in configs.items():
        target = output_dir / name
        text = _render(config)
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                stale.append(target)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(f"wrote {target.relative_to(REPO_ROOT)}")

    if args.check:
        if stale:
            print("Stale promptfoo configs (rerun without --check):")
            for path in stale:
                print(f"- {path}")
            return 1
        print("Promptfoo configs match the frozen suites.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
