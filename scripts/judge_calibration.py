"""Report Cohen's kappa between reference labels and the recorded judge."""

from __future__ import annotations

import argparse
from pathlib import Path

from modules.vehicle_ai.evaluation.judge_calibration import calibrate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = Path("eval/judge_calibration/labels.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="校准 judge 与参考标签的一致性（Cohen's kappa，零 API 成本）"
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--min-kappa",
        type=float,
        default=None,
        help="低于该阈值时以非零退出码结束",
    )
    args = parser.parse_args(argv)

    labels = args.labels if args.labels.is_absolute() else REPO_ROOT / args.labels
    report = calibrate(labels)
    provenance = report["label_provenance"]
    run = report["run"]
    print(
        f"参考标签：{provenance['labeler']}（独立人工标注：{provenance['independent_human_review']}）"
    )
    print(
        f"被校准 run：{run['provider']}/{run['model']} source={run['source_sha256'][:12]}…"
    )
    print(f"样本数：{report['total']}")
    print(f"一致率：{report['agreement']:.3f}")
    print(f"Cohen's kappa：{report['cohens_kappa']:.3f}")
    print(f"混淆（参考/judge）：{report['confusion']}")
    for item in report["disagreements"]:
        print(
            f"- 分歧 {item['case_id']}/trial-{item['trial_index']}: "
            f"参考={item['reference']} judge={item['judge']}；{item['note']}"
        )
    if args.min_kappa is not None and report["cohens_kappa"] < args.min_kappa:
        print(f"kappa {report['cohens_kappa']:.3f} 低于阈值 {args.min_kappa}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
