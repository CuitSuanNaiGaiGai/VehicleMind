from __future__ import annotations

import hashlib
import re
import shutil
import tempfile

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from modules.config.cabin import CabinPerceptionConfig
from modules.config.events import EventTimingConfig
from modules.config.perception import PerceptionConfig


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    created_at_utc: str
    git_commit: str
    dirty: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or self.run_id in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.run_id) is None
        ):
            raise ValueError(
                "run_id must be a safe 1-64 character directory name containing "
                "only letters, numbers, dot, underscore, or hyphen"
            )
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.strip():
            raise ValueError("created_at_utc must be a non-empty string")
        if (
            not isinstance(self.git_commit, str)
            or len(self.git_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.git_commit.lower()
            )
        ):
            raise ValueError("git_commit must be a 40-character hexadecimal commit")
        if not isinstance(self.dirty, bool):
            raise ValueError("dirty must be a boolean")


@dataclass(frozen=True)
class RunArtifactPaths:
    manifest: Path
    run_card: Path


@dataclass(frozen=True)
class RunArtifactTexts:
    manifest: str
    run_card: str


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_yaml(value: Mapping[str, object]) -> str:
    return yaml.safe_dump(_plain(value), sort_keys=True, allow_unicode=True)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"snapshot {name} must be a mapping")
    return value


def build_run_snapshot(
    *,
    cabin: CabinPerceptionConfig,
    perception: PerceptionConfig,
    events: EventTimingConfig | None = None,
    overrides: Mapping[str, object],
    assets: Sequence[Mapping[str, object]],
    provenance: RunProvenance,
    allow_dirty: bool = False,
) -> dict[str, object]:
    if provenance.dirty and not allow_dirty:
        raise ValueError(
            "formal run refused because of a dirty worktree; "
            "use allow_dirty only for debugging"
        )

    configuration = {
        "cabin": _plain(asdict(cabin)),
        "perception": _plain(asdict(perception)),
        "overrides": _plain(dict(overrides)),
    }
    if events is not None:
        configuration["events"] = _plain(asdict(events))
    config_sha256 = hashlib.sha256(
        _canonical_yaml(configuration).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "provenance": _plain(asdict(provenance)),
        "config_sha256": config_sha256,
        "resolved_config": {
            "cabin": configuration["cabin"],
            "perception": configuration["perception"],
            **({"events": configuration["events"]} if events is not None else {}),
        },
        "overrides": configuration["overrides"],
        "model_assets": [_plain(dict(asset)) for asset in assets],
    }


def select_asset_records(
    manifest_path: Path,
    asset_ids: Sequence[str],
) -> list[dict[str, object]]:
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("assets"), list):
        raise ValueError("model asset manifest must contain an assets list")

    assets_by_id = {
        str(asset["id"]): asset
        for asset in document["assets"]
        if isinstance(asset, dict) and "id" in asset
    }
    records: list[dict[str, object]] = []
    for asset_id in asset_ids:
        if asset_id not in assets_by_id:
            raise ValueError(f"unknown model asset id: {asset_id}")
        asset = assets_by_id[asset_id]
        records.append(
            {
                "id": str(asset["id"]),
                "expected_path": str(asset["expected_path"]),
                "size_bytes": int(asset["size_bytes"]),
                "sha256": str(asset["sha256"]),
            }
        )
    return records


def _render_run_card(snapshot: Mapping[str, object]) -> str:
    provenance = _require_mapping(snapshot["provenance"], "provenance")
    assets = snapshot.get("model_assets", [])
    if not isinstance(assets, list):
        raise ValueError("snapshot model_assets must be a list")
    resolved = _require_mapping(snapshot.get("resolved_config"), "resolved_config")
    perception = _require_mapping(
        resolved.get("perception"), "perception configuration"
    )
    phone = _require_mapping(perception.get("phone"), "phone configuration")
    driving = _require_mapping(perception.get("driving"), "driving configuration")
    lane = _require_mapping(perception.get("lane"), "lane configuration")

    lines = [
        "# VehicleMind 运行卡",
        "",
        "| 字段 | 数值 |",
        "|---|---|",
        f"| 运行标识 | {provenance['run_id']} |",
        f"| 创建时间（UTC） | {provenance['created_at_utc']} |",
        f"| Git 提交 | {provenance['git_commit']} |",
        f"| 工作树有未提交改动 | {'是' if provenance['dirty'] else '否'} |",
        f"| 配置 SHA-256 | {snapshot['config_sha256']} |",
        "",
        "## 关键解析参数",
        "",
        "| 参数 | 数值 |",
        "|---|---|",
        f"| 手机检测置信度阈值 | {phone['confidence_threshold']} |",
        f"| 道路检测分数阈值 | {driving['score_threshold']} |",
        f"| 道路检测 NMS 阈值 | {driving['nms_threshold']} |",
        f"| 道路处理尺寸 | {driving['work_width']} × {driving['work_height']} |",
        f"| 车道平滑系数 | {lane['smoothing']} |",
        f"| 车道 Canny 阈值 | {lane['canny_low']} / {lane['canny_high']} |",
        "",
        "## 已选择的模型资产",
        "",
    ]
    if assets:
        lines.extend(
            [
                "| 资产 | SHA-256 | 预期路径 |",
                "|---|---|---|",
            ]
        )
        for asset in assets:
            if not isinstance(asset, Mapping):
                raise ValueError("each model asset must be a mapping")
            lines.append(
                f"| {asset['id']} | {asset['sha256']} | {asset['expected_path']} |"
            )
    else:
        lines.append("_未选择模型资产。_")

    if "events" in resolved:
        lines.extend(
            [
                "",
                "## 事件时序",
                "",
                "事件持续时间、冷却时间与采样间隔设置均记录在 "
                "`resolved_config.yaml` 中。",
            ]
        )

    lines.extend(
        [
            "",
            "## 解析后配置",
            "",
            "- 舱内配置：记录在 `resolved_config.yaml` 中",
            "- 感知配置：记录在 `resolved_config.yaml` 中",
            "- 命令行覆盖项：记录在 `resolved_config.yaml` 中",
            "",
            "## 结果状态",
            "",
            "本次运行制品未记录基准评测指标。",
            "准确率、延迟和案例图必须由经过验证的评测流程产生。",
            "",
        ]
    )
    return "\n".join(lines)


def render_run_artifacts(
    snapshot: Mapping[str, object],
) -> RunArtifactTexts:
    return RunArtifactTexts(
        manifest=yaml.safe_dump(
            _plain(snapshot),
            sort_keys=False,
            allow_unicode=True,
        ),
        run_card=_render_run_card(snapshot),
    )


def write_run_artifacts(
    output_dir: Path,
    snapshot: Mapping[str, object],
) -> RunArtifactPaths:
    manifest = output_dir / "resolved_config.yaml"
    run_card = output_dir / "run_card.md"
    if output_dir.exists():
        raise FileExistsError(f"run artifact directory already exists: {output_dir}")

    rendered = render_run_artifacts(snapshot)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        (staging_dir / manifest.name).write_text(rendered.manifest, encoding="utf-8")
        (staging_dir / run_card.name).write_text(rendered.run_card, encoding="utf-8")
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    return RunArtifactPaths(manifest=manifest, run_card=run_card)
