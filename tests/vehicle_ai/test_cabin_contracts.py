from __future__ import annotations

import ast
from pathlib import Path

from modules.vehicle_ai.context import ContextManager, DriverPresence
from modules.vehicle_ai.integration import CabinContextAdapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parse(relative_path: str) -> ast.Module:
    source_path = REPOSITORY_ROOT / relative_path
    return ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )


def test_vehicle_ai_demos_use_semantic_presence() -> None:
    demo_paths = (
        "apps/vehicle_ai_demo/cabin_adapter_demo.py",
        "apps/vehicle_ai_demo/runtime_demo.py",
    )

    stale_arguments: list[str] = []
    for relative_path in demo_paths:
        for node in ast.walk(_parse(relative_path)):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "face_present":
                    stale_arguments.append(
                        f"{relative_path}:{keyword.lineno}"
                    )

    assert stale_arguments == [], (
        "frame-level face_present must not be used as semantic driver "
        f"presence: {stale_arguments}"
    )


def test_cabin_snapshot_models_unobserved_eye_state() -> None:
    tree = _parse("modules/cabin/perception_service.py")

    annotation = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "CabinPerceptionSnapshot":
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "eye_closed"
            ):
                annotation = ast.unparse(statement.annotation)

    assert annotation in {"bool | None", "None | bool"}


def test_cabin_adapter_preserves_unknown_eye_state() -> None:
    manager = ContextManager()
    adapter = CabinContextAdapter(manager)

    adapter.update(
        presence="PRESENT",
        driver_state="NORMAL",
        risk="LOW",
        eye_closed=None,
    )

    driver = manager.get_context().driver
    assert driver.presence is DriverPresence.PRESENT
    assert driver.eye_closed is None
