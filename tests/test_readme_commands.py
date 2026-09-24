from __future__ import annotations

import importlib.util
from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"


def test_readme_local_python_modules_exist() -> None:
    content = README.read_text(encoding="utf-8")
    modules = set(re.findall(r"python -m (apps\.[A-Za-z0-9_.]+)", content))

    assert modules
    assert {
        module for module in modules if importlib.util.find_spec(module) is None
    } == set()


def test_readme_local_script_paths_exist() -> None:
    content = README.read_text(encoding="utf-8")
    scripts = set(re.findall(r"python (scripts/[A-Za-z0-9_./-]+\.py)", content))

    assert scripts
    assert {
        script for script in scripts if not (REPOSITORY_ROOT / script).is_file()
    } == set()


def test_readme_does_not_reference_renamed_demo_modules() -> None:
    content = README.read_text(encoding="utf-8")

    assert "phone_test" not in content
    assert "real_perception_test" not in content


def test_readme_explains_replay_outcome_and_optional_online_cost() -> None:
    content = README.read_text(encoding="utf-8")
    quickstart = content.split('id="quickstart"', 1)[1].split('id="structure"', 1)[0]
    for expected in (
        "assets/scenarios/drowsy_rest_stop.yaml",
        "apps.vehicle_ai_demo.replay_demo",
        'open "runs/$VM_RUN_ID/report.html"',
        "运行后应看到",
        "DROWSY",
        "HIGH",
        "待确认",
        "模拟导航",
        "常见问题",
        ".env.example",
        "API 调用会产生费用",
        "modules.vehicle_ai.evaluation.cli",
    ):
        assert expected in quickstart, expected
