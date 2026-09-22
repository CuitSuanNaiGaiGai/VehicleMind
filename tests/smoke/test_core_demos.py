from __future__ import annotations

import os
import subprocess
import sys

from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CORE_DEMOS = (
    "apps.vehicle_ai_demo.context_demo",
    "apps.vehicle_ai_demo.context_manager_demo",
    "apps.vehicle_ai_demo.context_selector_demo",
    "apps.vehicle_ai_demo.event_demo",
    "apps.vehicle_ai_demo.tool_demo",
    "apps.vehicle_ai_demo.cabin_adapter_demo",
)


@pytest.mark.parametrize("module", CORE_DEMOS)
def test_core_demo_exits_successfully_without_external_services(module: str) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "DASHSCOPE_API_KEY",
            "GLM_API_KEY",
            "OPENAI_API_KEY",
        }
    }
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, (
        f"{module} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
