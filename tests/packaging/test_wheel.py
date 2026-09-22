from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_built_wheel_contains_loadable_default_cabin_config(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [
            "uv",
            "build",
            "--offline",
            "--no-build-isolation",
            "--wheel",
            "--out-dir",
            str(wheel_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_path = next(wheel_dir.glob("vehiclemind-*.whl"))
    extracted = tmp_path / "installed"
    with zipfile.ZipFile(wheel_path) as wheel:
        assert "modules/config/cabin.yaml" in wheel.namelist()
        wheel.extractall(extracted)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(extracted)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from modules.config import CabinPerceptionConfig; "
                "import modules.config.cabin as cabin; "
                "assert Path(cabin.__file__).is_relative_to(Path.cwd()); "
                "assert CabinPerceptionConfig.load_default().eye.ear_threshold == 0.21"
            ),
        ],
        cwd=extracted,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
