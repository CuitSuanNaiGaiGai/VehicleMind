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
        names = wheel.namelist()
        assert "modules/config/cabin.yaml" in names
        assert "modules/config/perception.yaml" in names
        license_paths = [
            name for name in names if name.endswith(".dist-info/licenses/LICENSE")
        ]
        assert len(license_paths) == 1
        metadata_path = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        assert (
            wheel.read(license_paths[0]) == (REPOSITORY_ROOT / "LICENSE").read_bytes()
        )
        assert "License-Expression: Apache-2.0" in wheel.read(metadata_path).decode(
            "utf-8"
        )
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
                "from modules.config import PerceptionConfig; "
                "import modules.config.cabin as cabin; "
                "assert Path(cabin.__file__).is_relative_to(Path.cwd()); "
                "assert CabinPerceptionConfig.load_default().eye.ear_threshold == 0.21; "
                "assert PerceptionConfig.load_default().driving.nms_threshold == 0.45"
            ),
        ],
        cwd=extracted,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
