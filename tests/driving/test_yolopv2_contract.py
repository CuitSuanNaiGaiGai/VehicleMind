from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _module(relative_path: str) -> ast.Module:
    path = REPOSITORY_ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_panoptic_detector_preserves_public_detect_contract() -> None:
    tree = _module("modules/driving/perception/panoptic_detector.py")
    detector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PanopticDrivingDetector"
    )
    detect = next(
        node
        for node in detector.body
        if isinstance(node, ast.FunctionDef) and node.name == "detect"
    )

    assert [argument.arg for argument in detect.args.args] == ["self", "frame"]
    assert ast.unparse(detect.returns) == "DrivingSceneResult"


def test_panoptic_responsibilities_are_split_into_focused_modules() -> None:
    expected_functions = {
        "modules/driving/perception/preprocess.py": "preprocess_frame",
        "modules/driving/perception/postprocess.py": "decode_masks",
        "modules/driving/perception/runtime.py": "warm_up_session",
    }

    for relative_path, function_name in expected_functions.items():
        functions = {
            node.name
            for node in _module(relative_path).body
            if isinstance(node, ast.FunctionDef)
        }
        assert function_name in functions
