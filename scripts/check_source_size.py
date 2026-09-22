from __future__ import annotations

import ast
import sys

from pathlib import Path


MODULE_MAX_LINES = 500
APP_HELPER_MAX_LINES = 500
APP_ENTRY_POINT_MAX_LINES = 300


def _is_entry_point(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False

    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
            return True
    return False


def _line_limit(path: Path, root: Path) -> int:
    relative = path.relative_to(root)
    if relative.parts[0] == "modules":
        return MODULE_MAX_LINES
    if path.name.endswith("_demo.py") and _is_entry_point(path):
        return APP_ENTRY_POINT_MAX_LINES
    return APP_HELPER_MAX_LINES


def find_oversized_sources(root: Path | None = None) -> list[str]:
    repository_root = root or Path(__file__).resolve().parents[1]
    oversized: list[str] = []

    for source_root in ("apps", "modules"):
        for path in sorted((repository_root / source_root).rglob("*.py")):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            limit = _line_limit(path, repository_root)
            if line_count > limit:
                relative = path.relative_to(repository_root).as_posix()
                oversized.append(f"{relative}: {line_count} lines (limit {limit})")

    return oversized


def main() -> int:
    oversized = find_oversized_sources()
    if oversized:
        print("Oversized Python sources:")
        for item in oversized:
            print(f"- {item}")
        return 1

    print("All Python sources satisfy the size policy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
