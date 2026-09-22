from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_modules_are_not_named_like_tests():
    offenders = sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "apps").rglob("*_test.py")
    )

    assert offenders == []


def test_implementation_modules_are_not_empty_placeholders():
    empty_modules = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "modules").rglob("*.py"))
        if path.name != "__init__.py" and not path.read_text(encoding="utf-8").strip()
    ]

    assert empty_modules == []


def test_ci_checks_python_formatting() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ruff format --check apps modules scripts tests" in workflow
