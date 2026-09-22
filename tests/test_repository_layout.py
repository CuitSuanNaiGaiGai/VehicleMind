from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_modules_are_not_named_like_tests():
    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "apps").rglob("*_test.py")
    )

    assert offenders == []
