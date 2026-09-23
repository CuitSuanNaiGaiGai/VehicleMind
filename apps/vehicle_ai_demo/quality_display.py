from __future__ import annotations


def format_age(age_seconds: float | None) -> str:
    """Render an unobserved domain without inventing a numeric age."""

    return "未观测" if age_seconds is None else f"{age_seconds:.3f} 秒"
