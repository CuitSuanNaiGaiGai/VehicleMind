from __future__ import annotations

import pytest

from modules.vehicle_ai.agent.pending_intent import classify_pending_intent


@pytest.mark.parametrize(
    "text", ["取消", "不要了", "cancel", "No thanks", "  不要了。 "]
)
def test_explicit_refusal_is_classified(text: str) -> None:
    assert classify_pending_intent(text) == "reject"


@pytest.mark.parametrize(
    "text",
    [
        "换成东湖服务区",
        "改去东湖服务区",
        "instead go east",
        "change destination to east",
    ],
)
def test_target_change_is_classified(text: str) -> None:
    assert classify_pending_intent(text) == "change_target"


@pytest.mark.parametrize(
    "text",
    ["好的", "可能不去了", "帮我看看路况", "不要开窗", "cancel navigation later"],
)
def test_ambiguous_or_unrelated_text_is_not_classified(text: str) -> None:
    assert classify_pending_intent(text) is None
