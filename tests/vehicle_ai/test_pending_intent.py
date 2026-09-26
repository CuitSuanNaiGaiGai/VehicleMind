from __future__ import annotations

import pytest

from modules.vehicle_ai.agent.pending_intent import (
    classify_pending_intent,
    requested_target,
    search_result_matches_target,
)


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


def test_requested_target_excludes_following_search_constraint() -> None:
    assert (
        requested_target("改去河滨服务区，请重新搜索；找不到就不要导航。")
        == "河滨服务区"
    )


def test_requested_target_preserves_english_internal_spaces_and_case() -> None:
    assert requested_target(
        "change destination to  East Lake Service Area, please search"
    ) == ("East Lake Service Area")


def test_requested_target_keeps_multiple_destinations_together() -> None:
    assert requested_target("改去西湖或河滨服务区") == "西湖或河滨服务区"


@pytest.mark.parametrize(
    "candidate",
    [
        {"poi_id": "river", "name": "河滨服务区"},
        {"poi_id": "river", "display_name_zh": "河滨服务区"},
        {"poi_id": "river", "aliases": ["河滨服务区"]},
        {"poi_id": "河滨服务区", "name": "Other"},
    ],
)
def test_search_match_checks_all_exact_candidate_labels(candidate: dict) -> None:
    assert search_result_matches_target("河滨服务区", candidate)


def test_search_match_ignores_malformed_alias_container() -> None:
    assert not search_result_matches_target(
        "r", {"poi_id": "river", "name": "Other", "aliases": "r"}
    )
