from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modules.vehicle_ai.agent.pending_intent import requested_target
from modules.vehicle_ai.agent.target_resolution import (
    TargetResolution,
    resolve_target,
    target_resolution_message,
)


@pytest.mark.parametrize(
    ("target", "candidate"),
    [
        ("河滨服务区", {"poi_id": "river", "name": "河滨服务区"}),
        (
            "河滨服务区",
            {"poi_id": "river", "display_name_zh": "河滨服务区"},
        ),
        (
            "河滨服务区",
            {"poi_id": "river", "aliases": ["河滨服务区"]},
        ),
        (
            "river",
            {"poi_id": "river", "name": "Riverside"},
        ),
    ],
)
def test_matches_only_exact_current_candidate_labels(
    target: str, candidate: dict[str, object]
) -> None:
    resolution = resolve_target(target, [candidate])

    assert resolution.status == "matched"
    assert resolution.candidate is candidate
    assert resolution.candidate_ids == ("river",)


def test_matching_normalizes_case_and_outer_whitespace() -> None:
    candidate = {"poi_id": "east", "name": " East Lake "}

    resolution = resolve_target("  EAST LAKE  ", [candidate])

    assert resolution.status == "matched"
    assert resolution.candidate is candidate


@pytest.mark.parametrize("target", ["河滨", "东湖服务区"])
def test_partial_or_unlisted_target_is_not_resolved(target: str) -> None:
    resolution = resolve_target(
        target,
        [{"poi_id": "river", "name": "河滨服务区", "aliases": ["河滨服务区"]}],
    )

    assert resolution.status == "not_found"
    assert resolution.candidate is None
    assert resolution.candidate_ids == ()


def test_duplicate_rows_for_one_id_do_not_create_false_ambiguity() -> None:
    first = {"poi_id": "river", "name": "河滨服务区"}
    duplicate = {"poi_id": "river", "aliases": ["河滨服务区"]}

    resolution = resolve_target("河滨服务区", [first, duplicate])

    assert resolution.status == "matched"
    assert resolution.candidate is first
    assert resolution.candidate_ids == ("river",)


def test_same_exact_alias_for_two_ids_is_ambiguous() -> None:
    resolution = resolve_target(
        "河滨服务区",
        [
            {"poi_id": "river-east", "aliases": ["河滨服务区"]},
            {"poi_id": "river-west", "display_name_zh": "河滨服务区"},
        ],
    )

    assert resolution.status == "ambiguous"
    assert resolution.candidate is None
    assert resolution.candidate_ids == ("river-east", "river-west")


def test_malformed_alias_container_and_values_are_ignored() -> None:
    malformed = {"poi_id": "bad", "name": "Other", "aliases": "r"}
    valid = {"poi_id": "river", "aliases": [None, 7, {"label": "r"}, "R"]}

    assert resolve_target("r", [malformed]).status == "not_found"
    assert resolve_target("r", [valid]).status == "matched"
    assert resolve_target("r", [{"poi_id": "broken", "aliases": None}]).status == (
        "not_found"
    )


def test_explicit_alternative_destinations_are_not_reduced_to_first_clause() -> None:
    target = requested_target("改去西湖服务区，或者河滨服务区；找不到就不要导航。")

    assert target == "西湖服务区，或者河滨服务区；找不到就不要导航"
    resolution = resolve_target(
        target,
        [
            {"poi_id": "west", "name": "西湖服务区"},
            {"poi_id": "river", "name": "河滨服务区"},
        ],
    )
    assert resolution.status == "not_found"


def test_resolution_record_is_frozen() -> None:
    resolution = TargetResolution(
        status="not_found", target="Unknown", candidate=None, candidate_ids=()
    )

    with pytest.raises(FrozenInstanceError):
        resolution.status = "matched"  # type: ignore[misc]


def test_matched_reply_names_the_tool_candidate_and_requires_confirmation() -> None:
    resolution = resolve_target(
        "河滨服务区",
        [
            {
                "poi_id": "river",
                "name": "Riverside",
                "display_name_zh": "河滨服务区",
            }
        ],
    )

    message = target_resolution_message(resolution, pending_created=True)

    assert "河滨服务区" in message
    assert "待确认" in message
    assert "确认" in message


def test_matched_reply_reports_pending_creation_failure() -> None:
    resolution = resolve_target(
        "河滨服务区", [{"poi_id": "river", "display_name_zh": "河滨服务区"}]
    )

    message = target_resolution_message(resolution, pending_created=False)

    assert "河滨服务区" in message
    assert "未能创建" in message
    assert "不能导航" in message


@pytest.mark.parametrize("status", ["not_found", "ambiguous"])
def test_unresolved_reply_does_not_claim_a_pending_action(status: str) -> None:
    resolution: TargetResolution = TargetResolution(
        status=status,  # type: ignore[arg-type]
        target="河滨服务区",
        candidate=None,
        candidate_ids=("river", "river-2") if status == "ambiguous" else (),
    )

    message = target_resolution_message(resolution, pending_created=False)

    assert "河滨服务区" in message
    assert "待确认导航" in message
