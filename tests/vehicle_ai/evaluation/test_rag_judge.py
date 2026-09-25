from modules.vehicle_ai.evaluation.rag_judge import (
    parse_abstention_review,
    parse_citation_review,
)
from modules.vehicle_ai.evaluation.rag_runner import RagTrial


def test_review_rejects_unknown_or_unsupported_source() -> None:
    raw = '{"facts":[{"fact":"应停车休息","citation_ids":["K001"],"supported":true,"review_note":"片段支持"},{"fact":"已经导航","citation_ids":["K999"],"supported":true,"review_note":"模型认为支持"}]}'
    reviews = parse_citation_review("R01", raw, {"K001"})
    assert len(reviews) == 2
    assert reviews[0].supported
    assert not reviews[1].supported
    assert all(review.reviewer_type == "ai_assisted" for review in reviews)


def test_review_malformed_output_is_not_silently_scored() -> None:
    try:
        parse_citation_review("R01", "not json", {"K001"})
    except ValueError:
        pass
    else:
        raise AssertionError("malformed review must fail")


def test_abstention_judge_tracks_semantic_scope_decision() -> None:
    trial = RagTrial(
        "R27",
        "VehicleMind 事件建议如何组织",
        "vehicle_common",
        False,
        ("K015",),
        "profile_mismatch",
        (),
        None,
        "通用资料不足以确定项目实现。",
        True,
        True,
        None,
        100,
        None,
    )
    review = parse_abstention_review(
        trial, '{"adequate":true,"review_note":"明确说明 profile 未覆盖项目实现"}'
    )
    assert review.adequate and review.reviewer_type == "ai_assisted"
