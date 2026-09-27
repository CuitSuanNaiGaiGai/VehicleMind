import json

import pytest

from modules.vehicle_ai.agent.budget import TurnBudget
from modules.vehicle_ai.agent.dialogue_interpreter import (
    DialogueIntent,
    Interpretation,
    interpret_turn,
    validate_proposal,
)
from modules.vehicle_ai.agent.dialogue_reducer import reduce_dialogue
from modules.vehicle_ai.agent.task_state import AgentTask, TaskStatus
from modules.vehicle_ai.llm.base import LLMResponse, LLMToolCall


def proposal(intent, changes=(), reference=None, evidence="", reason=None):
    return {
        "intent": intent,
        "changes": list(changes),
        "reference": reference,
        "evidence": evidence,
        "clarification_reason": reason,
    }


def change(field, value, evidence, op="SET"):
    return {"field": field, "op": op, "value": value, "evidence": evidence}


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_with_timeout(self, messages, tools=None, *, timeout_seconds):
        self.calls.append((messages, tools, timeout_seconds))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def budget():
    return TurnBudget(60.0, 4, lambda: 0.0, max_model_calls=3)


def test_dialogue_intent_has_the_eight_protocol_values():
    assert {item.value for item in DialogueIntent} == {
        "START",
        "UPDATE_CONSTRAINTS",
        "SELECT",
        "ASK_CANDIDATE",
        "SIDE_QUESTION",
        "RESUME",
        "CANCEL",
        "UNCLEAR",
    }


def test_canonical_controls_and_ordinals_do_not_call_the_model():
    examples = [
        ("取消本次任务。", "CANCEL", None),
        ("继续任务！", "RESUME", None),
        ("就去第2个", "SELECT", 2),
        ("选择第十个。", "SELECT", 10),
        ("第3个多远？", "ASK_CANDIDATE", 3),
    ]
    for text, expected, index in examples:
        client = Client([])
        result = interpret_turn(client, text, {}, budget())
        assert isinstance(result, Interpretation)
        assert result.valid and result.proposal["intent"] == expected
        assert (result.proposal["reference"] or {}).get("index") == index
        assert client.calls == []


def test_conditional_and_mixed_controls_are_not_rule_cancelled():
    for text in ("如果找不到就取消", "取消，然后继续"):
        client = Client(
            [
                LLMResponse(
                    '{"intent":"UNCLEAR","changes":[],"reference":null,'
                    '"evidence":"' + text + '","clarification_reason":"条件不明确"}',
                    [],
                )
            ]
        )
        result = interpret_turn(client, text, {}, budget())
        assert result.valid and result.proposal["intent"] == "UNCLEAR"
        assert len(client.calls) == 1
        assert client.calls[0][1] == []


def test_kilometers_and_meters_normalize_to_the_same_distance():
    for text, value in (
        ("1公里以内", 1.0),
        ("15公里以内", 15.0),
        ("99公里以内", 99.0),
        ("1.5公里以内", 1.5),
        ("15.5公里以内", 15.5),
        ("１５公里以内", 15.0),
        ("15000米以内", 15.0),
    ):
        item = proposal(
            "UPDATE_CONSTRAINTS",
            [change("max_distance_km", value, text)],
            evidence=text,
        )
        assert validate_proposal(item, text) is None


def test_chinese_integer_distance_is_source_checked():
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 15.0, "十五公里以内")],
        evidence="十五公里以内",
    )
    assert validate_proposal(item, "十五公里以内") is None


def test_integer_distance_survives_reduction_into_the_next_model_summary():
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 15, "15公里以内")],
    )
    task = AgentTask(goal="找服务区", status=TaskStatus.RUNNING)
    decision = reduce_dialogue(task, item, turn_id=2)
    summary = {
        "goal": decision.goal,
        "status": task.status.value,
        "revision": decision.revision,
        "constraints": {
            key: {
                "value": value.value,
                "source_turn_id": value.source_turn_id,
                "evidence": value.evidence,
            }
            for key, value in decision.constraints.items()
        },
        "unresolved_constraints": [],
        "candidates": [],
    }
    client = Client(
        [
            LLMResponse(
                '{"intent":"UNCLEAR","changes":[],"reference":null,'
                '"evidence":"","clarification_reason":"不明确"}',
                [],
            )
        ]
    )

    interpret_turn(client, "这条件还在吗", summary, budget())

    sent_summary = json.loads(client.calls[0][0][1]["content"])["summary"]
    assert sent_summary["constraints"]["max_distance_km"]["value"] == 15.0


def test_distance_source_value_must_match_even_when_the_quote_is_present():
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 10.0, "15公里以内")],
        evidence="15公里以内",
    )
    assert validate_proposal(item, "15公里以内") == "SOURCE_VALUE_MISMATCH"


def test_distance_evidence_must_cover_the_complete_number_and_unit():
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 15.0, "以内")],
        evidence="15公里以内",
    )

    assert validate_proposal(item, "15公里以内") == "SOURCE_EVIDENCE_MISMATCH"


def test_ambiguous_distance_is_not_resolved_by_picking_one_value():
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 15.0, "15或10公里")],
        evidence="15或10公里",
    )
    assert validate_proposal(item, "15或10公里") == "AMBIGUOUS_VALUE"


def test_distance_rejects_negative_infinite_and_boolean_values():
    negative = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 15.0, "-15公里以内")],
        evidence="-15公里以内",
    )
    infinite = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", float("inf"), "15公里以内")],
        evidence="15公里以内",
    )
    boolean = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", True, "15公里以内")],
        evidence="15公里以内",
    )
    assert validate_proposal(negative, "-15公里以内") == "INVALID_DISTANCE"
    assert validate_proposal(infinite, "15公里以内") == "INVALID_PROPOSAL"
    assert validate_proposal(boolean, "15公里以内") == "INVALID_PROPOSAL"


def test_distance_parser_rejects_suffix_matches_inside_invalid_numeric_tokens():
    for text, suffix_value in (
        ("一百五十公里以内", 50.0),
        ("1e3公里以内", 3.0),
        ("−15公里以内", 15.0),
    ):
        item = proposal(
            "UPDATE_CONSTRAINTS",
            [change("max_distance_km", suffix_value, text)],
            evidence=text,
        )
        assert validate_proposal(item, text) == "INVALID_DISTANCE"


def test_evidence_must_be_a_contiguous_quote_from_the_current_turn():
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("preferred_area", "西湖", "西湖附近")],
        evidence="附近西湖",
    )
    assert validate_proposal(item, "西湖附近") == "SOURCE_EVIDENCE_MISMATCH"


def test_candidate_intents_cannot_include_constraint_changes():
    for intent in ("SELECT", "ASK_CANDIDATE"):
        item = proposal(
            intent,
            [change("preferred_area", "西湖", "西湖")],
            reference={"index": None, "name": "西湖", "evidence": "西湖"},
        )
        assert validate_proposal(item, "西湖") == "CONFLICTING_INTENTS"


def test_remove_requires_explicit_semantics_and_null_value():
    valid = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", None, "取消距离限制", "REMOVE")],
        evidence="取消距离限制",
    )
    invalid = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", None, "距离限制", "REMOVE")],
        evidence="距离限制",
    )
    assert validate_proposal(valid, "取消距离限制") is None
    assert validate_proposal(invalid, "距离限制") == "SOURCE_OPERATION_MISMATCH"


def test_remove_must_be_affirmative_in_the_full_source_not_just_its_quote():
    cases = [
        ("我不想放弃按摩椅", "放弃按摩椅", "unsupported", "按摩椅"),
        ("不要放弃按摩椅", "放弃按摩椅", "unsupported", "按摩椅"),
        ("并非要取消区域偏好", "取消区域偏好", "preferred_area", None),
        ("如果需要放弃按摩椅", "放弃按摩椅", "unsupported", "按摩椅"),
        ("先别取消距离限制", "取消距离限制", "max_distance_km", None),
        ("放弃按摩椅吗", "放弃按摩椅", "unsupported", "按摩椅"),
        ("按摩椅要不要放弃", "按摩椅要不要放弃", "unsupported", "按摩椅"),
    ]
    for text, evidence, field, value in cases:
        item = proposal(
            "UPDATE_CONSTRAINTS",
            [change(field, value, evidence, "REMOVE")],
        )
        assert validate_proposal(item, text) == "SOURCE_OPERATION_MISMATCH"


@pytest.mark.parametrize("text", ("按摩椅，别放弃", "如果没有合适的，就放弃按摩椅"))
def test_remove_does_not_split_negation_or_condition_scope_at_commas(text):
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("unsupported", "按摩椅", text, "REMOVE")],
        evidence=text,
    )
    assert validate_proposal(item, text) == "SOURCE_OPERATION_MISMATCH"


def test_remove_accepts_only_documented_full_sentence_forms():
    simple = proposal(
        "UPDATE_CONSTRAINTS",
        [change("unsupported", "按摩椅", "放弃按摩椅", "REMOVE")],
        evidence="放弃按摩椅",
    )
    please = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", None, "请取消距离限制", "REMOVE")],
        evidence="请取消距离限制",
    )
    clear_all = proposal(
        "UPDATE_CONSTRAINTS",
        [
            change(
                "unsupported",
                "*",
                "按已支持条件继续，放弃其他要求",
                "REMOVE",
            )
        ],
        evidence="按已支持条件继续，放弃其他要求",
    )
    retain_name_preference = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", None, "取消距离限制", "REMOVE")],
        evidence="取消距离限制，仍保留西湖偏好",
    )
    replace_with_distance = proposal(
        "UPDATE_CONSTRAINTS",
        [
            change("unsupported", "按摩椅", "放弃按摩椅", "REMOVE"),
            change("max_distance_km", 15.0, "15公里以内"),
        ],
        evidence="放弃按摩椅，按15公里以内继续",
    )

    assert validate_proposal(simple, "放弃按摩椅") is None
    assert validate_proposal(please, "请取消距离限制") is None
    assert validate_proposal(clear_all, "按已支持条件继续，放弃其他要求") is None
    assert (
        validate_proposal(retain_name_preference, "取消距离限制，仍保留西湖偏好")
        is None
    )
    assert (
        validate_proposal(replace_with_distance, "放弃按摩椅，按15公里以内继续") is None
    )


@pytest.mark.parametrize("text", ("先取消距离限制", "取消距离限制然后搜索"))
def test_remove_rejects_unconsumed_source_prefix_and_suffix(text):
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", None, text, "REMOVE")],
        evidence=text,
    )

    assert validate_proposal(item, text) == "SOURCE_OPERATION_MISMATCH"


def test_distance_continuation_after_removal_requires_matching_set_change():
    text = "放弃按摩椅，按15公里以内继续"
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("unsupported", "按摩椅", "放弃按摩椅", "REMOVE")],
        evidence=text,
    )

    assert validate_proposal(item, text) == "SOURCE_OPERATION_MISMATCH"


@pytest.mark.parametrize(
    "field,value,verb",
    [
        ("preferred_area", None, "取消区域偏好"),
        ("unsupported", "按摩椅", "取消按摩椅"),
        ("preferred_area", None, "移除区域偏好"),
    ],
)
def test_valid_removal_and_distance_continuation_is_not_mixed_control(
    field, value, verb
):
    text = f"{verb}，按15公里以内继续"
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [
            change(field, value, verb, "REMOVE"),
            change("max_distance_km", 15.0, "15公里以内"),
        ],
        evidence=text,
    )

    assert validate_proposal(item, text) is None


def test_cancel_and_resume_controls_remain_ambiguous():
    text = "取消，然后继续"
    item = proposal("CANCEL", evidence=text)

    assert validate_proposal(item, text) == "AMBIGUOUS_CONTROL"


def test_removal_distance_continuation_rejects_unparsed_control_prose():
    text = "取消区域偏好，按原计划先取消任务，再按15公里以内继续"
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [
            change("preferred_area", None, "取消区域偏好", "REMOVE"),
            change("max_distance_km", 15.0, "15公里以内"),
        ],
        evidence=text,
    )

    assert validate_proposal(item, text) == "SOURCE_OPERATION_MISMATCH"


@pytest.mark.parametrize(
    "text,suffix_value",
    [("十五点五公里以内", 5.0), ("－15公里以内", 15.0)],
)
def test_distance_parser_rejects_chinese_decimal_and_full_width_negative_suffixes(
    text, suffix_value
):
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", suffix_value, text)],
        evidence=text,
    )
    assert validate_proposal(item, text) == "INVALID_DISTANCE"


@pytest.mark.parametrize(
    "text,suffix_value",
    [
        ("一百五十公里以内", 50.0),
        ("1 5公里以内", 5.0),
        ("−15公里以内", 15.0),
    ],
)
def test_distance_parser_rejects_whole_unsupported_numeric_tokens(text, suffix_value):
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", suffix_value, text)],
        evidence=text,
    )
    assert validate_proposal(item, text) == "INVALID_DISTANCE"


def test_distance_parser_rejects_multiple_values_within_one_unit_phrase():
    text = "15或10公里以内"
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 10.0, text)],
        evidence=text,
    )

    assert validate_proposal(item, text) == "AMBIGUOUS_VALUE"


def test_extra_model_tool_and_authorization_fields_are_rejected():
    for key in ("model_id", "tool", "authorization"):
        item = proposal("UNCLEAR")
        item[key] = "untrusted"
        assert validate_proposal(item, "") == "INVALID_PROPOSAL"


def test_ambiguous_replacement_cannot_be_misread_as_a_named_selection():
    item = proposal(
        "SELECT",
        reference={"index": None, "name": "一个", "evidence": "一个"},
    )
    assert validate_proposal(item, "换一个") == "AMBIGUOUS_CONTROL"


def test_conditional_cancel_and_bare_ack_are_not_actionable_proposals():
    conditional = proposal("CANCEL")
    bare_ack = proposal(
        "SELECT",
        reference={"index": None, "name": "可以", "evidence": "可以"},
    )
    assert validate_proposal(conditional, "如果找不到就取消") == "AMBIGUOUS_CONTROL"
    assert validate_proposal(bare_ack, "可以") == "AMBIGUOUS_CONTROL"


def test_nested_schema_rejects_missing_keys_extra_keys_and_non_exact_types():
    missing = proposal("UPDATE_CONSTRAINTS")
    del missing["reference"]
    extra_nested = proposal(
        "UPDATE_CONSTRAINTS",
        [{**change("max_distance_km", 15.0, "15公里"), "tool": "search"}],
    )
    invalid_enum = proposal("UPDATE_CONSTRAINTS", [change("poi_id", "poi-1", "poi-1")])
    bad_index = proposal(
        "SELECT",
        reference={"index": True, "name": None, "evidence": "第1个"},
    )
    too_many = proposal(
        "UPDATE_CONSTRAINTS",
        [change("preferred_area", "西湖", "西湖") for _ in range(9)],
    )
    for item in (missing, extra_nested, invalid_enum, bad_index, too_many):
        assert validate_proposal(item, "15公里 西湖 第1个") == "INVALID_PROPOSAL"


def test_evidence_length_limits_are_enforced_by_the_recursive_schema():
    item = proposal(
        "UPDATE_CONSTRAINTS",
        [change("preferred_area", "西湖", "西湖" * 121)],
    )
    assert validate_proposal(item, "西湖" * 121) == "INVALID_PROPOSAL"


def test_invalid_json_is_not_repaired_or_retried():
    client = Client(
        [
            LLMResponse(
                "not json", [], response_model="test-model", usage={"input_tokens": 2}
            )
        ]
    )
    call_budget = budget()
    result = interpret_turn(client, "找近处服务区", {}, call_budget)

    assert not result.valid and result.reason == "INVALID_PROPOSAL"
    assert len(client.calls) == call_budget.model_calls == 1
    assert result.response_model == "test-model"
    assert result.usage == {"input_tokens": 2}


def test_timeout_is_not_repaired_or_retried():
    client = Client([TimeoutError("provider timeout")])
    call_budget = budget()
    result = interpret_turn(client, "找近处服务区", {}, call_budget)

    assert not result.valid and result.reason == "INTERPRETER_TIMEOUT"
    assert len(client.calls) == call_budget.model_calls == 1


def test_interpreter_does_not_forward_ids_history_or_authority_from_summary():
    client = Client(
        [
            LLMResponse(
                '{"intent":"UNCLEAR","changes":[],"reference":null,'
                '"evidence":"","clarification_reason":"不明确"}',
                [],
            )
        ]
    )
    summary = {
        "goal": "找服务区",
        "status": "RUNNING",
        "revision": 1,
        "constraints": {
            "max_distance_km": {
                "value": 15.0,
                "source_turn_id": 1,
                "evidence": "十五公里以内",
            }
        },
        "unresolved_constraints": [],
        "candidates": [{"index": 1, "name": "西湖服务区", "aliases": ["西湖服务区"]}],
        "task_id": "private-task-id",
        "trace": ["private history"],
        "pending_action_id": "private-action-id",
        "authorization": "not for model",
    }
    result = interpret_turn(client, "嗯", summary, budget())
    prompt = repr(client.calls[0][0])
    request = client.calls[0][0]
    safe_summary = json.loads(request[1]["content"])["summary"]

    assert result.valid
    assert all(
        secret not in prompt
        for secret in (
            "private-task-id",
            "private history",
            "private-action-id",
            "not for model",
        )
    )
    assert set(safe_summary) == {
        "goal",
        "status",
        "revision",
        "constraints",
        "unresolved_constraints",
        "candidates",
    }
    assert safe_summary["constraints"]["max_distance_km"]["value"] == 15.0
    assert safe_summary["candidates"][0]["aliases"] == ["西湖服务区"]
    assert "additionalProperties" in request[0]["content"]
    assert "max_distance_km" in request[0]["content"]
    assert "unsupported SET value" in request[0]["content"]
    assert "meters to km" in request[0]["content"]
    assert "Bare 好/可以" in request[0]["content"]


def test_provider_tool_calls_are_rejected_and_never_run():
    tool_call = LLMToolCall("call-id", "search_nearby_rest_area", {}, "{}")
    client = Client([LLMResponse(None, [tool_call], response_model="test-model")])
    result = interpret_turn(client, "找附近服务区", {}, budget())

    assert not result.valid and result.reason == "INTERPRETER_TOOL_CALL"
    assert result.response_model == "test-model"
    assert len(client.calls) == 1 and client.calls[0][1] == []


def test_model_budget_error_is_preserved_without_starting_a_request():
    client = Client([])
    call_budget = budget()
    for _ in range(3):
        call_budget.claim_model()

    result = interpret_turn(client, "找服务区", {}, call_budget)

    assert not result.valid and result.reason == "MODEL_BUDGET"
    assert client.calls == [] and call_budget.model_calls == 3


def test_deadline_is_checked_after_the_single_response_and_metadata_is_kept():
    now = [0.0]

    class LateClient(Client):
        def chat_with_timeout(self, messages, tools=None, *, timeout_seconds):
            response = super().chat_with_timeout(
                messages, tools, timeout_seconds=timeout_seconds
            )
            now[0] = 2.0
            return response

    client = LateClient(
        [LLMResponse("{}", [], response_model="test-model", usage={"input_tokens": 4})]
    )
    call_budget = TurnBudget(1.0, 2, lambda: now[0], max_model_calls=1)

    result = interpret_turn(client, "找服务区", {}, call_budget)

    assert not result.valid and result.reason == "TIME_BUDGET"
    assert result.response_model == "test-model"
    assert result.usage == {"input_tokens": 4}
    assert len(client.calls) == call_budget.model_calls == 1
