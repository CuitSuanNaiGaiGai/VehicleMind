"""Integration tests for the opt-in rest-location business dialogue."""

from modules.vehicle_ai.agent.dialogue_candidates import (
    candidate_is_current,
    resolve_reference,
)
from tests.vehicle_ai.dialogue_support import interpretation, make_runtime


def test_select_second_candidate_stages_one_explicit_confirmation():
    runtime, _ = make_runtime(interpretation("START"))

    runtime.chat("找附近服务区", debug=False)
    task = runtime.agent.task
    assert task.status.value == "AWAITING_INPUT"
    expected = task.plan.candidates[1]["poi_id"]

    runtime.chat("就去第二个", debug=False)
    pending = runtime.agent.pending_actions.get()
    assert pending.arguments["poi_id"] == expected
    assert (
        pending.created_at + pending.expires_after_seconds
        <= task.plan.candidate_snapshot.expires_at
    )
    assert (
        runtime.context_manager.get_context().vehicle.navigation_destination_id is None
    )

    assert runtime.agent.confirm_pending(pending.action_id).success
    assert not runtime.agent.confirm_pending(pending.action_id).success
    writes = [
        item
        for item in runtime.tools.execution_history()
        if item.name == "start_navigation" and item.confirmed and item.success
    ]
    assert len(writes) == 1
    assert runtime.agent.task.task_id == task.task_id


def test_only_a_currently_displayed_candidate_set_resolves_ordinals():
    runtime, now = make_runtime(interpretation("START"))
    runtime.chat("找附近服务区", debug=False)

    plan = runtime.agent.task.plan
    snapshot = plan.candidate_snapshot
    assert candidate_is_current(runtime.agent)
    assert (
        resolve_reference(runtime.agent, {"index": 1})["poi_id"]
        == (snapshot.presented_poi_ids[0])
    )

    now[0] = snapshot.expires_at
    assert not candidate_is_current(runtime.agent)
    assert resolve_reference(runtime.agent, {"index": 1}) is None


def test_reset_replaces_dialogue_session_and_turn_counter():
    runtime, _ = make_runtime(interpretation("START"))
    runtime.chat("找附近服务区", debug=False)
    session_id = runtime.agent.dialogue_session.session_id
    assert runtime.agent.dialogue_session.turn_id == 1

    runtime.agent.reset()

    assert runtime.agent.dialogue_session.session_id != session_id
    assert runtime.agent.dialogue_session.turn_id == 0
    assert runtime.agent.task.plan is None
