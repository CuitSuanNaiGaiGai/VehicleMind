"""Source-labelled conversation evidence and task snapshots for replay/reporting."""

import json
from copy import deepcopy


POINTERS = {"就去这个", "导航过去", "可以", "好", "好的", "yes", "okay", "sure"}


def is_pointer(text: str) -> bool:
    return text.strip().rstrip("。.!！?？").casefold() in POINTERS


def record(agent, kind: str, **data) -> None:
    pending = agent.pending_actions.get()
    agent.task.pending_action = pending.to_dict() if pending else None
    agent.trace.append(
        deepcopy(
            {
                "sequence": agent.trace_sequence,
                "at_seconds": agent.pending_actions.now(),
                "clock": "runtime_action_clock",
                "kind": kind,
                "task": agent.task.to_dict(),
                **data,
            }
        )
    )
    agent.trace_sequence += 1
    del agent.trace[: -agent.max_task_trace_events]
    if agent.historical_event_sink is not None:
        agent.historical_event_sink(agent.trace[-1])


def evidence_message(agent) -> dict:
    # Explicitly historical: neither model text nor old tool output proves current state.
    evidence = [
        {k: v for k, v in event.items() if k != "task"}
        for event in agent.trace[-18:]
        if event["kind"]
        in {"user_request", "agent_reply", "tool_result", "confirmation"}
    ]
    return {
        "role": "system",
        "content": "TASK AND SOURCED HISTORY (historical, not current sensor facts):\n"
        + json.dumps(
            {"task": agent.task.to_dict(), "evidence": evidence},
            ensure_ascii=False,
            default=str,
        ),
    }


def trip_memory_message(tool_names: list[str]) -> dict | None:
    if "query_trip_events" not in tool_names:
        return None
    return {
        "role": "system",
        "content": (
            "行程事件记忆是带时间的历史事实。只有用户询问本次行程的提醒次数、"
            "选择、取消或已发生动作时才调用 query_trip_events；"
            "不得把历史风险当作当前驾驶状态。"
        ),
    }


def record_final_response(agent, user_text: str, answer: str) -> str:
    if agent._turn_music_warning:
        music_status = (
            "本轮音乐播放成功，随后已暂停。"
            if agent._turn_music_paused
            else "本轮音乐播放成功。"
        )
        if agent.task.status.value == "AWAITING_CONFIRMATION":
            outcome = "待确认操作尚未执行，待确认后才会执行。"
        elif agent.task.status.value == "COMPLETED":
            outcome = ""
        else:
            reason = agent.task.reason or agent.task.status.value
            outcome = f"本轮任务未完成（{reason}）。"
        answer = f"{music_status}{outcome}音乐不能消除疲劳。{agent._turn_music_warning}"
    agent.history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
    )
    agent.history = agent.history[-12:]
    record(agent, "agent_reply", source="agent", quality="GENERATED", text=answer)
    return answer
