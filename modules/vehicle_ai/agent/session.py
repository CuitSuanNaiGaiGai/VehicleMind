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
