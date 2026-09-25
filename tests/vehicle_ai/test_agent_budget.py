import json

import pytest

from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent
from modules.vehicle_ai.agent.budget import BudgetExceeded, TurnBudget
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall
from modules.vehicle_ai.tools import build_default_tool_registry


class Client(BaseLLMClient):
    def __init__(self, responses, tick=None):
        self.responses = iter(responses)
        self.tick = tick
        self.requests = []

    def chat(self, messages, tools=None):
        self.requests.append(messages.copy())
        if self.tick:
            self.tick()
        return next(self.responses)


def call(name, arguments=None, raw=None):
    arguments = arguments or {}
    return LLMToolCall(
        "c", name, arguments, json.dumps(arguments) if raw is None else raw
    )


def make_agent(responses, **kwargs):
    manager = ContextManager()
    client = Client(responses)
    return VehicleAgent(client, manager, build_default_tool_registry(manager), **kwargs)


def test_repeated_write_is_not_executed_twice():
    response = LLMResponse(None, [call("play_music", {"query": "轻音乐"})])
    agent = make_agent([response, response, LLMResponse("好了", [])])
    agent.chat("放音乐", debug=False)
    assert agent.task.reason == "REPEATED_CALL"
    assert len(agent.tool_registry.execution_history()) == 1


@pytest.mark.parametrize(
    "raw,args",
    [
        ("{bad", {}),
        ("[]", []),
        ('{"volume": true}', {"volume": True}),
        ('{"volume": 25}', {"volume": 30}),
    ],
)
def test_bad_arguments_never_execute(raw, args):
    agent = make_agent([LLMResponse(None, [LLMToolCall("c", "set_volume", args, raw)])])
    agent.chat("调音量", debug=False)
    assert agent.task.reason == "INVALID_ARGUMENTS"
    assert agent.context_manager.get_context().vehicle.volume != True  # noqa: E712


def test_late_model_response_cannot_start_a_write():
    clock = [0.0]
    agent = make_agent(
        [LLMResponse(None, [call("play_music", {"query": "音乐"})])],
        turn_timeout_seconds=1.0,
        budget_clock=lambda: clock[0],
    )
    agent.llm.tick = lambda: clock.__setitem__(0, 2.0)
    agent.chat("放音乐", debug=False)
    assert agent.task.reason == "TIME_BUDGET"
    assert not agent.tool_registry.execution_history()


def test_batch_tool_calls_respect_total_limit():
    response = LLMResponse(
        None, [call("set_volume", {"volume": n}) for n in [20, 30, 40]]
    )
    agent = make_agent([response], max_tool_calls=2)
    agent.chat("调音量", debug=False)
    assert agent.task.reason == "TOOL_BUDGET"
    assert len(agent.tool_registry.execution_history()) == 2


def test_uncertain_write_queries_state_and_never_replays():
    agent = make_agent([LLMResponse(None, [call("play_music", {"query": "音乐"})])])
    writes = []

    def uncertain(query):
        writes.append(query)
        agent.context_manager.update_vehicle(media_playing=True, media_title=query)
        raise TimeoutError("remote result unknown")

    agent.tool_registry.get("play_music").handler = uncertain
    agent.chat("放音乐", debug=False)
    assert writes == ["音乐"]
    assert [r.name for r in agent.tool_registry.execution_history()] == [
        "play_music",
        "get_media_status",
    ]
    assert agent.task.reconciliation["result"]["data"]["media_playing"] is True
    assert agent.task.status == "AWAITING_INPUT"
    assert agent.task.reason == "WRITE_OUTCOME_UNKNOWN"


def test_uncertain_write_gets_reserved_readback_at_tool_budget_limit():
    agent = make_agent(
        [LLMResponse(None, [call("play_music", {"query": "音乐"})])],
        max_tool_calls=1,
    )
    agent.tool_registry.get("play_music").handler = lambda **_kwargs: (
        _ for _ in ()
    ).throw(TimeoutError("uncertain"))
    agent.chat("放音乐", debug=False)
    assert [record.name for record in agent.tool_registry.execution_history()] == [
        "play_music",
        "get_media_status",
    ]


def test_turn_budget_allows_exactly_one_reserved_read_retry():
    budget = TurnBudget(10.0, 3, lambda: 0.0)
    budget.claim("search_nearby_rest_area", {})
    budget.claim_retry("search_nearby_rest_area", {})

    assert budget.calls == 2
    with pytest.raises(BudgetExceeded, match="REPEATED_CALL"):
        budget.claim_retry("search_nearby_rest_area", {})


def test_turn_budget_retry_counts_against_total_tool_limit():
    budget = TurnBudget(10.0, 1, lambda: 0.0)
    budget.claim("search_nearby_rest_area", {})

    with pytest.raises(BudgetExceeded, match="TOOL_BUDGET"):
        budget.claim_retry("search_nearby_rest_area", {})
