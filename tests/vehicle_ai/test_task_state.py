import pytest

from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall
from modules.vehicle_ai.tools import build_default_tool_registry


class Replies(BaseLLMClient):
    def __init__(self, *responses):
        self.responses = iter(responses)

    def chat(self, messages, tools=None):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def agent_for(*responses, **kwargs):
    manager = ContextManager()
    return VehicleAgent(
        Replies(*responses), manager, build_default_tool_registry(manager), **kwargs
    )


def search():
    return LLMResponse(None, [LLMToolCall("s1", "search_nearby_rest_area", {}, "{}")])


def test_pending_confirmation_then_completion_is_one_task():
    agent = agent_for(search(), LLMResponse("是否导航？", []))
    agent.chat("找个服务区", debug=False)
    assert agent.task.status == "AWAITING_CONFIRMATION"
    task_id = agent.task.task_id
    pending = agent.pending_actions.get()
    assert agent.confirm_pending(pending.action_id).success
    assert agent.task.task_id == task_id
    assert agent.task.status == "COMPLETED"
    assert agent.task.last_tool_result["success"] is True
    assert len(agent.task.transitions) >= 3
    agent.confirm_pending(pending.action_id)
    assert agent.task.status == "COMPLETED"


def test_cancellation_is_recorded_and_does_not_execute_navigation():
    agent = agent_for(search(), LLMResponse("是否导航？", []))
    agent.chat("找个服务区", debug=False)
    agent.chat("取消", debug=False)
    assert agent.task.status == "CANCELLED"
    assert agent.pending_actions.get() is None
    assert not any(
        r.name == "start_navigation" for r in agent.tool_registry.execution_history()
    )


@pytest.mark.parametrize(
    "response,reason",
    [
        (LLMResponse(" ", []), "EMPTY_RESPONSE"),
        (TimeoutError("secret detail"), "MODEL_TIMEOUT"),
        (RuntimeError("secret detail"), "MODEL_ERROR"),
    ],
)
def test_model_failure_is_explicit_and_does_not_leak_exception(response, reason):
    agent = agent_for(response)
    answer = agent.chat("你好", debug=False)
    assert answer.strip()
    assert "secret detail" not in answer
    assert agent.task.status == "FAILED"
    assert agent.task.reason == reason


def test_round_limit_and_reset():
    agent = agent_for(max_tool_rounds=0)
    agent.chat("你好", debug=False)
    assert agent.task.reason == "ROUND_LIMIT"
    agent.reset()
    assert agent.task.status == "IDLE"


def test_tool_failure_cannot_be_overridden_by_model_success_text():
    bad = LLMResponse(None, [LLMToolCall("bad", "missing_tool", {}, "{}")])
    agent = agent_for(bad, LLMResponse("已完成", []))
    agent.chat("执行操作", debug=False)
    assert agent.task.status == "FAILED"
    assert agent.task.last_tool_result["success"] is False
    snapshot = agent.task.to_dict()
    snapshot["transitions"].clear()
    assert agent.task.transitions


def test_reject_api_and_wrong_id_preserve_task_semantics():
    agent = agent_for(search(), LLMResponse("是否导航？", []))
    agent.chat("找服务区", debug=False)
    agent.reject_pending("wrong-id")
    assert agent.task.status == "AWAITING_CONFIRMATION"
    agent.reject_pending(agent.pending_actions.get().action_id)
    assert agent.task.status == "CANCELLED"


def test_unresolved_target_change_is_not_completed():
    agent = agent_for(
        search(), LLMResponse("是否导航？", []), LLMResponse("没有找到", [])
    )
    agent.chat("找服务区", debug=False)
    agent.chat("换成东湖服务区", debug=False)
    assert agent.task.goal == "换成东湖服务区"
    assert agent.task.status == "FAILED"
    assert agent.task.reason == "TARGET_NOT_FOUND"
