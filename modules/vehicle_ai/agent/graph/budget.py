"""将既有单轮预算转换为可 checkpoint 的数据。"""

import time

from modules.vehicle_ai.agent.budget import BudgetExceeded, TurnBudget


def create(agent) -> tuple[dict, float]:
    budget = TurnBudget(
        agent.turn_timeout_seconds, agent.max_tool_calls, agent.budget_clock
    )
    return dump(budget), time.time() + agent.turn_timeout_seconds


def load(agent, state: dict) -> TurnBudget:
    raw = state["budget"]
    budget = TurnBudget(
        agent.turn_timeout_seconds, agent.max_tool_calls, agent.budget_clock
    )
    budget.deadline = raw["deadline"]
    budget.calls = raw["calls"]
    budget.signatures = set(raw["signatures"])
    budget.used_calls = set(raw["used_calls"])
    budget.retry_signatures = set(raw["retry_signatures"])
    # 进程恢复由工作流入口重新换算 monotonic deadline；墙钟额外限制总体期限。
    if time.time() >= state["deadline_at"]:
        raise BudgetExceeded("TIME_BUDGET")
    return budget


def dump(budget: TurnBudget) -> dict:
    return {
        "deadline": budget.deadline,
        "calls": budget.calls,
        "signatures": sorted(budget.signatures),
        "used_calls": sorted(budget.used_calls),
        "retry_signatures": sorted(budget.retry_signatures),
    }
