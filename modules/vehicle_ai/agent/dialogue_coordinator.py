"""协调休息地点业务对话中的解释、状态归约与候选操作。"""

from dataclasses import asdict
from typing import TYPE_CHECKING

from modules.config.agent_dialogue import AgentDialogueConfig
from modules.vehicle_ai.agent.budget import TurnBudget
from modules.vehicle_ai.agent.dialogue_candidates import (
    candidate_is_current,
    invalidate_candidates,
    present_candidates,
    resolve_reference,
    search_candidates,
    select_candidate,
)
from modules.vehicle_ai.agent.dialogue_interpreter import interpret_turn
from modules.vehicle_ai.agent.dialogue_reducer import reduce_dialogue
from modules.vehicle_ai.agent.plan import PlanStatus
from modules.vehicle_ai.agent.session import record
from modules.vehicle_ai.agent.task_state import AgentTask, TaskStatus

if TYPE_CHECKING:
    from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent


class DialogueCoordinator:
    def __init__(self, agent: "VehicleAgent", config: AgentDialogueConfig) -> None:
        self.agent = agent
        self.config = config

    def handles(self, user_text: str) -> bool:
        if not self.config.enabled or not user_text.strip():
            return False
        plan = self.agent.task.plan
        return (
            self.agent.plan_flow.is_rest_location_request(user_text)
            or (plan is not None and plan.status is PlanStatus.RUNNING)
            or (
                self.agent.task.status
                in {
                    TaskStatus.RUNNING,
                    TaskStatus.AWAITING_INPUT,
                    TaskStatus.AWAITING_CONFIRMATION,
                }
                and self.agent.plan_flow.is_rest_location_request(self.agent.task.goal)
            )
        )

    def _summary(self) -> dict:
        agent, task = self.agent, self.agent.task
        candidates = []
        if candidate_is_current(agent):
            snapshot = task.plan.candidate_snapshot
            for index, poi_id in enumerate(snapshot.presented_poi_ids, 1):
                candidate = next(
                    item for item in task.plan.candidates if item["poi_id"] == poi_id
                )
                aliases = candidate.get("aliases", [])
                candidates.append(
                    {
                        "index": index,
                        "name": agent.plan_flow.display_name(candidate),
                        "aliases": [item for item in aliases if isinstance(item, str)],
                    }
                )
        return {
            "goal": task.goal,
            "status": task.status.value,
            "revision": task.revision,
            "constraints": {
                key: asdict(value) for key, value in task.constraints.items()
            },
            "unresolved_constraints": [
                asdict(value) for value in task.unresolved_constraints
            ],
            "candidates": candidates,
        }

    def chat(self, user_text: str, debug: bool = True) -> str:
        agent = self.agent
        user_text = user_text.strip()
        if not user_text:
            return ""
        self.begin_turn(user_text)
        budget = TurnBudget(
            agent.turn_timeout_seconds,
            agent.max_tool_calls,
            agent.budget_clock,
            max_model_calls=1,
        )
        interpretation = self.interpret(user_text, budget)
        decision = self.reduce(interpretation["proposal"], interpretation["valid"])
        answer = self.execute_decision(decision, budget)
        return agent._record_final_response(user_text, answer)

    def begin_turn(self, user_text: str) -> None:
        agent = self.agent
        agent.current_user_intent = user_text
        agent._turn_music_warning = ""
        agent._turn_music_paused = False
        agent.dialogue_session.next_turn()
        record(
            agent,
            "user_request",
            source="user",
            quality="SELF_REPORTED",
            text=user_text,
        )

    def interpret(self, user_text: str, budget: TurnBudget) -> dict:
        agent = self.agent
        interpretation = interpret_turn(
            agent.llm,
            user_text,
            self._summary(),
            budget,
            timeout_seconds=self.config.interpreter_timeout_seconds,
        )
        record(
            agent,
            "dialogue_proposal",
            quality="VALIDATED" if interpretation.valid else "REJECTED",
            **asdict(interpretation),
        )
        return asdict(interpretation)

    def reduce(self, proposal: dict, valid: bool) -> dict:
        agent = self.agent
        if not valid:
            agent.task.clarification = (
                "这次请求未能可靠理解，请明确说明查找条件或候选选择。"
            )
            return {
                "operation": "clarify",
                "reference": None,
                "answer": agent.task.clarification,
            }
        decision = reduce_dialogue(
            agent.task,
            proposal,
            turn_id=agent.dialogue_session.turn_id,
            max_revisions=self.config.max_revisions,
        )
        if decision.operation == "start":
            agent.plan_flow.cancel(agent, "NEW_TASK")
            agent.pending_actions.clear()
            agent.task = AgentTask(goal=decision.goal)
        task = agent.task
        if decision.changed:
            invalidate_candidates(agent, "CONSTRAINT_CHANGED")
        task.goal = decision.goal
        task.constraints = dict(decision.constraints)
        task.unresolved_constraints = list(decision.unresolved_constraints)
        task.revision = decision.revision
        task.clarification = decision.clarification
        record(
            agent,
            "dialogue_decision",
            source="dialogue_reducer",
            quality="VALIDATED",
            decision=asdict(decision),
        )
        return {
            "operation": decision.operation,
            "reference": decision.reference,
            "answer": None,
        }

    def execute_decision(self, decision: dict, budget: TurnBudget) -> str:
        if decision["answer"] is not None:
            return decision["answer"]
        return self._execute(decision["operation"], decision["reference"], budget)

    def _execute(
        self, operation: str, reference: dict | None, budget: TurnBudget
    ) -> str:
        agent = self.agent
        if operation == "clarify":
            agent.task.transition(TaskStatus.AWAITING_INPUT, "DIALOGUE_CLARIFICATION")
            return agent.task.clarification
        if operation == "side_question":
            return "当前休息地点任务已保留。请说明要查询哪个候选，或继续调整查找条件。"
        if operation == "cancel":
            invalidate_candidates(agent, "USER_CANCELLED")
            agent.pending_actions.clear()
            if not agent.plan_flow.cancel(agent, "USER_CANCELLED"):
                agent.task.transition(TaskStatus.CANCELLED, "USER_CANCELLED")
            return "已取消本次休息地点任务。"
        if operation in {"select", "candidate_answer"}:
            candidate = resolve_reference(agent, reference)
            if candidate is None:
                return "无法确定当前有效候选，请重新查询或明确候选名称与序号。"
            if operation == "select":
                return select_candidate(agent, candidate)
            name = agent.plan_flow.display_name(candidate)
            distance = candidate.get("distance_km")
            eta = candidate.get("eta_minutes")
            details = [name]
            details.append(
                f"距离 {distance} 公里"
                if isinstance(distance, int | float)
                else "距离未提供"
            )
            details.append(
                f"预计 {eta} 分钟" if isinstance(eta, int | float) else "预计时间未提供"
            )
            return "，".join(details) + "。"
        if operation == "resume" and candidate_is_current(agent):
            return present_candidates(agent)
        agent.task.transition(TaskStatus.RUNNING, "DIALOGUE_SEARCH")
        agent.plan_flow.begin(agent, agent.task.goal)
        answer = search_candidates(agent, budget)
        return answer or present_candidates(agent)
