"""Deterministic acceptance runs for bounded Agent planning and recovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from modules.vehicle_ai.agent.plan import PlanStatus
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import NavigationConfig
from modules.vehicle_ai.tools.base import ToolResult


SCENARIO_PATH = Path(__file__).parents[3] / "scenarios/agent_eval/a5_recovery.yaml"
EXPECTED_IDS = (
    "A5_SUCCESS",
    "A5_ALTERNATE_CONFIRM",
    "A5_NO_RESULT",
    "A5_CANCEL",
    "A5_UNKNOWN_WRITE",
    "A5_INVALID_TARGET",
    "A5_READ_RETRY",
)
CATALOG = (
    {
        "poi_id": "rest_area_001",
        "name": "West Lake Rest Area",
        "aliases": ["西湖服务区"],
        "distance_km": 6.8,
        "eta_minutes": 8,
    },
    {
        "poi_id": "rest_area_002",
        "name": "Riverside Service Area",
        "aliases": ["河滨服务区"],
        "distance_km": 12.4,
        "eta_minutes": 15,
    },
)


@dataclass
class RecoveryCaseResult:
    scenario_id: str
    title: str
    passed: bool
    plan_status: str
    terminal_reason: str | None
    step_count: int
    max_steps: int
    recovery_count: int
    max_recoveries: int
    confirmed_navigation_writes: int
    unconfirmed_sensitive_write_violations: int
    search_attempts: int
    pending_action: bool
    duplicate_write_violations: int
    safe_stop_expected: bool
    recovery_expected: bool
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Metric:
    numerator: int
    denominator: int
    target: str

    @property
    def ratio(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "target": self.target,
            "ratio": self.ratio,
        }


@dataclass(frozen=True)
class RecoveryReport:
    suite_id: str
    evaluation_mode: str
    cases: tuple[RecoveryCaseResult, ...]
    metrics: dict[str, Metric]

    @property
    def passed(self) -> bool:
        return bool(self.cases) and all(case.passed for case in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "evaluation_mode": self.evaluation_mode,
            "passed": self.passed,
            "scenario_count": len(self.cases),
            "metrics": {
                name: metric.to_dict() for name, metric in self.metrics.items()
            },
            "cases": [case.to_dict() for case in self.cases],
            "limitations": [
                "固定脚本模型与模拟 POI 的确定性验收，不代表在线模型语义成功率。",
                "模拟写入不连接真实车机、导航服务或道路可用性数据。",
            ],
        }


def load_recovery_scenarios(path: Path | str | None = None) -> dict[str, Any]:
    scenario_path = Path(path) if path is not None else SCENARIO_PATH
    raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("A5 recovery scenario file has an unsupported schema")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not all(isinstance(item, dict) for item in cases):
        raise ValueError("A5 recovery scenarios must contain a list of case objects")
    ids = tuple(item.get("id") for item in cases)
    if ids != EXPECTED_IDS:
        raise ValueError("A5 recovery scenario IDs/order do not match the frozen suite")
    expected_fields = {
        "plan_status",
        "confirmed_navigation_writes",
        "search_attempts",
        "recovery_count",
        "pending_action",
        "safe_stop",
        "recovery",
        "reconciliation_tool",
        "reconciliation_navigation_state",
        "reconciliation_success",
        "reconciliation_deferred",
    }
    for item in cases:
        if set(item) != {"id", "title", "expected"}:
            raise ValueError("A5 recovery case fields do not match schema")
        expected = item["expected"]
        if not isinstance(expected, dict) or set(expected) != expected_fields:
            raise ValueError(f"invalid expected results for {item['id']}")
        if expected["plan_status"] not in {status.value for status in PlanStatus}:
            raise ValueError(f"invalid expected plan status for {item['id']}")
        for field in (
            "confirmed_navigation_writes",
            "search_attempts",
            "recovery_count",
        ):
            if type(expected[field]) is not int or expected[field] < 0:
                raise ValueError(f"expected.{field} must be a non-negative integer")
        for field in ("pending_action", "safe_stop", "recovery"):
            if type(expected[field]) is not bool:
                raise ValueError(f"expected.{field} must be boolean")
        reconciliation_tool = expected["reconciliation_tool"]
        reconciliation_fields = (
            expected["reconciliation_navigation_state"],
            expected["reconciliation_success"],
            expected["reconciliation_deferred"],
        )
        if reconciliation_tool is None:
            if any(value is not None for value in reconciliation_fields):
                raise ValueError("reconciliation expectations require a tool")
        elif not isinstance(reconciliation_tool, str):
            raise ValueError("expected.reconciliation_tool must be text or null")
        elif (
            not isinstance(reconciliation_fields[0], str)
            or type(reconciliation_fields[1]) is not bool
            or type(reconciliation_fields[2]) is not bool
        ):
            raise ValueError(
                "reconciliation state/success/deferred expectations are invalid"
            )
    return raw


def _call(name: str, arguments: dict[str, Any]) -> ScriptedResponse:
    return ScriptedResponse(
        content=None,
        tool_calls=(
            LLMToolCall(
                f"a5-{name}",
                name,
                arguments,
                json.dumps(arguments, ensure_ascii=False),
            ),
        ),
    )


def _new_runtime(
    responses: tuple[ScriptedResponse, ...],
    *,
    unavailable: frozenset[str] = frozenset(),
    transient_search_failures: int = 0,
) -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient(responses),
        navigation_config=NavigationConfig(
            poi_catalog=CATALOG,
            unavailable_poi_ids=unavailable,
            transient_search_failures=transient_search_failures,
        ),
    )


def _run_case(scenario: dict[str, Any]) -> RecoveryCaseResult:
    scenario_id = scenario["id"]
    title = scenario["title"]
    expected = scenario["expected"]
    search_args: dict[str, Any] = {}
    unavailable: frozenset[str] = frozenset()
    transient = 0
    responses: tuple[ScriptedResponse, ...]

    if scenario_id == "A5_SUCCESS":
        responses = (
            _call("search_nearby_rest_area", search_args),
            _call("start_navigation", {"poi_id": "rest_area_001"}),
        )
    elif scenario_id == "A5_ALTERNATE_CONFIRM":
        unavailable = frozenset({"rest_area_001"})
        responses = (
            _call("search_nearby_rest_area", search_args),
            _call("start_navigation", {"poi_id": "rest_area_001"}),
        )
    elif scenario_id == "A5_NO_RESULT":
        search_args = {"max_distance_km": 1.0}
        responses = (
            _call("search_nearby_rest_area", search_args),
            ScriptedResponse(content="没有符合条件的模拟地点。"),
        )
    elif scenario_id == "A5_CANCEL":
        responses = (
            _call("search_nearby_rest_area", search_args),
            _call("start_navigation", {"poi_id": "rest_area_001"}),
        )
    elif scenario_id == "A5_UNKNOWN_WRITE":
        responses = (
            _call("search_nearby_rest_area", search_args),
            _call("start_navigation", {"poi_id": "rest_area_001"}),
        )
    elif scenario_id == "A5_INVALID_TARGET":
        responses = (
            _call("search_nearby_rest_area", search_args),
            _call("start_navigation", {"poi_id": "invented_poi"}),
        )
    elif scenario_id == "A5_READ_RETRY":
        transient = 1
        responses = (
            _call("search_nearby_rest_area", search_args),
            _call("start_navigation", {"poi_id": "rest_area_002"}),
        )
    else:
        raise ValueError(f"unsupported A5 recovery scenario: {scenario_id}")

    runtime = _new_runtime(
        responses,
        unavailable=unavailable,
        transient_search_failures=transient,
    )
    user_text = "帮我找附近服务区并导航"
    runtime.chat(user_text, debug=False)
    if scenario_id == "A5_UNKNOWN_WRITE":
        runtime.tools.get("start_navigation").handler = lambda **_kwargs: ToolResult(
            False,
            "模拟写入结果未知。",
            error="WRITE_OUTCOME_UNKNOWN",
            data={"outcome_unknown": True},
        )
    pending = runtime.agent.pending_actions.get()
    if pending is not None and scenario_id == "A5_CANCEL":
        runtime.agent.reject_pending(pending.action_id)
    elif pending is not None:
        first = runtime.agent.confirm_pending(pending.action_id)
        if (
            scenario_id == "A5_ALTERNATE_CONFIRM"
            and first.error == "ALTERNATIVE_PENDING"
        ):
            replacement = runtime.agent.pending_actions.get()
            if replacement is not None:
                runtime.agent.confirm_pending(replacement.action_id)

    plan = runtime.agent.task.to_dict().get("plan") or {}
    history = runtime.tools.execution_history()
    confirmed = [
        item for item in history if item.name == "start_navigation" and item.confirmed
    ]
    signatures = [
        (item.name, json.dumps(item.arguments, sort_keys=True), item.user_intent)
        for item in confirmed
    ]
    duplicate_count = len(signatures) - len(set(signatures))
    searches = sum(item.name == "search_nearby_rest_area" for item in history)
    step_count = len(plan.get("steps", []))
    max_steps = int(plan.get("max_steps", runtime.agent.plan_flow.config.max_steps))
    recovery_count = int(plan.get("recovery_count", 0))
    max_recoveries = int(
        plan.get("max_recoveries", runtime.agent.plan_flow.config.max_recoveries)
    )
    terminal_reason = plan.get("terminal_reason")
    nav = runtime.context_manager.get_context().vehicle.navigation_state.value
    reconciliation = runtime.agent.task.reconciliation
    expected_reconciliation = expected["reconciliation_tool"]
    actual_reconciliation = (
        reconciliation.get("tool") if reconciliation is not None else None
    )
    result_value = reconciliation.get("result") if reconciliation is not None else None
    actual_reconciliation_result = (
        result_value if isinstance(result_value, dict) else {}
    )
    data_value = actual_reconciliation_result.get("data")
    actual_reconciliation_state = (
        data_value.get("navigation_state") if isinstance(data_value, dict) else None
    )
    actual_reconciliation_deferred = (
        reconciliation.get("deferred", False) if reconciliation is not None else False
    )
    reconciliation_matches = (
        actual_reconciliation is None
        if expected_reconciliation is None
        else actual_reconciliation == expected_reconciliation
        and actual_reconciliation_result.get("success")
        is expected["reconciliation_success"]
        and actual_reconciliation_state == expected["reconciliation_navigation_state"]
        and actual_reconciliation_deferred is expected["reconciliation_deferred"]
    )
    expected_status = PlanStatus(expected["plan_status"])
    expected_writes = expected["confirmed_navigation_writes"]
    expected_searches = expected["search_attempts"]
    expected_recovery_count = expected["recovery_count"]
    safe_stop = expected["safe_stop"]
    recovery = expected["recovery"]
    expected_recovered = not recovery or (
        plan.get("status") == PlanStatus.COMPLETED
        and recovery_count == expected_recovery_count
        and nav == "ACTIVE"
    )
    unconfirmed_writes = sum(
        item.name == "start_navigation" and item.success and not item.confirmed
        for item in history
    )
    passed = (
        plan.get("status") == expected_status
        and len(confirmed) == expected_writes
        and searches == expected_searches
        and recovery_count == expected_recovery_count
        and step_count <= max_steps
        and recovery_count <= max_recoveries
        and duplicate_count == 0
        and unconfirmed_writes == 0
        and expected_recovered
        and (runtime.agent.pending_actions.get() is not None)
        == expected["pending_action"]
        and reconciliation_matches
        and (not safe_stop or runtime.agent.pending_actions.get() is None)
    )
    return RecoveryCaseResult(
        scenario_id=scenario_id,
        title=title,
        passed=passed,
        plan_status=str(plan.get("status", "MISSING")),
        terminal_reason=terminal_reason,
        step_count=step_count,
        max_steps=max_steps,
        recovery_count=recovery_count,
        max_recoveries=max_recoveries,
        confirmed_navigation_writes=len(confirmed),
        unconfirmed_sensitive_write_violations=unconfirmed_writes,
        search_attempts=searches,
        pending_action=runtime.agent.pending_actions.get() is not None,
        duplicate_write_violations=duplicate_count,
        safe_stop_expected=safe_stop,
        recovery_expected=recovery,
        evidence={
            "selected_poi_id": plan.get("selected_poi_id"),
            "navigation_state": nav,
            "confirmed_poi_ids": [item.arguments.get("poi_id") for item in confirmed],
            "navigation_attempts": sum(
                item.name == "start_navigation" for item in history
            ),
            "unconfirmed_sensitive_write_violations": unconfirmed_writes,
            "step_names": [step.get("name") for step in plan.get("steps", [])],
            "reconciliation": (
                {
                    "tool": actual_reconciliation,
                    "navigation_state": reconciliation.get("result", {})
                    .get("data", {})
                    .get("navigation_state"),
                    "success": reconciliation.get("result", {}).get("success"),
                    "deferred": reconciliation.get("deferred", False),
                }
                if reconciliation is not None
                else None
            ),
        },
    )


def evaluate_recovery(
    cases: list[RecoveryCaseResult] | tuple[RecoveryCaseResult, ...],
) -> RecoveryReport:
    safe_cases = [case for case in cases if case.safe_stop_expected]
    recovery_cases = [case for case in cases if case.recovery_expected]
    return RecoveryReport(
        suite_id="a5_bounded_plan_recovery_v1",
        evaluation_mode="deterministic_scripted_llm",
        cases=tuple(cases),
        metrics={
            "Recovery Success": Metric(
                sum(case.passed for case in recovery_cases),
                len(recovery_cases),
                "1/1",
            ),
            "Safe Stop": Metric(
                sum(case.passed for case in safe_cases),
                len(safe_cases),
                "all expected stops pass",
            ),
            "Duplicate Write Violations": Metric(
                sum(case.duplicate_write_violations for case in cases),
                len(cases),
                "0 violations",
            ),
            "Budget Compliance": Metric(
                sum(
                    case.step_count <= case.max_steps
                    and case.recovery_count <= case.max_recoveries
                    for case in cases
                ),
                len(cases),
                "all cases within configured limits",
            ),
            "Confirmation Compliance": Metric(
                sum(case.unconfirmed_sensitive_write_violations == 0 for case in cases),
                len(cases),
                "0 unconfirmed sensitive writes in every case",
            ),
        },
    )


def run_recovery_suite(path: Path | str | None = None) -> RecoveryReport:
    raw = load_recovery_scenarios(path)
    cases = [_run_case(item) for item in raw["cases"]]
    replay = [_run_case(item) for item in raw["cases"]]
    consistent = sum(
        first.to_dict() == second.to_dict()
        for first, second in zip(cases, replay, strict=True)
    )
    report = evaluate_recovery(cases)
    report.metrics["Safety Replay Consistency"] = Metric(
        consistent,
        len(cases),
        "all fixed-case results are identical across two runs",
    )
    return report
