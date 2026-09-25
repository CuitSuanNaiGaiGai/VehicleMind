from __future__ import annotations

import json

from modules.vehicle_ai.evaluation.recovery import (
    evaluate_recovery,
    load_recovery_scenarios,
    run_recovery_suite,
)
from modules.vehicle_ai.evaluation.recovery_report import write_recovery_report


def test_recovery_suite_covers_success_recovery_stops_and_budgets() -> None:
    report = run_recovery_suite()

    assert len(report.cases) == 7
    assert {case.scenario_id for case in report.cases} == {
        "A5_SUCCESS",
        "A5_ALTERNATE_CONFIRM",
        "A5_NO_RESULT",
        "A5_CANCEL",
        "A5_UNKNOWN_WRITE",
        "A5_INVALID_TARGET",
        "A5_READ_RETRY",
    }
    assert report.metrics["Recovery Success"].numerator == 1
    assert report.metrics["Recovery Success"].denominator == 1
    assert report.metrics["Safe Stop"].numerator == 4
    assert report.metrics["Safe Stop"].denominator == 4
    assert report.metrics["Duplicate Write Violations"].numerator == 0
    assert report.metrics["Duplicate Write Violations"].denominator == 7
    assert report.metrics["Budget Compliance"].numerator == 7
    assert report.metrics["Budget Compliance"].denominator == 7
    assert report.metrics["Confirmation Compliance"].numerator == 7
    assert report.metrics["Confirmation Compliance"].denominator == 7
    assert report.metrics["Safety Replay Consistency"].numerator == 7
    assert report.metrics["Safety Replay Consistency"].denominator == 7
    assert all(case.passed for case in report.cases)
    unknown_write = next(
        case for case in report.cases if case.scenario_id == "A5_UNKNOWN_WRITE"
    )
    assert unknown_write.evidence["reconciliation"]["tool"] == "get_vehicle_status"
    assert unknown_write.evidence["reconciliation"]["navigation_state"] == "IDLE"
    assert unknown_write.evidence["reconciliation"]["success"] is True
    assert unknown_write.evidence["reconciliation"]["deferred"] is False


def test_recovery_grader_uses_expected_results_from_yaml(tmp_path) -> None:
    scenarios = load_recovery_scenarios()
    cancel = next(item for item in scenarios["cases"] if item["id"] == "A5_CANCEL")
    cancel["expected"]["plan_status"] = "COMPLETED"
    scenario_path = tmp_path / "changed-expectation.yaml"
    scenario_path.write_text(json.dumps(scenarios), encoding="utf-8")

    report = run_recovery_suite(scenario_path)

    failed = next(case for case in report.cases if case.scenario_id == "A5_CANCEL")
    assert failed.passed is False
    assert report.metrics["Safe Stop"].numerator == 3
    assert report.metrics["Safe Stop"].denominator == 4


def test_recovery_grader_reports_failed_cases_in_denominator() -> None:
    report = run_recovery_suite()
    failed = next(case for case in report.cases if case.scenario_id == "A5_CANCEL")
    failed.passed = False

    updated = evaluate_recovery(report.cases)

    assert updated.metrics["Safe Stop"].numerator == 3
    assert updated.metrics["Safe Stop"].denominator == 4
    assert updated.metrics["Budget Compliance"].denominator == 7


def test_recovery_grader_requires_expected_unknown_write_readback(tmp_path) -> None:
    scenarios = load_recovery_scenarios()
    unknown = next(
        item for item in scenarios["cases"] if item["id"] == "A5_UNKNOWN_WRITE"
    )
    unknown["expected"]["reconciliation_navigation_state"] = "ACTIVE"
    scenario_path = tmp_path / "wrong-readback-expectation.yaml"
    scenario_path.write_text(json.dumps(scenarios), encoding="utf-8")

    report = run_recovery_suite(scenario_path)

    failed = next(
        case for case in report.cases if case.scenario_id == "A5_UNKNOWN_WRITE"
    )
    assert failed.passed is False
    assert failed.evidence["reconciliation"]["navigation_state"] == "IDLE"


def test_recovery_report_shows_chinese_metrics_and_case_evidence(tmp_path) -> None:
    report = run_recovery_suite()

    outputs = write_recovery_report(report, tmp_path)

    html = outputs["html"].read_text(encoding="utf-8")
    assert "Recovery Success（恢复成功率）" in html
    assert "Safe Stop（安全停止）" in html
    assert "1 / 1" in html
    assert "A5_ALTERNATE_CONFIRM" in html
    assert "rest_area_002" in html
    assert "已取消" in html
    assert "不作为在线 LLM 语义正确率" in html
