from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from modules.vehicle_ai.evaluation.judge_calibration import (
    calibrate,
    cohens_kappa,
    load_calibration_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_PATH = REPO_ROOT / "eval/judge_calibration/labels.yaml"


def test_kappa_matches_known_values() -> None:
    # po=0.5, pe=0.5 -> kappa = 0
    assert cohens_kappa(["p", "p", "f", "f"], ["p", "f", "p", "f"]) == 0.0
    # po=0.75, pe=0.5 -> kappa = 0.5
    assert cohens_kappa(["p", "p", "p", "f"], ["p", "p", "f", "f"]) == 0.5
    assert cohens_kappa(["p", "f"], ["p", "f"]) == 1.0


def test_kappa_handles_degenerate_constant_raters() -> None:
    assert cohens_kappa(["pass", "pass"], ["pass", "pass"]) == 1.0
    assert cohens_kappa(["pass", "pass"], ["fail", "fail"]) == 0.0


@pytest.mark.parametrize(
    "first,second",
    [([], []), (["pass"], ["pass", "fail"])],
)
def test_kappa_rejects_unpaired_sequences(first, second) -> None:
    with pytest.raises(ValueError):
        cohens_kappa(first, second)


def _labels_document(entries: list[dict], **overrides) -> dict:
    document = {
        "schema_version": 1,
        "run": {
            "directory": "runs/fake",
            "provider": "test",
            "model": "stub",
            "source_sha256": "a" * 64,
        },
        "label_provenance": {
            "labeler": "test labeler",
            "labeled_at_utc": "2026-10-23T00:00:00+00:00",
            "independent_human_review": False,
        },
        "entries": entries,
    }
    document.update(overrides)
    return document


def _entry(**overrides) -> dict:
    entry = {
        "case_id": "C01",
        "trial_index": 1,
        "verdict": "pass",
        "note": "测试标注。",
    }
    entry.update(overrides)
    return entry


def _write_run(tmp_path: Path, trials: list[dict], **overrides) -> Path:
    run_root = tmp_path / "runs" / "fake"
    run_root.mkdir(parents=True)
    reviewed = {
        "provider": "test",
        "model": "stub",
        "source_sha256": "a" * 64,
        "trials": trials,
    }
    reviewed.update(overrides)
    (run_root / "reviewed.json").write_text(
        json.dumps(reviewed, ensure_ascii=False), encoding="utf-8"
    )
    return run_root


def _trial(case_id: str, verdict: str, trial_index: int = 1) -> dict:
    return {"case_id": case_id, "trial_index": trial_index, "semantic_verdict": verdict}


def _write_labels(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "labels.yaml"
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_loader_accepts_the_checked_in_labels() -> None:
    document = load_calibration_labels(LABELS_PATH)
    assert len(document["entries"]) == 20
    assert document["label_provenance"]["independent_human_review"] is False


@pytest.mark.parametrize(
    "document",
    [
        _labels_document([], ),
        _labels_document([_entry(verdict="maybe")]),
        _labels_document([_entry(extra="field")]),
        _labels_document([_entry(trial_index=0)]),
        _labels_document([_entry(), _entry()]),
        _labels_document([_entry()], schema_version=2),
    ],
)
def test_loader_rejects_malformed_documents(document, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_calibration_labels(_write_labels(tmp_path, document))


def test_calibrate_computes_agreement_and_disagreements(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        [
            _trial("C01", "pass"),
            _trial("C02", "fail"),
            _trial("C03", "pass"),
            _trial("C04", "fail"),
        ],
    )
    document = _labels_document(
        [
            _entry(case_id="C01", verdict="pass", note="一致通过。"),
            _entry(case_id="C02", verdict="fail", note="一致失败。"),
            _entry(case_id="C03", verdict="fail", note="分歧样本。"),
            _entry(case_id="C04", verdict="pass", note="分歧样本。"),
        ]
    )
    report = calibrate(_write_labels(tmp_path, document), repo_root=tmp_path)
    assert report["total"] == 4
    assert report["agreement"] == 0.5
    assert report["cohens_kappa"] == 0.0
    assert report["confusion"] == {
        "pass_pass": 1,
        "fail_fail": 1,
        "pass_fail": 1,
        "fail_pass": 1,
    }
    assert [item["case_id"] for item in report["disagreements"]] == ["C03", "C04"]


def test_calibrate_rejects_labels_bound_to_a_different_run(tmp_path: Path) -> None:
    _write_run(tmp_path, [_trial("C01", "pass")], source_sha256="b" * 64)
    with pytest.raises(ValueError):
        calibrate(_write_labels(tmp_path, _labels_document([_entry()])), repo_root=tmp_path)


def test_calibrate_rejects_entries_missing_from_the_run(tmp_path: Path) -> None:
    _write_run(tmp_path, [_trial("C01", "pass")])
    document = _labels_document(
        [_entry(case_id="C01"), _entry(case_id="C404", verdict="fail", note="不在 run 中。")]
    )
    with pytest.raises(ValueError):
        calibrate(_write_labels(tmp_path, document), repo_root=tmp_path)


def test_checked_in_labels_calibrate_against_the_recorded_run() -> None:
    report = calibrate(LABELS_PATH)
    assert report["total"] == 20
    assert report["agreement"] >= 0.8
    assert report["cohens_kappa"] >= 0.6
    assert {item["case_id"] for item in report["disagreements"]} <= {"C03", "M07"}
