"""Cohen's kappa calibration of the recorded AI judge against reference labels.

The reference labels live in ``eval/judge_calibration/labels.yaml`` and are
bound to one recorded run via its ``source_sha256``. The judge verdicts are
read from that run's ``reviewed.json`` — calibration itself costs no API
calls and reproduces exactly what the production judge emitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_VERDICTS = {"pass", "fail"}


def cohens_kappa(first: list[str], second: list[str]) -> float:
    """Unweighted Cohen's kappa for two paired label sequences."""
    if not first or len(first) != len(second):
        raise ValueError("kappa requires equally sized non-empty label sequences")
    total = len(first)
    observed = sum(1 for a, b in zip(first, second, strict=True) if a == b) / total
    labels = sorted(set(first) | set(second))
    expected = sum(
        (first.count(label) / total) * (second.count(label) / total) for label in labels
    )
    if expected == 1.0:
        # Both raters used a single label: kappa is degenerate. Report 1.0 for
        # perfect agreement and 0.0 otherwise, so the threshold stays strict.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def load_calibration_labels(path: Path) -> dict[str, Any]:
    """Load and strictly validate the reference label document."""
    document = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")), "calibration labels"
    )
    if set(document) != {
        "schema_version",
        "run",
        "label_provenance",
        "entries",
    } or document["schema_version"] != 1:
        raise ValueError("calibration labels schema mismatch")
    run = _mapping(document["run"], "run")
    if set(run) != {"directory", "provider", "model", "source_sha256"}:
        raise ValueError("calibration run fields mismatch")
    for field in ("directory", "provider", "model", "source_sha256"):
        _text(run[field], f"run.{field}")
    provenance = _mapping(document["label_provenance"], "label_provenance")
    if set(provenance) != {"labeler", "labeled_at_utc", "independent_human_review"}:
        raise ValueError("label provenance fields mismatch")
    if not isinstance(provenance["independent_human_review"], bool):
        raise ValueError("independent_human_review must be boolean")
    for field in ("labeler", "labeled_at_utc"):
        _text(provenance[field], f"label_provenance.{field}")
    entries = document["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("calibration entries must be a non-empty list")
    seen: set[tuple[str, int]] = set()
    for item in entries:
        entry = _mapping(item, "entry")
        if set(entry) != {"case_id", "trial_index", "verdict", "note"}:
            raise ValueError("calibration entry fields mismatch")
        _text(entry["case_id"], "entry.case_id")
        _text(entry["note"], "entry.note")
        if entry["verdict"] not in _VERDICTS:
            raise ValueError("calibration verdict must be pass or fail")
        if type(entry["trial_index"]) is not int or entry["trial_index"] < 1:
            raise ValueError("entry.trial_index must be a positive integer")
        key = (entry["case_id"], entry["trial_index"])
        if key in seen:
            raise ValueError(f"duplicate calibration entry: {key}")
        seen.add(key)
    return document


def calibrate(labels_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Compare reference labels with the recorded judge verdicts of one run."""
    document = load_calibration_labels(labels_path)
    run_ref = document["run"]
    run_root = Path(run_ref["directory"])
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    reviewed = json.loads((run_root / "reviewed.json").read_text(encoding="utf-8"))
    if (
        reviewed.get("provider") != run_ref["provider"]
        or reviewed.get("model") != run_ref["model"]
        or reviewed.get("source_sha256") != run_ref["source_sha256"]
    ):
        raise ValueError("calibration labels reference a different run")
    judge_by_key = {
        (item["case_id"], item["trial_index"]): item["semantic_verdict"]
        for item in reviewed["trials"]
    }
    reference: list[str] = []
    judge: list[str] = []
    disagreements: list[dict[str, Any]] = []
    confusion = {"pass_pass": 0, "fail_fail": 0, "pass_fail": 0, "fail_pass": 0}
    for item in document["entries"]:
        key = (item["case_id"], item["trial_index"])
        if key not in judge_by_key:
            raise ValueError(f"calibration entry not present in the run: {key}")
        reference.append(item["verdict"])
        judged = judge_by_key[key]
        judge.append(judged)
        confusion[f"{item['verdict']}_{judged}"] += 1
        if item["verdict"] != judged:
            disagreements.append(
                {
                    "case_id": item["case_id"],
                    "trial_index": item["trial_index"],
                    "reference": item["verdict"],
                    "judge": judged,
                    "note": item["note"],
                }
            )
    total = len(reference)
    agreement = sum(1 for a, b in zip(reference, judge, strict=True) if a == b) / total
    return {
        "schema_version": 1,
        "labels_path": str(labels_path),
        "run": dict(run_ref),
        "label_provenance": dict(document["label_provenance"]),
        "total": total,
        "agreement": agreement,
        "cohens_kappa": cohens_kappa(reference, judge),
        "confusion": confusion,
        "disagreements": disagreements,
    }
