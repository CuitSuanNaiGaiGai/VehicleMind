"""Create isolated copies of historical runs for a fresh semantic review."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence

from modules.vehicle_ai.evaluation.batch import load_frozen_cases


def _resolved_revision(repo_root: Path, revision: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def prepare_rejudge_copy(
    source_root: Path,
    destination: Path,
    *,
    dataset_root: Path,
    expected_case_ids: Sequence[str],
    repetitions: int,
    source_revision: str,
    grading_revision: str,
) -> dict:
    """Copy only verified run metadata/traces, leaving historic review untouched."""
    if destination.exists():
        raise FileExistsError(destination)
    if not expected_case_ids or repetitions < 1:
        raise ValueError("expected cases and positive repetitions are required")
    run_path = source_root / "run.json"
    run_bytes = run_path.read_bytes()
    run = json.loads(run_bytes)
    if run.get("status") != "completed":
        raise ValueError("cannot prepare an incomplete baseline")
    expected = {
        (case_id, trial_index)
        for case_id in expected_case_ids
        for trial_index in range(1, repetitions + 1)
    }
    entries = run.get("trials", [])
    by_key = {(item.get("case_id"), item.get("trial_index")): item for item in entries}
    if len(by_key) != len(entries) or set(by_key) != expected:
        raise ValueError("baseline trial identities do not match the frozen set")
    frozen_cases = {case.id: case for case in load_frozen_cases(dataset_root)}
    if set(expected_case_ids) - frozen_cases.keys():
        raise ValueError("expected baseline case is not in the frozen manifest")

    verified: list[tuple[dict, bytes]] = []
    for key in sorted(expected):
        item = by_key[key]
        if item.get("case_sha256") != frozen_cases[key[0]].sha256:
            raise ValueError(f"frozen case hash mismatch: {key}")
        relative = PurePosixPath(item.get("trace", ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"invalid trace path: {key}")
        trace_path = source_root.joinpath(*relative.parts)
        if not trace_path.resolve().is_relative_to(source_root.resolve()):
            raise ValueError(f"trace path escapes baseline run: {key}")
        trace_bytes = trace_path.read_bytes()
        if hashlib.sha256(trace_bytes).hexdigest() != item.get("trace_sha256"):
            raise ValueError(f"trace hash mismatch: {key}")
        verified.append((item, trace_bytes))

    repo_root = Path(__file__).resolve().parents[3]
    resolved_source_revision = _resolved_revision(repo_root, source_revision)
    resolved_grading_revision = _resolved_revision(repo_root, grading_revision)
    case_digest = hashlib.sha256()
    rubric_digest = hashlib.sha256()
    for case_id in sorted(expected_case_ids):
        case_digest.update(case_id.encode())
        case_digest.update(b"\0")
        case_digest.update(frozen_cases[case_id].sha256.encode())
        case_digest.update(b"\n")
        rubric_digest.update(case_id.encode())
        rubric_digest.update(b"\0")
        rubric_digest.update(
            (dataset_root / "rubrics" / f"{case_id}.yaml").read_bytes()
        )
        rubric_digest.update(b"\0")
    manifest_sha256 = hashlib.sha256(
        (dataset_root / "manifest.yaml").read_bytes()
    ).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        for item, trace_bytes in verified:
            relative = PurePosixPath(item["trace"])
            trace_path = temporary.joinpath(*relative.parts)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_bytes(trace_bytes)
        run["provenance"] = {
            **(run.get("provenance") or {}),
            "source_revision": resolved_source_revision,
            "source_tree_clean": None,
            "manifest_sha256": manifest_sha256,
            "case_set_sha256": case_digest.hexdigest(),
            "rubric_set_sha256": rubric_digest.hexdigest(),
            "provenance_kind": "historical-source-revision-from-published-record",
        }
        run["comparison_derivation"] = {
            "source_run_id": source_root.name,
            "source_run_sha256": hashlib.sha256(run_bytes).hexdigest(),
            "grading_revision": resolved_grading_revision,
            "trace_content_changed": False,
            "purpose": "fresh semantic review under current protocol",
        }
        (temporary / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()
        raise
    return {
        "run_id": destination.name,
        "trial_count": len(verified),
        "source_run_sha256": run["comparison_derivation"]["source_run_sha256"],
        "source_revision": resolved_source_revision,
        "grading_revision": resolved_grading_revision,
    }
