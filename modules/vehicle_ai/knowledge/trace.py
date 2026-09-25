"""Append one structured retrieval record per Agent knowledge call."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from modules.vehicle_ai.knowledge.models import RetrievalResult


def record_retrieval_trace(path: Path, result: RetrievalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(result)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
