from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    split: str
    category: str
    review_status: str
    steps: tuple[dict[str, Any], ...]
    expected: dict[str, Any]

    @property
    def sha256(self) -> str:
        payload = {
            "id": self.id, "split": self.split, "category": self.category,
            "review_status": self.review_status, "steps": self.steps,
            "expected": self.expected,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EvaluationCase:
        allowed = {"id", "split", "category", "review_status", "steps", "expected"}
        if set(data) != allowed:
            raise ValueError(f"case fields mismatch: {sorted(set(data) ^ allowed)}")
        if data["split"] not in {"dev", "heldout"}:
            raise ValueError("split must be dev or heldout")
        if data["review_status"] not in {"candidate", "reviewed"}:
            raise ValueError("invalid review_status")
        if not isinstance(data["steps"], list) or not data["steps"]:
            raise ValueError("steps must be a non-empty list")
        if not isinstance(data["id"], str) or not data["id"].strip():
            raise ValueError("id must be non-empty text")
        if not isinstance(data["category"], str) or not data["category"].strip():
            raise ValueError("category must be non-empty text")
        expected = data["expected"]
        if not isinstance(expected, dict):
            raise ValueError("expected must be a mapping")
        if set(expected) != {
            "tools", "final_vehicle", "required_facts", "forbidden_phrases"
        }:
            raise ValueError("expected fields mismatch")
        if not isinstance(expected["tools"], list):
            raise ValueError("expected.tools must be a list")
        for tool in expected["tools"]:
            if not isinstance(tool, dict) or set(tool) != {"name", "arguments"}:
                raise ValueError("expected tool must contain name and arguments")
            if not isinstance(tool["name"], str) or not isinstance(
                tool["arguments"], dict
            ):
                raise ValueError("invalid expected tool")
        if not isinstance(expected["final_vehicle"], dict):
            raise ValueError("expected.final_vehicle must be a mapping")
        for field in ("required_facts", "forbidden_phrases"):
            if not isinstance(expected[field], list) or not all(
                isinstance(item, str) and item for item in expected[field]
            ):
                raise ValueError(f"expected.{field} must be a text list")
        for step in data["steps"]:
            if not isinstance(step, dict) or set(step) - {
                "at_ms", "cabin", "road", "vehicle", "user_text",
                "confirm_pending", "reject_pending"
            }:
                raise ValueError("invalid step")
            if "at_ms" in step and (
                type(step["at_ms"]) is not int or step["at_ms"] < 0
            ):
                raise ValueError("step.at_ms must be a non-negative integer")
            for domain in ("cabin", "road", "vehicle"):
                if domain in step and not isinstance(step[domain], dict):
                    raise ValueError(f"step.{domain} must be a mapping")
            if "user_text" in step and not isinstance(step["user_text"], str):
                raise ValueError("step.user_text must be text")
            for field in ("confirm_pending", "reject_pending"):
                if field in step and not isinstance(step[field], bool):
                    raise ValueError(f"step.{field} must be boolean")
        if not any("user_text" in step for step in data["steps"]):
            raise ValueError("at least one user_text is required")
        return cls(
            id=str(data["id"]), split=data["split"],
            category=str(data["category"]), review_status=data["review_status"],
            steps=tuple(dict(step) for step in data["steps"]),
            expected=dict(expected),
        )
