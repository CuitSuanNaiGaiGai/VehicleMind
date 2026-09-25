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
    policy_expectations: dict[str, Any] | None = None

    @property
    def sha256(self) -> str:
        payload = {
            "id": self.id,
            "split": self.split,
            "category": self.category,
            "review_status": self.review_status,
            "steps": self.steps,
            "expected": self.expected,
        }
        if self.policy_expectations is not None:
            payload["policy_expectations"] = self.policy_expectations
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EvaluationCase:
        required = {"id", "split", "category", "review_status", "steps", "expected"}
        allowed = required | {"policy_expectations"}
        if not required.issubset(data) or set(data) - allowed:
            raise ValueError(f"case fields mismatch: {sorted(set(data) ^ required)}")
        policy_expectations = data.get("policy_expectations")
        if policy_expectations is not None:
            if not isinstance(policy_expectations, dict) or set(
                policy_expectations
            ) != {"reasons", "required_phrases", "forbidden_phrases"}:
                raise ValueError("invalid policy_expectations fields")
            for field in ("reasons", "required_phrases", "forbidden_phrases"):
                if not isinstance(policy_expectations[field], list) or not all(
                    isinstance(value, str) and value
                    for value in policy_expectations[field]
                ):
                    raise ValueError(f"policy_expectations.{field} must be a text list")
        if data["split"] not in {"dev", "heldout"}:
            raise ValueError("split must be dev or heldout")
        if data["review_status"] not in {"candidate", "ai_reviewed", "reviewed"}:
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
            "tools",
            "final_vehicle",
            "required_facts",
            "forbidden_phrases",
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
        previous_at_ms = -1
        for step in data["steps"]:
            if not isinstance(step, dict) or set(step) - {
                "at_ms",
                "cabin",
                "road",
                "road_quality",
                "cabin_quality",
                "policy_probe",
                "tool_failure",
                "vehicle",
                "user_text",
                "confirm_pending",
                "reject_pending",
            }:
                raise ValueError("invalid step")
            if "at_ms" in step and (
                type(step["at_ms"]) is not int or step["at_ms"] < 0
            ):
                raise ValueError("step.at_ms must be a non-negative integer")
            if "at_ms" in step:
                if step["at_ms"] < previous_at_ms:
                    raise ValueError("step.at_ms must be monotonic")
                previous_at_ms = step["at_ms"]
            for domain in ("cabin", "road", "vehicle"):
                if domain in step and not isinstance(step[domain], dict):
                    raise ValueError(f"step.{domain} must be a mapping")
            if "road_quality" in step and step["road_quality"] != {"valid": False}:
                raise ValueError("step.road_quality must be {valid: false}")
            if "cabin_quality" in step and step["cabin_quality"] != {"valid": False}:
                raise ValueError("step.cabin_quality must be {valid: false}")
            if "policy_probe" in step and (
                policy_expectations is None
                or step["policy_probe"] != "high_driver_risk"
            ):
                raise ValueError(
                    "policy_probe is only allowed for A2 high-driver-risk cases"
                )
            if "tool_failure" in step:
                failure = step["tool_failure"]
                if (
                    not isinstance(failure, dict)
                    or set(failure) != {"name", "error"}
                    or not all(
                        isinstance(value, str) and value for value in failure.values()
                    )
                ):
                    raise ValueError("step.tool_failure requires name and error")
            if "user_text" in step and not isinstance(step["user_text"], str):
                raise ValueError("step.user_text must be text")
            for field in ("confirm_pending", "reject_pending"):
                if field in step and not isinstance(step[field], bool):
                    raise ValueError(f"step.{field} must be boolean")
        if not any("user_text" in step for step in data["steps"]):
            raise ValueError("at least one user_text is required")
        return cls(
            id=str(data["id"]),
            split=data["split"],
            category=str(data["category"]),
            review_status=data["review_status"],
            steps=tuple(dict(step) for step in data["steps"]),
            expected=dict(expected),
            policy_expectations=(
                dict(policy_expectations) if policy_expectations is not None else None
            ),
        )
