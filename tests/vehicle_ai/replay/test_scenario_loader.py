from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modules.vehicle_ai.replay import load_replay_scenario


ROOT = Path(__file__).resolve().parents[3]


def _observation(**values: object) -> dict[str, object]:
    return {
        "source": "recorded_test",
        "confidence": 0.9,
        "valid": True,
        "values": values,
    }


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scenario_id": "test-scenario",
        "title": "Test scenario",
        "description": "Strict loader fixture",
        "media": {
            "cabin": "assets/demo/cabin_demo.gif",
            "road": "assets/demo/driving_perception.gif",
        },
        "agent_responses": [
            {
                "content": "A nearby rest area is available.",
                "tool_calls": [
                    {
                        "id": "call-search",
                        "name": "search_nearby_rest_area",
                        "arguments": {},
                    }
                ],
            }
        ],
        "steps": [
            {
                "at_ms": 0,
                "vehicle": _observation(speed_kmh=68.0, gear="D"),
                "cabin": _observation(
                    presence="PRESENT",
                    driver_state="NORMAL",
                    risk="LOW",
                    eye_closed=None,
                ),
            },
            {"at_ms": 100, "user_text": "Find a rest area."},
            {"at_ms": 200, "confirm_pending": True},
        ],
        "expected": {
            "event_types": ["HIGH_RISK_DETECTED"],
            "successful_tools": ["search_nearby_rest_area"],
            "final_vehicle": {"navigation_state": "ACTIVE"},
            "unauthorized_sensitive_executions": 0,
        },
    }


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_repository_showcase_scenario_loads() -> None:
    scenario = load_replay_scenario(
        ROOT / "assets/scenarios/drowsy_rest_stop.yaml",
        repository_root=ROOT,
    )

    assert scenario.schema_version == 1
    assert scenario.scenario_id == "drowsy-rest-stop"
    assert scenario.steps[0].at_ms == 0
    assert scenario.media["cabin"] == ROOT / "assets/demo/cabin_demo.gif"
    assert scenario.media["road"] == ROOT / "assets/demo/driving_perception.gif"
    assert scenario.expected.unauthorized_sensitive_executions == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda doc: doc.update({"mystery": True}), "mystery"),
        (lambda doc: doc.update({"scenario_id": " "}), "scenario_id"),
        (lambda doc: doc.update({"agent_responses": []}), "agent_responses"),
        (lambda doc: doc["steps"][0].update({"at_ms": -1}), r"steps\[0\].at_ms"),
        (
            lambda doc: doc["steps"][1].update({"at_ms": 0}),
            r"steps\[1\].at_ms",
        ),
        (
            lambda doc: doc["steps"][0]["cabin"].update({"confidence": 1.1}),
            r"steps\[0\].cabin.confidence",
        ),
        (
            lambda doc: doc["steps"][0]["cabin"].update({"unexpected": 1}),
            r"steps\[0\].cabin.unexpected",
        ),
    ],
)
def test_invalid_scenario_documents_are_rejected(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    document = _document()
    mutation(document)

    with pytest.raises(ValueError, match=match):
        load_replay_scenario(_write(tmp_path, document), repository_root=ROOT)


@pytest.mark.parametrize(
    "media_path",
    [
        "/Users/example/private.gif",
        "../private.gif",
        "assets/demo/missing.gif",
    ],
)
def test_unsafe_or_missing_media_is_rejected(
    tmp_path: Path,
    media_path: str,
) -> None:
    document = _document()
    document["media"]["cabin"] = media_path

    with pytest.raises(ValueError, match="media.cabin"):
        load_replay_scenario(_write(tmp_path, document), repository_root=ROOT)


def test_confirmation_cannot_precede_a_user_turn(tmp_path: Path) -> None:
    document = _document()
    document["steps"] = [
        {"at_ms": 0, "confirm_pending": True},
        {"at_ms": 100, "user_text": "Find a rest area."},
    ]

    with pytest.raises(ValueError, match=r"steps\[0\].confirm_pending"):
        load_replay_scenario(_write(tmp_path, document), repository_root=ROOT)


def test_loaded_scenario_mappings_are_immutable(tmp_path: Path) -> None:
    document = _document()
    scenario = load_replay_scenario(_write(tmp_path, document), repository_root=ROOT)

    assert scenario.steps[0].cabin is not None
    with pytest.raises(TypeError):
        scenario.steps[0].cabin.values["risk"] = "HIGH"  # type: ignore[index]
