from __future__ import annotations

from typing import Any

import cv2

from modules.cabin.fatigue.eye_state import EyeStateAnalyzer
from modules.cabin.fatigue.mouth_state import MouthStateAnalyzer
from modules.cabin.presence.driver_presence import DriverPresence
from modules.cabin.state.driver_state import DriverState


def _landmark_point(frame, landmark) -> tuple[int, int]:
    height, width = frame.shape[:2]
    return int(landmark.x * width), int(landmark.y * height)


def draw_face_landmarks(frame, face) -> None:
    height, width = frame.shape[:2]
    for landmark in face:
        x, y = _landmark_point(frame, landmark)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, (80, 220, 120), -1)


def draw_eye_landmarks(frame, face) -> None:
    for index in EyeStateAnalyzer.eye_indices():
        cv2.circle(
            frame,
            _landmark_point(frame, face[index]),
            3,
            (0, 255, 255),
            -1,
        )


def draw_mouth_landmarks(frame, face) -> None:
    for index in MouthStateAnalyzer.landmark_indices():
        cv2.circle(
            frame,
            _landmark_point(frame, face[index]),
            3,
            (255, 120, 255),
            -1,
        )


def _state_color(state: DriverState) -> tuple[int, int, int]:
    return {
        DriverState.NORMAL: (80, 220, 80),
        DriverState.WARMING_UP: (0, 220, 255),
        DriverState.SUSPECTED: (0, 165, 255),
        DriverState.DROWSY: (0, 0, 255),
    }.get(state, (180, 180, 180))


def _text(
    frame,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.58,
    color: tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_text(
    frame,
    text: str,
    x: int,
    y: int,
    scale: float = 0.60,
    color: tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
) -> None:
    _text(
        frame,
        text,
        x,
        y,
        scale=scale,
        color=color,
        thickness=thickness,
    )


def _row(
    frame,
    label: str,
    value: str,
    left: int,
    y: int,
    *,
    color: tuple[int, int, int] = (230, 230, 230),
) -> int:
    _text(frame, label, left, y)
    _text(frame, value, left + 180, y, color=color, thickness=2)
    return y + 32


def _risk_color(risk: str) -> tuple[int, int, int]:
    return {
        "LOW": (80, 220, 80),
        "MEDIUM": (0, 165, 255),
        "HIGH": (0, 0, 255),
    }.get(risk, (170, 170, 170))


def _assistant_text(state: DriverState, recommendation: Any) -> list[str]:
    if state is DriverState.DROWSY:
        lines = ["FATIGUE WARNING", "Consider taking a break."]
        if recommendation is not None:
            lines.extend(
                [
                    recommendation.name,
                    (
                        f"{recommendation.distance_km:.1f} km | "
                        f"ETA {recommendation.eta_minutes} min"
                    ),
                ]
            )
        return lines
    if state is DriverState.SUSPECTED:
        return ["Fatigue signs detected."]
    if state is DriverState.NORMAL:
        return ["Driver condition normal."]
    if state is DriverState.WARMING_UP:
        return ["Collecting observations..."]
    return ["Driver unavailable."]


def draw_cabin_dashboard(
    frame,
    presence_result,
    driver_state_result,
    face_visible: bool,
    eye_closed: bool | None,
    current_yawn: bool,
    blink_count: int,
    recommendation,
) -> None:
    height, width = frame.shape[:2]
    left = max(0, width - 390) + 25
    overlay = frame.copy()
    cv2.rectangle(overlay, (left - 25, 0), (width, height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    _text(frame, "VEHICLEMIND", left, 45, scale=0.82, thickness=2)
    _text(frame, "CABIN PERCEPTION", left, 75, color=(170, 170, 170))
    y = 115

    presence = presence_result.state
    presence_color = {
        DriverPresence.PRESENT: (80, 220, 80),
        DriverPresence.ABSENT: (0, 0, 255),
    }.get(presence, (0, 220, 255))
    y = _row(frame, "Presence", presence.value, left, y, color=presence_color)
    observation = "TRACKING" if face_visible else "TEMP LOST"
    if presence is not DriverPresence.PRESENT:
        observation = "NO FACE"
    y = _row(frame, "Observation", observation, left, y)

    state = DriverState.UNKNOWN
    risk = "UNKNOWN"
    if driver_state_result is not None:
        state = driver_state_result.state
        risk = driver_state_result.risk_level.value
    y += 12
    y = _row(frame, "State", state.value, left, y, color=_state_color(state))
    y = _row(frame, "Risk", risk, left, y, color=_risk_color(risk))

    y += 12
    if driver_state_result is None or presence is not DriverPresence.PRESENT:
        perclos, closure, yawns = "--", "--", "--"
    else:
        perclos = (
            f"{driver_state_result.perclos * 100:.1f}%"
            if driver_state_result.perclos_ready
            else "WARMING UP"
        )
        closure = f"{driver_state_result.continuous_eye_closure:.1f}s"
        yawns = str(driver_state_result.recent_yawns)
    y = _row(frame, "PERCLOS", perclos, left, y)
    y = _row(frame, "Eye Closure", closure, left, y)
    y = _row(frame, "Recent Yawns", yawns, left, y)

    eye_text = "--" if eye_closed is None else ("CLOSED" if eye_closed else "OPEN")
    eye_color = (
        (170, 170, 170)
        if eye_closed is None
        else ((0, 0, 255) if eye_closed else (80, 220, 80))
    )
    y += 12
    y = _row(frame, "Eye", eye_text, left, y, color=eye_color)
    y = _row(frame, "Blinks", str(blink_count), left, y)
    y = _row(frame, "Yawn", "DETECTED" if current_yawn else "NO", left, y)

    y += 15
    cv2.line(frame, (left, y), (width - 25, y), (90, 90, 90), 1)
    y += 30
    _text(frame, "SAFETY ASSISTANT", left, y, color=(160, 160, 160), thickness=2)
    for line in _assistant_text(state, recommendation):
        y += 28
        _text(frame, line, left, y, color=_state_color(state))
