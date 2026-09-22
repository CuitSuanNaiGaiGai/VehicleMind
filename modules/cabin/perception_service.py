from __future__ import annotations

import time

from pathlib import Path

from modules.config import CabinPerceptionConfig
from modules.observation import ObservationSequencer
from modules.cabin.snapshot import CabinPerceptionSnapshot


from modules.cabin.face.landmarks import (
    FaceLandmarkDetector,
)

from modules.cabin.fatigue.eye_state import (
    EyeStateAnalyzer,
)

from modules.cabin.fatigue.blink import (
    BlinkDetector,
)

from modules.cabin.fatigue.perclos import (
    PerclosEstimator,
)

from modules.cabin.fatigue.mouth_state import (
    MouthStateAnalyzer,
)

from modules.cabin.fatigue.yawn import (
    YawnDetector,
)

from modules.cabin.presence.driver_presence import (
    DriverPresence,
    DriverPresenceTracker,
)

from modules.cabin.state.driver_state import (
    DriverStateEstimator,
    DriverStateResult,
)


# ============================================================
# Service
# ============================================================


class CabinPerceptionService:
    """
    Reusable Cabin Intelligence pipeline.

    Algorithm settings are kept consistent with the current
    apps/cabin_demo/main.py implementation.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: CabinPerceptionConfig | None = None,
    ):
        if config is None:
            config = CabinPerceptionConfig.load_default()

        # ====================================================
        # Face
        # ====================================================

        self.face_detector = FaceLandmarkDetector(model_path=(Path(model_path)))

        # ====================================================
        # Presence
        # ====================================================

        self.presence_tracker = DriverPresenceTracker(
            present_confirm_seconds=(config.presence.present_confirm_seconds),
            absence_timeout_seconds=(config.presence.absence_timeout_seconds),
            startup_timeout_seconds=(config.presence.startup_timeout_seconds),
        )

        # ====================================================
        # Eye
        # ====================================================

        self.eye_analyzer = EyeStateAnalyzer(
            ear_threshold=config.eye.ear_threshold,
        )

        self.blink_detector = BlinkDetector(
            min_closed_frames=config.blink.min_closed_frames,
            max_closed_frames=config.blink.max_closed_frames,
        )

        self.perclos_estimator = PerclosEstimator(
            window_seconds=config.perclos.window_seconds,
            min_observation_seconds=(config.perclos.min_observation_seconds),
        )

        # ====================================================
        # Mouth / Yawn
        # ====================================================

        self.mouth_analyzer = MouthStateAnalyzer(
            mar_threshold=config.mouth.mar_threshold,
        )

        self.yawn_detector = YawnDetector(
            min_open_seconds=config.yawn.min_open_seconds,
        )

        # ====================================================
        # Driver state
        # ====================================================

        self.driver_state_estimator = DriverStateEstimator(
            suspected_perclos=(config.driver_state.suspected_perclos),
            drowsy_perclos=config.driver_state.drowsy_perclos,
            suspected_closure_seconds=(config.driver_state.suspected_closure_seconds),
            drowsy_closure_seconds=(config.driver_state.drowsy_closure_seconds),
            yawn_window_seconds=(config.driver_state.yawn_window_seconds),
            suspected_yawns=config.driver_state.suspected_yawns,
        )

        # ====================================================
        # Runtime
        # ====================================================

        self.start_time = time.perf_counter()
        self._observations = ObservationSequencer("cabin_perception")

        self.last_driver_state_result: DriverStateResult | None = None

        self.last_yawn_count = 0

        self.last_blink_count = 0

    # ========================================================
    # Timestamp
    # ========================================================

    def current_timestamp_ms(
        self,
    ) -> int:

        return int((time.perf_counter() - self.start_time) * 1000)

    # ========================================================
    # Process one frame
    # ========================================================

    def process_frame(
        self,
        frame,
        timestamp_ms: int | None = None,
    ) -> CabinPerceptionSnapshot:
        processing_started = time.perf_counter()

        if timestamp_ms is None:
            timestamp_ms = self.current_timestamp_ms()

        # ====================================================
        # Face
        # ====================================================

        faces = self.face_detector.detect(
            frame,
            timestamp_ms,
        )

        face_visible = len(faces) > 0

        # ====================================================
        # Presence
        # ====================================================

        presence_result = self.presence_tracker.update(
            timestamp_ms=timestamp_ms,
            face_detected=face_visible,
        )

        eye_closed_now = None

        current_yawn = False

        perclos_value = None

        perclos_ready = False
        driver_state_result: DriverStateResult | None

        # ====================================================
        # Valid face observation
        # ====================================================

        if face_visible:
            face = faces[0]

            height, width = frame.shape[:2]

            # ------------------------------------------------
            # Eye
            # ------------------------------------------------

            eye_result = self.eye_analyzer.analyze(
                face,
                width,
                height,
            )

            eye_closed_now = bool(eye_result.is_closed)

            # ------------------------------------------------
            # Blink
            # ------------------------------------------------

            blink_result = self.blink_detector.update(eye_result.is_closed)

            self.last_blink_count = blink_result.blink_count

            # ------------------------------------------------
            # PERCLOS
            # ------------------------------------------------

            perclos_result = self.perclos_estimator.update(
                timestamp_ms,
                eye_result.is_closed,
            )

            perclos_ready = bool(perclos_result.ready)

            if perclos_ready:
                perclos_value = float(perclos_result.perclos)

            # ------------------------------------------------
            # Mouth
            # ------------------------------------------------

            mouth_result = self.mouth_analyzer.analyze(
                face,
                width,
                height,
            )

            # ------------------------------------------------
            # Yawn
            # ------------------------------------------------

            yawn_result = self.yawn_detector.update(
                timestamp_ms,
                mouth_result.is_open,
            )

            current_yawn = bool(yawn_result.is_yawning)

            self.last_yawn_count = yawn_result.yawn_count

            # ------------------------------------------------
            # Driver state
            # ------------------------------------------------

            driver_state_result = self.driver_state_estimator.update(
                timestamp_ms=timestamp_ms,
                driver_presence=(presence_result.state),
                eye_closed=(eye_result.is_closed),
                perclos=(perclos_result.perclos),
                perclos_ready=(perclos_result.ready),
                yawn_count=(yawn_result.yawn_count),
            )

            self.last_driver_state_result = driver_state_result

        # ====================================================
        # Face temporarily unavailable
        # ====================================================

        else:
            # No face does not mean eyes are open.
            perclos_result = self.perclos_estimator.update(
                timestamp_ms,
                None,
            )

            perclos_ready = bool(perclos_result.ready)

            if perclos_ready:
                perclos_value = float(perclos_result.perclos)

            # ------------------------------------------------
            # Short detector dropout.
            # Keep last reliable Driver State.
            # ------------------------------------------------

            if presence_result.state == DriverPresence.PRESENT:
                driver_state_result = self.last_driver_state_result

            # ------------------------------------------------
            # Driver genuinely unavailable.
            # ------------------------------------------------

            else:
                driver_state_result = self.driver_state_estimator.update(
                    timestamp_ms=timestamp_ms,
                    driver_presence=(presence_result.state),
                    eye_closed=False,
                    perclos=(perclos_result.perclos),
                    perclos_ready=(perclos_result.ready),
                    yawn_count=(self.last_yawn_count),
                )

                self.last_driver_state_result = driver_state_result

        # ====================================================
        # Semantic result
        # ====================================================

        if driver_state_result is None:
            driver_state = "UNKNOWN"

            risk = "UNKNOWN"

            eye_closure_seconds = 0.0

            recent_yawns = 0

        else:
            driver_state = driver_state_result.state

            risk = driver_state_result.risk_level

            eye_closure_seconds = float(driver_state_result.continuous_eye_closure)

            recent_yawns = int(driver_state_result.recent_yawns)

        return CabinPerceptionSnapshot(
            metadata=self._observations.next(
                timestamp_ms=timestamp_ms,
                processing_ms=(time.perf_counter() - processing_started) * 1000,
            ),
            face_visible=face_visible,
            presence=(presence_result.state),
            driver_state=driver_state,
            risk=risk,
            perclos=perclos_value,
            perclos_ready=perclos_ready,
            eye_closed=eye_closed_now,
            eye_closure_seconds=(eye_closure_seconds),
            recent_yawns=recent_yawns,
            blink_count=(self.last_blink_count),
            current_yawn=(current_yawn),
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ) -> None:

        self.face_detector.close()
