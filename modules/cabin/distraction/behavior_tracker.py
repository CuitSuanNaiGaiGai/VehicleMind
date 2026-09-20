from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from modules.cabin.distraction.phone_detector import (
    PhoneDetection,
    PhoneDetectionResult,
)


class PhoneBehaviorState(str, Enum):

    NO_PHONE = "NO_PHONE"

    PHONE_PRESENT = "PHONE_PRESENT"

    PHONE_NEAR_DRIVER = "PHONE_NEAR_DRIVER"

    PHONE_USE = "PHONE_USE"


@dataclass
class DriverInteractionZone:

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class PhoneBehaviorResult:

    state: PhoneBehaviorState

    phone_detected: bool

    phone_near_driver: bool

    associated_duration: float

    confidence: float

    interaction_zone: Optional[
        DriverInteractionZone
    ]

    associated_phone: Optional[
        PhoneDetection
    ]


class PhoneBehaviorTracker:
    """
    Convert frame-level cell-phone detections into
    driver phone-use behavior.

    Pipeline:

        phone detection
            ↓
        driver interaction zone
            ↓
        spatial association
            ↓
        temporal persistence
            ↓
        PHONE_USE

    Important:

        PHONE_PRESENT does NOT mean the driver
        is distracted.

        PHONE_USE is currently a visual proxy
        based on spatial association + duration.

    Later versions may add:
        - hand landmarks
        - wrist-phone association
        - gaze
        - head pose
    """

    def __init__(
        self,
        near_duration_seconds: float = 0.5,
        use_duration_seconds: float = 1.5,
        missing_tolerance_seconds: float = 0.25,
    ):

        self.near_duration_ms = int(
            near_duration_seconds * 1000
        )

        self.use_duration_ms = int(
            use_duration_seconds * 1000
        )

        self.missing_tolerance_ms = int(
            missing_tolerance_seconds * 1000
        )

        self._associated_since_ms = None

        self._last_associated_ms = None

    # =========================================================
    # Face bounding box
    # =========================================================

    @staticmethod
    def _face_bbox(
        face,
        frame_width: int,
        frame_height: int,
    ):

        xs = []

        ys = []

        for landmark in face:

            xs.append(
                landmark.x
                * frame_width
            )

            ys.append(
                landmark.y
                * frame_height
            )

        x1 = int(
            max(
                0,
                min(xs),
            )
        )

        y1 = int(
            max(
                0,
                min(ys),
            )
        )

        x2 = int(
            min(
                frame_width - 1,
                max(xs),
            )
        )

        y2 = int(
            min(
                frame_height - 1,
                max(ys),
            )
        )

        return (
            x1,
            y1,
            x2,
            y2,
        )

    # =========================================================
    # Driver interaction zone
    # =========================================================

    def build_interaction_zone(
        self,
        face,
        frame_width: int,
        frame_height: int,
    ) -> DriverInteractionZone:

        (
            face_x1,
            face_y1,
            face_x2,
            face_y2,
        ) = self._face_bbox(
            face,
            frame_width,
            frame_height,
        )

        face_width = max(
            1,
            face_x2 - face_x1,
        )

        face_height = max(
            1,
            face_y2 - face_y1,
        )

        face_center_x = (
            face_x1
            + face_x2
        ) / 2.0

        # -----------------------------------------------------
        # Build an approximate driver upper-body interaction
        # zone around the face.
        #
        # Horizontal:
        #     ~2.5 face widths
        #
        # Vertical:
        #     starts slightly above the face and extends
        #     downward toward chest/lap region.
        # -----------------------------------------------------

        zone_width = (
            face_width
            * 2.6
        )

        zone_x1 = int(
            max(
                0,
                face_center_x
                - zone_width / 2.0,
            )
        )

        zone_x2 = int(
            min(
                frame_width - 1,
                face_center_x
                + zone_width / 2.0,
            )
        )

        zone_y1 = int(
            max(
                0,
                face_y1
                - 0.25 * face_height,
            )
        )

        zone_y2 = int(
            min(
                frame_height - 1,
                face_y2
                + 3.0 * face_height,
            )
        )

        return DriverInteractionZone(
            x1=zone_x1,
            y1=zone_y1,
            x2=zone_x2,
            y2=zone_y2,
        )

    # =========================================================
    # Spatial association
    # =========================================================

    @staticmethod
    def _phone_center(
        phone: PhoneDetection,
    ):

        center_x = (
            phone.x1
            + phone.x2
        ) / 2.0

        center_y = (
            phone.y1
            + phone.y2
        ) / 2.0

        return (
            center_x,
            center_y,
        )

    @staticmethod
    def _inside_zone(
        phone: PhoneDetection,
        zone: DriverInteractionZone,
    ) -> bool:

        (
            center_x,
            center_y,
        ) = PhoneBehaviorTracker._phone_center(
            phone
        )

        return (
            zone.x1
            <= center_x
            <= zone.x2
            and
            zone.y1
            <= center_y
            <= zone.y2
        )

    def _find_associated_phone(
        self,
        detections: PhoneDetectionResult,
        zone: DriverInteractionZone,
    ) -> Optional[PhoneDetection]:

        candidates = []

        for phone in detections.detections:

            if self._inside_zone(
                phone,
                zone,
            ):

                candidates.append(
                    phone
                )

        if not candidates:

            return None

        return max(
            candidates,
            key=lambda item: (
                item.confidence
            ),
        )

    # =========================================================
    # Temporal behavior
    # =========================================================

    def update(
        self,
        timestamp_ms: int,
        phone_result: PhoneDetectionResult,
        face,
        frame_width: int,
        frame_height: int,
    ) -> PhoneBehaviorResult:

        # -----------------------------------------------------
        # No face:
        #
        # We cannot reliably associate the phone with
        # the driver.
        # -----------------------------------------------------

        if face is None:

            self._reset_if_too_long_missing(
                timestamp_ms
            )

            return PhoneBehaviorResult(
                state=(
                    PhoneBehaviorState.PHONE_PRESENT
                    if phone_result.detected
                    else PhoneBehaviorState.NO_PHONE
                ),
                phone_detected=phone_result.detected,
                phone_near_driver=False,
                associated_duration=0.0,
                confidence=(
                    phone_result.best_detection.confidence
                    if phone_result.best_detection
                    is not None
                    else 0.0
                ),
                interaction_zone=None,
                associated_phone=None,
            )

        # -----------------------------------------------------
        # Driver zone
        # -----------------------------------------------------

        zone = self.build_interaction_zone(
            face,
            frame_width,
            frame_height,
        )

        # -----------------------------------------------------
        # No phone
        # -----------------------------------------------------

        if not phone_result.detected:

            self._reset_if_too_long_missing(
                timestamp_ms
            )

            return PhoneBehaviorResult(
                state=PhoneBehaviorState.NO_PHONE,
                phone_detected=False,
                phone_near_driver=False,
                associated_duration=0.0,
                confidence=0.0,
                interaction_zone=zone,
                associated_phone=None,
            )

        # -----------------------------------------------------
        # Phone exists.
        #
        # Determine whether it belongs to the driver's
        # interaction region.
        # -----------------------------------------------------

        associated_phone = (
            self._find_associated_phone(
                phone_result,
                zone,
            )
        )

        if associated_phone is None:

            self._reset_if_too_long_missing(
                timestamp_ms
            )

            return PhoneBehaviorResult(
                state=PhoneBehaviorState.PHONE_PRESENT,
                phone_detected=True,
                phone_near_driver=False,
                associated_duration=0.0,
                confidence=(
                    phone_result.best_detection.confidence
                    if phone_result.best_detection
                    is not None
                    else 0.0
                ),
                interaction_zone=zone,
                associated_phone=None,
            )

        # -----------------------------------------------------
        # Associated with driver
        # -----------------------------------------------------

        self._last_associated_ms = (
            timestamp_ms
        )

        if self._associated_since_ms is None:

            self._associated_since_ms = (
                timestamp_ms
            )

        associated_ms = (
            timestamp_ms
            - self._associated_since_ms
        )

        associated_duration = (
            associated_ms
            / 1000.0
        )

        # -----------------------------------------------------
        # Behavior states
        # -----------------------------------------------------

        if (
            associated_ms
            >= self.use_duration_ms
        ):

            state = (
                PhoneBehaviorState.PHONE_USE
            )

        elif (
            associated_ms
            >= self.near_duration_ms
        ):

            state = (
                PhoneBehaviorState.PHONE_NEAR_DRIVER
            )

        else:

            state = (
                PhoneBehaviorState.PHONE_PRESENT
            )

        return PhoneBehaviorResult(
            state=state,
            phone_detected=True,
            phone_near_driver=True,
            associated_duration=(
                associated_duration
            ),
            confidence=(
                associated_phone.confidence
            ),
            interaction_zone=zone,
            associated_phone=associated_phone,
        )

    # =========================================================
    # Missing-detection tolerance
    # =========================================================

    def _reset_if_too_long_missing(
        self,
        timestamp_ms: int,
    ) -> None:

        if (
            self._last_associated_ms
            is None
        ):

            self._associated_since_ms = None

            return

        missing_ms = (
            timestamp_ms
            - self._last_associated_ms
        )

        if (
            missing_ms
            > self.missing_tolerance_ms
        ):

            self._associated_since_ms = None

            self._last_associated_ms = None

    def reset(self):

        self._associated_since_ms = None

        self._last_associated_ms = None
