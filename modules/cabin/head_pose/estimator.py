from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HeadPoseResult:
    """
    Head orientation in degrees.

    yaw:
        horizontal head rotation

    pitch:
        vertical head rotation

    roll:
        in-plane head tilt

    Note:
        Sign direction depends on image mirroring
        and camera coordinate convention.
    """

    yaw: float
    pitch: float
    roll: float

    success: bool


class HeadPoseEstimator:
    """
    Estimate head orientation from MediaPipe Face Landmarks
    using OpenCV solvePnP.

    This module only estimates geometric head pose.
    Temporal driver-distraction reasoning should be handled
    by a separate module.
    """

    # ========================================================
    # MediaPipe landmark indices
    # ========================================================

    NOSE_TIP = 1
    CHIN = 152

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    LANDMARK_INDICES = (
        NOSE_TIP,
        CHIN,
        LEFT_EYE_OUTER,
        RIGHT_EYE_OUTER,
        LEFT_MOUTH,
        RIGHT_MOUTH,
    )

    def __init__(self):

        # -----------------------------------------------------
        # Generic 3D face model.
        #
        # Absolute dimensions are not important here.
        # Relative geometry determines the rotation estimate.
        # -----------------------------------------------------

        self.model_points = np.array(
            [
                # Nose
                (0.0, 0.0, 0.0),

                # Chin
                (0.0, -63.6, -12.5),

                # Left eye outer corner
                (-43.3, 32.7, -26.0),

                # Right eye outer corner
                (43.3, 32.7, -26.0),

                # Left mouth corner
                (-28.9, -28.9, -24.1),

                # Right mouth corner
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float64,
        )

    @classmethod
    def landmark_indices(cls):
        return cls.LANDMARK_INDICES

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:
        """
        Normalize angle into [-180, 180].
        """

        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle

    @staticmethod
    def _normalize_pitch(
        pitch: float,
    ) -> float:
        """
        OpenCV Euler decomposition may represent a nearly
        frontal pose as approximately +/-180 degrees because
        of coordinate-system ambiguity.

        Convert pitch into the physically useful range
        [-90, 90].

        Example:
            179.4  -> -0.6
           -178.0  ->  2.0
        """

        pitch = (
            HeadPoseEstimator
            ._normalize_angle(pitch)
        )

        if pitch > 90.0:
            pitch -= 180.0

        elif pitch < -90.0:
            pitch += 180.0

        return pitch

    def estimate(
        self,
        face,
        frame_width: int,
        frame_height: int,
    ) -> HeadPoseResult:

        # =====================================================
        # 1. MediaPipe normalized coordinates -> image pixels
        # =====================================================

        image_points = []

        for index in self.LANDMARK_INDICES:

            landmark = face[index]

            x = (
                landmark.x
                * frame_width
            )

            y = (
                landmark.y
                * frame_height
            )

            image_points.append(
                (x, y)
            )

        image_points = np.asarray(
            image_points,
            dtype=np.float64,
        )

        # =====================================================
        # 2. Approximate camera intrinsics
        # =====================================================

        focal_length = float(
            frame_width
        )

        center_x = (
            frame_width
            / 2.0
        )

        center_y = (
            frame_height
            / 2.0
        )

        camera_matrix = np.array(
            [
                [
                    focal_length,
                    0.0,
                    center_x,
                ],
                [
                    0.0,
                    focal_length,
                    center_y,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        distortion = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        # =====================================================
        # 3. solvePnP
        # =====================================================

        success, rotation_vector, _ = (
            cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

        if not success:

            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                success=False,
            )

        # =====================================================
        # 4. Rotation vector -> matrix
        # =====================================================

        rotation_matrix, _ = (
            cv2.Rodrigues(
                rotation_vector
            )
        )

        # =====================================================
        # 5. Rotation matrix -> Euler angles
        # =====================================================

        projection_matrix = np.hstack(
            (
                rotation_matrix,
                np.zeros(
                    (3, 1),
                    dtype=np.float64,
                ),
            )
        )

        (
            _,
            _,
            _,
            _,
            _,
            _,
            euler_angles,
        ) = cv2.decomposeProjectionMatrix(
            projection_matrix
        )

        raw_pitch = float(
            euler_angles[0, 0]
        )

        raw_yaw = float(
            euler_angles[1, 0]
        )

        raw_roll = float(
            euler_angles[2, 0]
        )

        # =====================================================
        # 6. Normalize Euler representation
        # =====================================================

        pitch = self._normalize_pitch(
            raw_pitch
        )

        yaw = self._normalize_angle(
            raw_yaw
        )

        roll = self._normalize_angle(
            raw_roll
        )

        return HeadPoseResult(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            success=True,
        )