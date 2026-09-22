from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarkDetector:
    """MediaPipe-based facial landmark detector."""

    def __init__(
        self,
        model_path: str | Path,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> List:
        """
        Detect facial landmarks from one BGR frame.

        Args:
            frame_bgr:
                OpenCV BGR image.

            timestamp_ms:
                Monotonically increasing frame timestamp.

        Returns:
            List of detected faces.
            Each face contains normalized facial landmarks.
        """

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        frame_rgb = np.ascontiguousarray(frame_rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = self._detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        return result.face_landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()
