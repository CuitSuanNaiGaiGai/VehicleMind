import time
from pathlib import Path

import cv2

from modules.cabin.face.landmarks import FaceLandmarkDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """
    Open MacBook camera.

    AVFoundation is preferred on macOS.
    """

    cap = cv2.VideoCapture(
        camera_id,
        cv2.CAP_AVFOUNDATION,
    )

    # Fallback
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open camera. "
            "Please check macOS camera permission."
        )

    # Request 1280x720.
    # Actual resolution depends on the camera.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


def draw_face_landmarks(
    frame,
    face_landmarks,
) -> None:
    """
    Draw facial landmarks on an OpenCV frame.
    """

    height, width = frame.shape[:2]

    for face in face_landmarks:
        for landmark in face:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Ignore invalid points outside the image.
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1,
                )


def main() -> None:
    detector = FaceLandmarkDetector(
        model_path=MODEL_PATH,
    )

    cap = open_camera(0)

    start_time = time.perf_counter()

    fps = 0.0
    frame_count = 0
    fps_start = time.perf_counter()

    print("Camera started.")
    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read frame.")
                break

            # MacBook front camera behaves like a mirror.
            frame = cv2.flip(frame, 1)

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            faces = detector.detect(
                frame,
                timestamp_ms,
            )

            draw_face_landmarks(
                frame,
                faces,
            )

            frame_count += 1

            elapsed = time.perf_counter() - fps_start

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()

            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "VehicleMind - Cabin Perception",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
