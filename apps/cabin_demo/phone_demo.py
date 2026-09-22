import time
from pathlib import Path

import cv2

from apps.cabin_demo.ui.phone import draw_interaction_zone, open_camera
from modules.cabin.face.landmarks import (
    FaceLandmarkDetector,
)

from modules.cabin.distraction.phone_detector import (
    PhoneDetector,
)

from modules.cabin.distraction.behavior_tracker import (
    PhoneBehaviorState,
    PhoneBehaviorTracker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


def main():
    face_detector = FaceLandmarkDetector(
        model_path=MODEL_PATH,
    )

    phone_detector = PhoneDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.35,
        image_size=640,
    )

    behavior_tracker = PhoneBehaviorTracker(
        near_duration_seconds=0.5,
        use_duration_seconds=1.5,
        missing_tolerance_seconds=0.25,
    )

    cap = open_camera()

    start_time = time.perf_counter()

    fps = 0.0

    fps_counter = 0

    fps_start = time.perf_counter()

    print("[VehicleMind] Phone behavior monitoring started.")

    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Mirror front camera.
            frame = cv2.flip(
                frame,
                1,
            )

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)

            height, width = frame.shape[:2]

            # =================================================
            # Face
            # =================================================

            faces = face_detector.detect(
                frame,
                timestamp_ms,
            )

            face = faces[0] if faces else None

            # =================================================
            # Phone Detection
            # =================================================

            phone_result = phone_detector.detect(frame)

            # =================================================
            # Behavior
            # =================================================

            behavior_result = behavior_tracker.update(
                timestamp_ms=timestamp_ms,
                phone_result=phone_result,
                face=face,
                frame_width=width,
                frame_height=height,
            )

            # =================================================
            # Interaction Zone
            # =================================================

            draw_interaction_zone(
                frame,
                behavior_result.interaction_zone,
            )

            # =================================================
            # Phone bounding boxes
            # =================================================

            for detection in phone_result.detections:
                is_associated = behavior_result.associated_phone is detection

                if is_associated:
                    color = (
                        0,
                        0,
                        255,
                    )

                else:
                    color = (
                        0,
                        200,
                        255,
                    )

                cv2.rectangle(
                    frame,
                    (
                        detection.x1,
                        detection.y1,
                    ),
                    (
                        detection.x2,
                        detection.y2,
                    ),
                    color,
                    2,
                )

                label = f"CELL PHONE {detection.confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (
                        detection.x1,
                        max(
                            20,
                            detection.y1 - 8,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # =================================================
            # Behavior State
            # =================================================

            state = behavior_result.state

            if state == PhoneBehaviorState.NO_PHONE:
                state_color = (
                    0,
                    255,
                    0,
                )

            elif state == PhoneBehaviorState.PHONE_PRESENT:
                state_color = (
                    0,
                    220,
                    255,
                )

            elif state == PhoneBehaviorState.PHONE_NEAR_DRIVER:
                state_color = (
                    0,
                    140,
                    255,
                )

            else:
                state_color = (
                    0,
                    0,
                    255,
                )

            cv2.putText(
                frame,
                state.value,
                (
                    20,
                    45,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                state_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (f"Associated Duration: {behavior_result.associated_duration:.2f}s"),
                (
                    20,
                    80,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (f"Phone Confidence: {behavior_result.confidence:.2f}"),
                (
                    20,
                    115,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # =================================================
            # FPS
            # =================================================

            fps_counter += 1

            elapsed = time.perf_counter() - fps_start

            if elapsed >= 1.0:
                fps = fps_counter / elapsed

                fps_counter = 0

                fps_start = time.perf_counter()

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (
                    20,
                    150,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # =================================================
            # Display
            # =================================================

            cv2.imshow(
                "VehicleMind - Phone Behavior",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()

        face_detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
