from pathlib import Path

import cv2

from modules.driving.perception.panoptic_detector import (
    PanopticDrivingDetector,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def main():

    model_path = (
        PROJECT_ROOT
        / "models"
        / "driving"
        / "YOLOPv2.onnx"
    )

    video_path = (
        PROJECT_ROOT
        / "assets"
        / "driving"
        / "road_test.mp4"
    )

    detector = (
        PanopticDrivingDetector(
            model_path=model_path,
        )
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    success, frame = (
        cap.read()
    )

    cap.release()

    if not success:
        raise RuntimeError(
            "Failed to read road_test.mp4"
        )

    result = detector.detect(
        frame
    )

    print()
    print(
        "========== VehicleMind =========="
    )

    print(
        f"Objects: "
        f"{len(result.objects)}"
    )

    print(
        f"Drivable mask: "
        f"{result.drivable_mask.shape}"
    )

    print(
        f"Lane mask: "
        f"{result.lane_mask.shape}"
    )

    print(
        f"Inference: "
        f"{result.inference_ms:.1f} ms"
    )

    print(
        "Drivable pixels:",
        int(
            (
                result.drivable_mask > 0
            ).sum()
        ),
    )

    print(
        "Lane pixels:",
        int(
            (
                result.lane_mask > 0
            ).sum()
        ),
    )


if __name__ == "__main__":
    main()
