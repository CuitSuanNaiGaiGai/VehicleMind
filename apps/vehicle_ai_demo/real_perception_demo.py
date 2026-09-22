from pathlib import Path

import cv2

from modules.cabin.perception_service import (
    CabinPerceptionService,
)

from modules.driving.perception_service import (
    DrivingPerceptionService,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def main():

    # ========================================================
    # Cabin
    # ========================================================

    cabin = (
        CabinPerceptionService(
            model_path=(
                PROJECT_ROOT
                / "models"
                / "mediapipe"
                / "face_landmarker.task"
            )
        )
    )

    # ========================================================
    # Driving
    # ========================================================

    driving = (
        DrivingPerceptionService(
            model_path=(
                PROJECT_ROOT
                / "models"
                / "driving"
                / "YOLOPv2_512.onnx"
            )
        )
    )

    # ========================================================
    # Camera
    # ========================================================

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION,
    )

    if not camera.isOpened():

        camera = (
            cv2.VideoCapture(
                0
            )
        )

    # ========================================================
    # Road video
    # ========================================================

    road = cv2.VideoCapture(
        str(
            PROJECT_ROOT
            / "assets"
            / "driving"
            / "road_test.mp4"
        )
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Cabin camera unavailable."
        )

    if not road.isOpened():

        raise RuntimeError(
            "Road video unavailable."
        )

    try:

        # ----------------------------------------------------
        # One cabin frame
        # ----------------------------------------------------

        ok, cabin_frame = (
            camera.read()
        )

        if not ok:

            raise RuntimeError(
                "Failed to read cabin frame."
            )

        cabin_frame = (
            cv2.flip(
                cabin_frame,
                1,
            )
        )

        cabin_result = (
            cabin.process_frame(
                cabin_frame
            )
        )

        print()
        print(
            "========== CABIN =========="
        )

        print(
            cabin_result
        )

        print()
        print(
            "Context kwargs:"
        )

        print(
            cabin_result
            .to_context_kwargs()
        )

        # ----------------------------------------------------
        # One road frame
        # ----------------------------------------------------

        ok, road_frame = (
            road.read()
        )

        if not ok:

            raise RuntimeError(
                "Failed to read road frame."
            )

        driving_result = (
            driving.process_frame(
                road_frame
            )
        )

        print()
        print(
            "========== DRIVING =========="
        )

        print(
            "Vehicles:",
            driving_result
            .vehicle_count,
        )

        print(
            "Pedestrians:",
            driving_result
            .pedestrian_count,
        )

        print(
            "Riders:",
            driving_result
            .rider_count,
        )

        print(
            "Traffic lights:",
            driving_result
            .traffic_light_count,
        )

        print(
            "Traffic signs:",
            driving_result
            .traffic_sign_count,
        )

        print(
            "Lane:",
            driving_result
            .lane_detected,
        )

        print(
            "Drivable:",
            driving_result
            .drivable_area_detected,
        )

        print(
            "Drivable ratio:",
            f"{driving_result.drivable_ratio:.3f}",
        )

        print()
        print(
            "Context kwargs:"
        )

        print(
            driving_result
            .to_context_kwargs()
        )

    finally:

        camera.release()

        road.release()

        cabin.close()

        driving.close()


if __name__ == "__main__":
    main()