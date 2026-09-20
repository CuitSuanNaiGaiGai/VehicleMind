import argparse
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from modules.driving.perception.panoptic_detector import (
    PanopticDrivingDetector,
    DrivingObject,
    DrivingSceneResult,
)


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_WORK_WIDTH = 1280
DEFAULT_WORK_HEIGHT = 720


# ============================================================
# Object colors
# ============================================================

OBJECT_COLORS = {
    "person": (0, 80, 255),
    "rider": (255, 160, 0),

    "car": (80, 220, 80),
    "truck": (255, 120, 80),
    "bus": (0, 200, 255),
    "train": (220, 120, 255),

    "motorcycle": (255, 100, 255),
    "bicycle": (255, 180, 0),

    "traffic light": (0, 255, 255),
    "traffic sign": (0, 0, 255),
}


# ============================================================
# Text
# ============================================================

def draw_text(
    frame,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
    color=(225, 225, 225),
    thickness: int = 1,
):
    """
    Draw anti-aliased text.
    """

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


# ============================================================
# Object statistics
# ============================================================

def build_object_counts(
    scene_result: DrivingSceneResult,
) -> Dict[str, int]:
    """
    Count YOLOPv2 object classes.
    """

    counts: Dict[
        str,
        int,
    ] = {}

    for obj in (
        scene_result.objects
    ):

        class_name = (
            obj.class_name
        )

        counts[class_name] = (
            counts.get(
                class_name,
                0,
            )
            + 1
        )

    return counts


# ============================================================
# Drivable Area
# ============================================================

def draw_drivable_area(
    frame: np.ndarray,
    drivable_mask: np.ndarray,
    alpha: float = 0.22,
) -> None:
    """
    Draw drivable-area segmentation.

    Green translucent region = drivable road.
    """

    if drivable_mask is None:
        return

    mask = (
        drivable_mask > 0
    )

    if not np.any(
        mask
    ):
        return

    # --------------------------------------------------------
    # Only modify masked pixels.
    #
    # This avoids creating another full-resolution overlay
    # image and keeps visualization relatively lightweight.
    # --------------------------------------------------------

    original_pixels = (
        frame[
            mask
        ]
        .astype(
            np.float32
        )
    )

    target_color = np.array(
        [
            180,
            0,
            90,
        ],
        dtype=np.float32,
    )

    blended = (
        original_pixels
        * (
            1.0
            - alpha
        )
        +
        target_color
        * alpha
    )

    frame[
        mask
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# Lane mask
# ============================================================

def draw_lane_mask(
    frame: np.ndarray,
    lane_mask: np.ndarray,
) -> None:
    """
    Draw lane-line segmentation.

    Yellow = detected lane markings.
    """

    if lane_mask is None:
        return

    mask_uint8 = (
        (
            lane_mask > 0
        )
        .astype(
            np.uint8
        )
        * 255
    )

    if not np.any(
        mask_uint8
    ):
        return

    # --------------------------------------------------------
    # Slight dilation improves visibility in the video.
    # --------------------------------------------------------

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )
    )

    mask_uint8 = (
        cv2.dilate(
            mask_uint8,
            kernel,
            iterations=1,
        )
    )

    visible = (
        mask_uint8 > 0
    )

    frame[
        visible
    ] = (
        0,
        230,
        255,
    )


# ============================================================
# Object box
# ============================================================

def draw_object(
    frame: np.ndarray,
    obj: DrivingObject,
) -> None:
    """
    Draw one YOLOPv2 detection.
    """

    height, width = (
        frame.shape[:2]
    )

    x1 = int(
        np.clip(
            obj.x1,
            0,
            width - 1,
        )
    )

    y1 = int(
        np.clip(
            obj.y1,
            0,
            height - 1,
        )
    )

    x2 = int(
        np.clip(
            obj.x2,
            0,
            width - 1,
        )
    )

    y2 = int(
        np.clip(
            obj.y2,
            0,
            height - 1,
        )
    )

    color = (
        OBJECT_COLORS.get(
            obj.class_name,
            (
                220,
                220,
                220,
            ),
        )
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        color,
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        f"{obj.class_name.upper()} "
        f"{obj.confidence:.2f}"
    )

    (
        text_width,
        text_height,
    ), baseline = (
        cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            1,
        )
    )

    label_top = max(
        0,
        y1
        - text_height
        - baseline
        - 7,
    )

    label_bottom = (
        y1
    )

    label_right = min(
        width - 1,
        x1
        + text_width
        + 10,
    )

    cv2.rectangle(
        frame,
        (
            x1,
            label_top,
        ),
        (
            label_right,
            label_bottom,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 5,
            max(
                text_height + 2,
                y1 - 5,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Dashboard
# ============================================================

def draw_dashboard(
    frame: np.ndarray,
    scene_result: DrivingSceneResult,
    pipeline_fps: float,
    source_fps: float,
) -> None:
    """
    Draw compact VehicleMind Driving Perception dashboard.
    """

    height, width = (
        frame.shape[:2]
    )

    panel_width = min(
        335,
        width // 3,
    )

    x1 = (
        width
        - panel_width
    )

    # ========================================================
    # Background
    # ========================================================

    roi = (
        frame[
            :,
            x1:width,
        ]
    )

    dark = np.full_like(
        roi,
        (
            18,
            18,
            18,
        ),
    )

    cv2.addWeighted(
        dark,
        0.78,
        roi,
        0.22,
        0,
        dst=roi,
    )

    left = (
        x1 + 22
    )

    value_x = (
        x1 + 205
    )

    right = (
        width - 20
    )

    y = 42

    # ========================================================
    # Header
    # ========================================================

    draw_text(
        frame,
        "VEHICLEMIND",
        left,
        y,
        scale=0.76,
        color=(255, 255, 255),
        thickness=2,
    )

    y += 29

    draw_text(
        frame,
        "DRIVING PERCEPTION",
        left,
        y,
        scale=0.46,
        color=(165, 165, 165),
    )

    y += 27

    cv2.line(
        frame,
        (
            left,
            y,
        ),
        (
            right,
            y,
        ),
        (
            85,
            85,
            85,
        ),
        1,
    )

    y += 32

    # ========================================================
    # Object perception
    # ========================================================

    draw_text(
        frame,
        "OBJECT PERCEPTION",
        left,
        y,
        scale=0.48,
        color=(165, 165, 165),
        thickness=2,
    )

    y += 32

    counts = (
        build_object_counts(
            scene_result
        )
    )

    vehicles = (
        counts.get(
            "car",
            0,
        )
        +
        counts.get(
            "truck",
            0,
        )
        +
        counts.get(
            "bus",
            0,
        )
        +
        counts.get(
            "train",
            0,
        )
    )

    vulnerable_users = (
        counts.get(
            "person",
            0,
        )
        +
        counts.get(
            "rider",
            0,
        )
        +
        counts.get(
            "motorcycle",
            0,
        )
        +
        counts.get(
            "bicycle",
            0,
        )
    )

    traffic_lights = (
        counts.get(
            "traffic light",
            0,
        )
    )

    traffic_signs = (
        counts.get(
            "traffic sign",
            0,
        )
    )

    total_objects = len(
        scene_result.objects
    )

    rows = (
        (
            "Vehicles",
            vehicles,
        ),
        (
            "Road Users",
            vulnerable_users,
        ),
        (
            "Traffic Lights",
            traffic_lights,
        ),
        (
            "Traffic Signs",
            traffic_signs,
        ),
        (
            "Total",
            total_objects,
        ),
    )

    for label, value in rows:

        draw_text(
            frame,
            label,
            left,
            y,
            scale=0.49,
        )

        draw_text(
            frame,
            str(value),
            value_x,
            y,
            scale=0.54,
            color=(255, 255, 255),
            thickness=2,
        )

        y += 27

    y += 10

    # ========================================================
    # Road perception
    # ========================================================

    draw_text(
        frame,
        "ROAD PERCEPTION",
        left,
        y,
        scale=0.48,
        color=(165, 165, 165),
        thickness=2,
    )

    y += 32

    drivable_pixels = int(
        np.count_nonzero(
            scene_result
            .drivable_mask
        )
    )

    lane_pixels = int(
        np.count_nonzero(
            scene_result
            .lane_mask
        )
    )

    total_pixels = int(
        scene_result
        .drivable_mask
        .size
    )

    drivable_ratio = (
        drivable_pixels
        /
        max(
            total_pixels,
            1,
        )
    )

    drivable_detected = (
        drivable_pixels
        > 1000
    )

    lane_detected = (
        lane_pixels
        > 300
    )

    # --------------------------------------------------------
    # Drivable Area
    # --------------------------------------------------------

    draw_text(
        frame,
        "Drivable",
        left,
        y,
        scale=0.49,
    )

    draw_text(
        frame,
        (
            "DETECTED"
            if drivable_detected
            else "SEARCH"
        ),
        value_x - 25,
        y,
        scale=0.46,
        color=(
            (
                80,
                220,
                80,
            )
            if drivable_detected
            else (
                0,
                180,
                255,
            )
        ),
        thickness=2,
    )

    y += 28

    draw_text(
        frame,
        "Coverage",
        left,
        y,
        scale=0.49,
    )

    draw_text(
        frame,
        (
            f"{drivable_ratio * 100:.1f}%"
        ),
        value_x,
        y,
        scale=0.51,
        color=(80, 220, 80),
        thickness=2,
    )

    y += 28

    # --------------------------------------------------------
    # Lane
    # --------------------------------------------------------

    draw_text(
        frame,
        "Lane",
        left,
        y,
        scale=0.49,
    )

    draw_text(
        frame,
        (
            "DETECTED"
            if lane_detected
            else "SEARCH"
        ),
        value_x - 25,
        y,
        scale=0.46,
        color=(
            (
                0,
                230,
                255,
            )
            if lane_detected
            else (
                0,
                180,
                255,
            )
        ),
        thickness=2,
    )

    y += 37

    cv2.line(
        frame,
        (
            left,
            y,
        ),
        (
            right,
            y,
        ),
        (
            85,
            85,
            85,
        ),
        1,
    )

    y += 31

    # ========================================================
    # Runtime
    # ========================================================

    draw_text(
        frame,
        "RUNTIME",
        left,
        y,
        scale=0.48,
        color=(165, 165, 165),
        thickness=2,
    )

    y += 31

    runtime_rows = (
        (
            "Preprocess",
            (
                f"{scene_result.preprocess_ms:.1f} ms"
            ),
        ),
        (
            "Inference",
            (
                f"{scene_result.inference_ms:.1f} ms"
            ),
        ),
        (
            "Postprocess",
            (
                f"{scene_result.postprocess_ms:.1f} ms"
            ),
        ),
        (
            "Detector",
            (
                f"{scene_result.total_ms:.1f} ms"
            ),
        ),
    )

    for label, value in (
        runtime_rows
    ):

        draw_text(
            frame,
            label,
            left,
            y,
            scale=0.46,
        )

        draw_text(
            frame,
            value,
            value_x - 25,
            y,
            scale=0.46,
            color=(220, 220, 220),
        )

        y += 25

    y += 5

    # ========================================================
    # Realtime status
    # ========================================================

    realtime = (
        pipeline_fps
        >= source_fps
        * 0.95
    )

    draw_text(
        frame,
        "Pipeline",
        left,
        y,
        scale=0.49,
    )

    draw_text(
        frame,
        (
            f"{pipeline_fps:.1f} FPS"
        ),
        value_x - 15,
        y,
        scale=0.51,
        color=(
            (
                80,
                220,
                80,
            )
            if realtime
            else (
                0,
                165,
                255,
            )
        ),
        thickness=2,
    )

    y += 27

    draw_text(
        frame,
        "Target",
        left,
        y,
        scale=0.49,
    )

    draw_text(
        frame,
        (
            f"{source_fps:.1f} FPS"
        ),
        value_x - 15,
        y,
        scale=0.51,
        color=(220, 220, 220),
    )

    y += 27

    draw_text(
        frame,
        "Realtime",
        left,
        y,
        scale=0.49,
    )

    draw_text(
        frame,
        (
            "READY"
            if realtime
            else "OPTIMIZING"
        ),
        value_x - 25,
        y,
        scale=0.45,
        color=(
            (
                80,
                220,
                80,
            )
            if realtime
            else (
                0,
                165,
                255,
            )
        ),
        thickness=2,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = (
        argparse.ArgumentParser(
            description=(
                "VehicleMind Integrated "
                "Driving Perception Demo"
            )
        )
    )

    # ========================================================
    # Arguments
    # ========================================================

    parser.add_argument(
        "--video",
        type=str,
        default=(
            "assets/driving/"
            "road_test.mp4"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "assets/driving/"
            "scene_result.mp4"
        ),
    )

    parser.add_argument(
        "--panoptic-model",
        type=str,
        default=(
            "models/driving/"
            "YOLOPv2_512.onnx"
        ),
    )

    parser.add_argument(
        "--work-width",
        type=int,
        default=(
            DEFAULT_WORK_WIDTH
        ),
    )

    parser.add_argument(
        "--work-height",
        type=int,
        default=(
            DEFAULT_WORK_HEIGHT
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # Paths
    # ========================================================

    input_path = (
        PROJECT_ROOT
        / args.video
    )

    output_path = (
        PROJECT_ROOT
        / args.output
    )

    panoptic_model_path = (
        PROJECT_ROOT
        / args.panoptic_model
    )

    if not input_path.exists():

        raise FileNotFoundError(
            "Video not found: "
            f"{input_path}"
        )

    if not panoptic_model_path.exists():

        raise FileNotFoundError(
            "YOLOPv2 model not found: "
            f"{panoptic_model_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Header
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        " VehicleMind Driving Perception"
    )

    print(
        "========================================"
    )

    print()

    # ========================================================
    # Panoptic detector
    #
    # One YOLOPv2 only.
    #
    # No YOLO11.
    # ========================================================

    detector = (
        PanopticDrivingDetector(
            model_path=(
                panoptic_model_path
            ),
            score_threshold=(
                args.conf
            ),
            nms_threshold=(
                args.nms
            ),
            prefer_coreml=True,
            warmup_runs=2,
        )
    )

    # ========================================================
    # Input video
    # ========================================================

    cap = cv2.VideoCapture(
        str(
            input_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Failed to open video: "
            f"{input_path}"
        )

    source_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    source_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if source_fps <= 0:

        source_fps = 30.0

    work_width = int(
        args.work_width
    )

    work_height = int(
        args.work_height
    )

    print(
        "[VehicleMind] "
        f"Input: {input_path}"
    )

    print(
        "[VehicleMind] "
        f"Source resolution: "
        f"{source_width}x"
        f"{source_height}"
    )

    print(
        "[VehicleMind] "
        f"Working resolution: "
        f"{work_width}x"
        f"{work_height}"
    )

    print(
        "[VehicleMind] "
        f"Source FPS: "
        f"{source_fps:.2f}"
    )

    print(
        "[VehicleMind] "
        f"Frame budget: "
        f"{1000.0 / source_fps:.1f} ms"
    )

    print(
        "[VehicleMind] "
        f"Frames: "
        f"{total_frames}"
    )

    # ========================================================
    # Output writer
    #
    # IMPORTANT:
    #
    # Output is 1280x720 but retains original FPS.
    #
    # Therefore scene_result.mp4 plays at exactly the same
    # temporal speed as road_test.mp4.
    # ========================================================

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = (
        cv2.VideoWriter(
            str(
                output_path
            ),
            fourcc,
            source_fps,
            (
                work_width,
                work_height,
            ),
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Failed to create output video: "
            f"{output_path}"
        )

    # ========================================================
    # Runtime state
    # ========================================================

    frame_index = 0

    total_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # EMA pipeline FPS.
    #
    # More stable than calculating from a 1-second block.
    # --------------------------------------------------------

    pipeline_fps = 0.0

    fps_alpha = 0.90

    # --------------------------------------------------------
    # Timing accumulation
    # --------------------------------------------------------

    sum_detector_ms = 0.0

    sum_inference_ms = 0.0

    sum_preprocess_ms = 0.0

    sum_postprocess_ms = 0.0

    sum_pipeline_ms = 0.0

    # ========================================================
    # Processing loop
    # ========================================================

    try:

        while True:

            loop_start = (
                time.perf_counter()
            )

            # =================================================
            # Read source frame
            # =================================================

            success, source_frame = (
                cap.read()
            )

            if not success:
                break

            frame_index += 1

            # =================================================
            # Resize source once
            #
            # 2562×1440
            #      ↓
            # 1280×720
            #
            # All following perception / visualization works
            # on this resolution.
            # =================================================

            working_frame = (
                cv2.resize(
                    source_frame,
                    (
                        work_width,
                        work_height,
                    ),
                    interpolation=(
                        cv2.INTER_AREA
                    ),
                )
            )

            # =================================================
            # YOLOPv2
            #
            # Single model:
            #
            # Objects
            # +
            # Drivable Area
            # +
            # Lane Markings
            # =================================================

            scene_result = (
                detector.detect(
                    working_frame
                )
            )

            # =================================================
            # Visualization
            # =================================================

            draw_drivable_area(
                working_frame,
                scene_result
                .drivable_mask,
            )

            draw_lane_mask(
                working_frame,
                scene_result
                .lane_mask,
            )

            for obj in (
                scene_result.objects
            ):

                draw_object(
                    working_frame,
                    obj,
                )

            # =================================================
            # Current pipeline timing
            # =================================================

            # Temporarily estimate pipeline time before
            # dashboard / writer so the displayed FPS remains
            # responsive.

            elapsed_so_far = (
                time.perf_counter()
                - loop_start
            )

            instantaneous_fps = (
                1.0
                /
                max(
                    elapsed_so_far,
                    1e-6,
                )
            )

            if pipeline_fps <= 0:

                pipeline_fps = (
                    instantaneous_fps
                )

            else:

                pipeline_fps = (
                    fps_alpha
                    * pipeline_fps
                    +
                    (
                        1.0
                        - fps_alpha
                    )
                    * instantaneous_fps
                )

            # =================================================
            # Dashboard
            # =================================================

            draw_dashboard(
                frame=(
                    working_frame
                ),
                scene_result=(
                    scene_result
                ),
                pipeline_fps=(
                    pipeline_fps
                ),
                source_fps=(
                    source_fps
                ),
            )

            # =================================================
            # Write
            # =================================================

            writer.write(
                working_frame
            )

            # =================================================
            # Preview
            # =================================================

            if args.show:

                cv2.imshow(
                    (
                        "VehicleMind - "
                        "Driving Perception"
                    ),
                    working_frame,
                )

                # ------------------------------------------------
                # Do NOT artificially wait 33 ms here.
                #
                # We are benchmarking the actual algorithm.
                #
                # If pipeline becomes faster than the source
                # frame rate, a later real-time playback layer
                # can pace frames separately.
                # ------------------------------------------------

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if (
                    key
                    == ord("q")
                ):
                    break

            # =================================================
            # Complete timing
            # =================================================

            loop_ms = (
                (
                    time.perf_counter()
                    - loop_start
                )
                * 1000.0
            )

            true_frame_fps = (
                1000.0
                /
                max(
                    loop_ms,
                    1e-6,
                )
            )

            pipeline_fps = (
                fps_alpha
                * pipeline_fps
                +
                (
                    1.0
                    - fps_alpha
                )
                * true_frame_fps
            )

            # -------------------------------------------------
            # Accumulate performance metrics
            # -------------------------------------------------

            sum_preprocess_ms += (
                scene_result
                .preprocess_ms
            )

            sum_inference_ms += (
                scene_result
                .inference_ms
            )

            sum_postprocess_ms += (
                scene_result
                .postprocess_ms
            )

            sum_detector_ms += (
                scene_result
                .total_ms
            )

            sum_pipeline_ms += (
                loop_ms
            )

            # =================================================
            # Progress
            # =================================================

            if (
                frame_index
                % 50
                == 0
            ):

                if total_frames > 0:

                    progress = (
                        frame_index
                        /
                        total_frames
                        * 100.0
                    )

                else:

                    progress = 0.0

                average_pipeline_ms = (
                    sum_pipeline_ms
                    /
                    frame_index
                )

                average_pipeline_fps = (
                    1000.0
                    /
                    max(
                        average_pipeline_ms,
                        1e-6,
                    )
                )

                print(
                    "[VehicleMind] "
                    f"{frame_index}/"
                    f"{total_frames} "
                    f"({progress:.1f}%) "
                    f"| Pipeline "
                    f"{average_pipeline_fps:.1f} FPS "
                    f"| Inference "
                    f"{scene_result.inference_ms:.1f} ms "
                    f"| Detector "
                    f"{scene_result.total_ms:.1f} ms"
                )

    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    # ========================================================
    # Summary
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    if frame_index > 0:

        average_preprocess_ms = (
            sum_preprocess_ms
            /
            frame_index
        )

        average_inference_ms = (
            sum_inference_ms
            /
            frame_index
        )

        average_postprocess_ms = (
            sum_postprocess_ms
            /
            frame_index
        )

        average_detector_ms = (
            sum_detector_ms
            /
            frame_index
        )

        average_pipeline_ms = (
            sum_pipeline_ms
            /
            frame_index
        )

        average_pipeline_fps = (
            1000.0
            /
            max(
                average_pipeline_ms,
                1e-6,
            )
        )

    else:

        average_preprocess_ms = 0.0
        average_inference_ms = 0.0
        average_postprocess_ms = 0.0
        average_detector_ms = 0.0
        average_pipeline_ms = 0.0
        average_pipeline_fps = 0.0

    realtime_target_ms = (
        1000.0
        /
        source_fps
    )

    realtime_ready = (
        average_pipeline_ms
        <= realtime_target_ms
    )

    print()

    print(
        "========================================"
    )

    print(
        " VehicleMind Performance Summary"
    )

    print(
        "========================================"
    )

    print(
        f"Processed frames : "
        f"{frame_index}"
    )

    print(
        f"Source FPS       : "
        f"{source_fps:.2f}"
    )

    print(
        f"Frame budget     : "
        f"{realtime_target_ms:.2f} ms"
    )

    print()

    print(
        f"Preprocess       : "
        f"{average_preprocess_ms:.2f} ms"
    )

    print(
        f"Inference        : "
        f"{average_inference_ms:.2f} ms"
    )

    print(
        f"Postprocess      : "
        f"{average_postprocess_ms:.2f} ms"
    )

    print(
        f"Detector total   : "
        f"{average_detector_ms:.2f} ms"
    )

    print(
        f"Whole pipeline   : "
        f"{average_pipeline_ms:.2f} ms"
    )

    print(
        f"Pipeline FPS     : "
        f"{average_pipeline_fps:.2f}"
    )

    print()

    print(
        "Realtime status  : "
        + (
            "READY"
            if realtime_ready
            else "NOT YET"
        )
    )

    print(
        f"Wall-clock time  : "
        f"{total_elapsed:.2f} s"
    )

    print(
        f"Output           : "
        f"{output_path}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()