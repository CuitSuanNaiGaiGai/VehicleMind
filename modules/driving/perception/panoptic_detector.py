from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from modules.driving.perception import yolopv2_utils


# ============================================================
# COCO class names
#
# The YOLOPv2 ONNX sample detection head uses 85 values:
#
#   4 bbox
# + 1 objectness
# + 80 classes
#
# therefore the detection branch follows COCO-80 indexing.
# ============================================================

BDD100K_CLASSES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

# ============================================================
# Data structures
# ============================================================

@dataclass
class DrivingObject:
    """
    One detected traffic / road object.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    confidence: float

    class_id: int
    class_name: str


@dataclass
class DrivingSceneResult:
    """
    Unified VehicleMind driving-perception output.

    objects:
        Object detections.

    drivable_mask:
        Binary drivable-area mask.
        Shape == original video resolution.

    lane_mask:
        Binary lane-marking mask.
        Shape == original video resolution.

    inference_ms:
        ONNX model inference latency only.

    preprocess_ms:
        Resize / letterbox / tensor conversion latency.

    postprocess_ms:
        Detection decode + mask decode latency.

    total_ms:
        Complete detector latency.
    """

    objects: List[DrivingObject]

    drivable_mask: np.ndarray
    lane_mask: np.ndarray

    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    total_ms: float


# ============================================================
# Detector
# ============================================================

class PanopticDrivingDetector:
    """
    YOLOPv2 ONNX wrapper for VehicleMind.

    One model provides:

        Object Detection
            +
        Drivable Area
            +
        Lane Segmentation

    Optimized primarily for Apple Silicon through
    ONNX Runtime CoreML Execution Provider.

    CPUExecutionProvider remains as fallback.
    """

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        prefer_coreml: bool = True,
        warmup_runs: int = 2,
    ):
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                "YOLOPv2 model not found: "
                f"{self.model_path}"
            )

        self.score_threshold = float(
            score_threshold
        )

        self.nms_threshold = float(
            nms_threshold
        )

        self.default_input_size = int(
            input_size
        )

        # ====================================================
        # ONNX Runtime
        # ====================================================

        available_providers = (
            ort.get_available_providers()
        )

        print(
            "[VehicleMind] "
            f"ONNX providers: "
            f"{available_providers}"
        )

        # ----------------------------------------------------
        # Session optimization
        # ----------------------------------------------------

        session_options = (
            ort.SessionOptions()
        )

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Sequential execution tends to work better when
        # CoreML owns the accelerated graph.
        session_options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        # ----------------------------------------------------
        # CoreML cache
        #
        # Avoid recompiling the CoreML graph every launch.
        # ----------------------------------------------------

        cache_dir = (
            self.model_path.parent
            / ".coreml_cache"
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # Providers
        # ====================================================

        providers = []

        if (
            prefer_coreml
            and
            "CoreMLExecutionProvider"
            in available_providers
        ):

            providers.append(
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",

                        # CPU + GPU + ANE where CoreML
                        # determines appropriate execution.
                        "MLComputeUnits": "ALL",

                        # Keep compatible with both the current
                        # dynamic YOLOPv2 model and the future
                        # static 640 model.
                        #
                        # A genuinely static ONNX model can
                        # still benefit from its fixed shape.
                        "RequireStaticInputShapes": "0",

                        "EnableOnSubgraphs": "0",

                        # Reuse compiled CoreML representation
                        # across launches.
                        "ModelCacheDirectory": str(
                            cache_dir
                        ),

                        # Prefer prediction latency.
                        "SpecializationStrategy":
                            "FastPrediction",
                    },
                )
            )

        providers.append(
            "CPUExecutionProvider"
        )

        # ====================================================
        # Create session
        # ====================================================

        print(
            "[VehicleMind] "
            f"Loading YOLOPv2: "
            f"{self.model_path}"
        )

        session_start = (
            time.perf_counter()
        )

        self.session = (
            ort.InferenceSession(
                str(
                    self.model_path
                ),
                sess_options=(
                    session_options
                ),
                providers=providers,
            )
        )

        session_load_ms = (
            (
                time.perf_counter()
                - session_start
            )
            * 1000.0
        )

        print(
            "[VehicleMind] "
            "Active providers: "
            f"{self.session.get_providers()}"
        )

        print(
            "[VehicleMind] "
            f"Session load: "
            f"{session_load_ms:.1f} ms"
        )

        # ====================================================
        # Input metadata
        # ====================================================

        input_meta = (
            self.session
            .get_inputs()[0]
        )

        self.input_name = (
            input_meta.name
        )

        input_shape = (
            input_meta.shape
        )

        print(
            "[VehicleMind] "
            f"Raw ONNX input shape: "
            f"{input_shape}"
        )

        # ----------------------------------------------------
        # Static vs dynamic ONNX
        # ----------------------------------------------------

        self.static_input = (
            len(input_shape) >= 4
            and
            isinstance(
                input_shape[2],
                int,
            )
            and
            isinstance(
                input_shape[3],
                int,
            )
        )

        if self.static_input:

            self.input_height = int(
                input_shape[2]
            )

            self.input_width = int(
                input_shape[3]
            )

            print(
                "[VehicleMind] "
                "Input mode: STATIC"
            )

        else:

            self.input_height = (
                self.default_input_size
            )

            self.input_width = (
                self.default_input_size
            )

            print(
                "[VehicleMind] "
                "Input mode: DYNAMIC"
            )

        print(
            "[VehicleMind] "
            f"YOLOPv2 target input: "
            f"{self.input_width}x"
            f"{self.input_height}"
        )

        # ====================================================
        # Output metadata
        # ====================================================

        outputs = (
            self.session
            .get_outputs()
        )

        print(
            "[VehicleMind] "
            f"YOLOPv2 outputs: "
            f"{len(outputs)}"
        )

        for index, output in enumerate(
            outputs
        ):

            print(
                f"  [{index}] "
                f"{output.name} "
                f"{output.shape}"
            )

        if len(outputs) != 6:

            raise RuntimeError(
                "Unexpected YOLOPv2 output count: "
                f"{len(outputs)}. "
                "Expected 6."
            )

        # ====================================================
        # Warmup
        # ====================================================

        if warmup_runs > 0:

            self._warmup(
                warmup_runs
            )

    # ========================================================
    # Warmup
    # ========================================================

    def _warmup(
        self,
        runs: int,
    ) -> None:
        """
        Warm up ONNX / CoreML execution.

        CoreML may perform graph compilation or specialization
        on the first inference, therefore first-frame timing
        should not be used as steady-state latency.
        """

        print(
            "[VehicleMind] "
            f"Warming up YOLOPv2 "
            f"({runs} runs)..."
        )

        dummy = np.zeros(
            (
                1,
                3,
                self.input_height,
                self.input_width,
            ),
            dtype=np.float32,
        )

        for index in range(
            runs
        ):

            start = (
                time.perf_counter()
            )

            self.session.run(
                None,
                {
                    self.input_name:
                        dummy
                },
            )

            elapsed_ms = (
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0
            )

            print(
                "[VehicleMind] "
                f"Warmup {index + 1}: "
                f"{elapsed_ms:.1f} ms"
            )

    # ========================================================
    # Preprocess
    # ========================================================

    def _preprocess(
        self,
        frame: np.ndarray,
    ):
        """
        OpenCV BGR frame -> YOLOPv2 NCHW float tensor.

        Static model:
            always produce exact H×W input.

        Dynamic model:
            retain YOLO letterbox auto-padding behavior.
        """

        image = (
            frame.copy()
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # static model:
        #
        #     auto=False
        #
        # Otherwise a 16:9 video could become 640x384 and a
        # fixed [1,3,640,640] model would reject the input.
        #
        # dynamic model:
        #
        #     auto=True
        #
        # which reduces unnecessary padding/computation.
        # ----------------------------------------------------

        image, ratio, (
            pad_w,
            pad_h,
        ) = (
            yolopv2_utils
            .letterbox(
                image,
                new_shape=(
                    self.input_height,
                    self.input_width,
                ),
                auto=(
                    not self.static_input
                ),
                scaleFill=False,
                scaleup=True,
                stride=32,
            )
        )

        # BGR -> RGB
        image = (
            image[
                :,
                :,
                ::-1
            ]
        )

        # HWC -> CHW
        image = (
            image.transpose(
                2,
                0,
                1,
            )
        )

        image = (
            np.ascontiguousarray(
                image
            )
        )

        image = (
            image.astype(
                np.float32,
                copy=False,
            )
            / 255.0
        )

        # CHW -> NCHW
        image = np.expand_dims(
            image,
            axis=0,
        )

        return (
            image,
            ratio,
            (
                pad_w,
                pad_h,
            ),
        )

    # ========================================================
    # Object decode
    # ========================================================

    def _decode_objects(
        self,
        outputs,
        input_image: np.ndarray,
        original_frame: np.ndarray,
    ) -> List[DrivingObject]:
        """
        Decode YOLOPv2 detection head.
        """

        # ----------------------------------------------------
        # results[0] is a sequence containing the three
        # detection scales.
        # ----------------------------------------------------

        detection_heads = [
            outputs[0][0],
            outputs[0][1],
            outputs[0][2],
        ]

        anchor_grid = [
            outputs[1],
            outputs[2],
            outputs[3],
        ]

        prediction = (
            yolopv2_utils
            .split_for_trace_model(
                detection_heads,
                anchor_grid,
            )
        )

        detections = (
            yolopv2_utils
            .non_max_suppression(
                prediction,
                conf_thres=(
                    self.score_threshold
                ),
                iou_thres=(
                    self.nms_threshold
                ),
            )
        )

        objects: List[
            DrivingObject
        ] = []

        for detection in detections:

            if (
                detection is None
                or
                len(detection) == 0
            ):
                continue

            # ------------------------------------------------
            # Model input coordinates -> original video
            # ------------------------------------------------

            detection[:, :4] = (
                yolopv2_utils
                .scale_coords(
                    input_image.shape[2:],
                    detection[:, :4],
                    original_frame.shape,
                )
                .round()
            )

            for item in detection:

                x1 = int(
                    item[0]
                )

                y1 = int(
                    item[1]
                )

                x2 = int(
                    item[2]
                )

                y2 = int(
                    item[3]
                )

                confidence = float(
                    item[4]
                )

                class_id = int(
                    item[5]
                )

                if (
                    0
                    <= class_id
                    < len(
                        BDD100K_CLASSES
                    )
                ):

                    class_name = (
                        BDD100K_CLASSES[
                            class_id
                        ]
                    )

                else:

                    class_name = (
                        f"class_{class_id}"
                    )

                objects.append(
                    DrivingObject(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return objects

    # ========================================================
    # Segmentation decode
    # ========================================================

    def _decode_masks(
        self,
        outputs,
        pad,
        original_size,
    ):
        """
        Decode:

            Drivable Area
            Lane Markings

        Preserve probability maps as float32 until after
        resizing back to the original resolution.
        """

        pad_w, pad_h = (
            pad
        )

        # ====================================================
        # Raw probability maps
        # ====================================================

        drivable_prob = (
            yolopv2_utils
            .driving_area_mask(
                outputs[4],
                (
                    pad_w,
                    pad_h,
                ),
            )
        )

        lane_prob = (
            yolopv2_utils
            .lane_line_mask(
                outputs[5],
                (
                    pad_w,
                    pad_h,
                ),
            )
        )

        drivable_prob = (
            np.asarray(
                drivable_prob,
                dtype=np.float32,
            )
        )

        lane_prob = (
            np.asarray(
                lane_prob,
                dtype=np.float32,
            )
        )

        # ====================================================
        # Restore original video resolution
        # ====================================================

        (
            original_width,
            original_height,
        ) = original_size

        drivable_prob = (
            cv2.resize(
                drivable_prob,
                (
                    original_width,
                    original_height,
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                ),
            )
        )

        lane_prob = (
            cv2.resize(
                lane_prob,
                (
                    original_width,
                    original_height,
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                ),
            )
        )

        # ====================================================
        # Binary masks
        # ====================================================

        drivable_mask = (
            drivable_prob
            > 0.5
        ).astype(
            np.uint8
        )

        lane_mask = (
            lane_prob
            > 0.5
        ).astype(
            np.uint8
        )

        return (
            drivable_mask,
            lane_mask,
        )

    # ========================================================
    # Main inference
    # ========================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> DrivingSceneResult:
        """
        Run complete VehicleMind road perception.

        Timing is separated into:

            preprocess
            inference
            postprocess
            total
        """

        total_start = (
            time.perf_counter()
        )

        # ====================================================
        # Preprocess
        # ====================================================

        preprocess_start = (
            time.perf_counter()
        )

        (
            input_image,
            _,
            pad,
        ) = self._preprocess(
            frame
        )

        preprocess_ms = (
            (
                time.perf_counter()
                - preprocess_start
            )
            * 1000.0
        )

        # ====================================================
        # ONNX inference
        # ====================================================

        inference_start = (
            time.perf_counter()
        )

        outputs = (
            self.session.run(
                None,
                {
                    self.input_name:
                        input_image
                },
            )
        )

        inference_ms = (
            (
                time.perf_counter()
                - inference_start
            )
            * 1000.0
        )

        # ====================================================
        # Validate outputs
        # ====================================================

        if len(outputs) != 6:

            raise RuntimeError(
                "Unexpected YOLOPv2 runtime "
                "output count: "
                f"{len(outputs)}"
            )

        # ====================================================
        # Postprocess
        # ====================================================

        postprocess_start = (
            time.perf_counter()
        )

        objects = (
            self._decode_objects(
                outputs=outputs,
                input_image=input_image,
                original_frame=frame,
            )
        )

        height, width = (
            frame.shape[:2]
        )

        (
            drivable_mask,
            lane_mask,
        ) = self._decode_masks(
            outputs=outputs,
            pad=pad,
            original_size=(
                width,
                height,
            ),
        )

        postprocess_ms = (
            (
                time.perf_counter()
                - postprocess_start
            )
            * 1000.0
        )

        total_ms = (
            (
                time.perf_counter()
                - total_start
            )
            * 1000.0
        )

        return DrivingSceneResult(
            objects=objects,
            drivable_mask=(
                drivable_mask
            ),
            lane_mask=(
                lane_mask
            ),
            inference_ms=(
                inference_ms
            ),
            preprocess_ms=(
                preprocess_ms
            ),
            postprocess_ms=(
                postprocess_ms
            ),
            total_ms=(
                total_ms
            ),
        )

    # ========================================================
    # Information
    # ========================================================

    def get_runtime_info(
        self,
    ) -> dict:
        """
        Runtime information useful for benchmarking.
        """

        return {
            "model": str(
                self.model_path
            ),
            "providers": (
                self.session
                .get_providers()
            ),
            "static_input": (
                self.static_input
            ),
            "input_width": (
                self.input_width
            ),
            "input_height": (
                self.input_height
            ),
        }