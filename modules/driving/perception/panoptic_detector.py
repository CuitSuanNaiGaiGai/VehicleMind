from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

from modules.driving.perception import yolopv2_utils
from modules.driving.perception.postprocess import decode_masks
from modules.driving.perception.preprocess import preprocess_frame
from modules.driving.perception.runtime import coreml_cache_directory, warm_up_session
from modules.driving.perception.types import DrivingObject, DrivingSceneResult


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
        coreml_cache_dir: Path | None = None,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOPv2 model not found: {self.model_path}")

        self.score_threshold = float(score_threshold)

        self.nms_threshold = float(nms_threshold)

        self.default_input_size = int(input_size)

        # ====================================================
        # ONNX Runtime
        # ====================================================

        available_providers = ort.get_available_providers()

        print(f"[VehicleMind] ONNX providers: {available_providers}")

        # ----------------------------------------------------
        # Session optimization
        # ----------------------------------------------------

        session_options = ort.SessionOptions()

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Sequential execution tends to work better when
        # CoreML owns the accelerated graph.
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # ----------------------------------------------------
        # CoreML cache
        #
        # Avoid recompiling the CoreML graph every launch.
        # ----------------------------------------------------

        cache_dir = coreml_cache_directory(
            model_path=self.model_path,
            prefer_coreml=prefer_coreml,
            available_providers=available_providers,
            override=coreml_cache_dir,
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # ====================================================
        # Providers
        # ====================================================

        providers = []

        if prefer_coreml and "CoreMLExecutionProvider" in available_providers:
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
                        "ModelCacheDirectory": str(cache_dir),
                        # Prefer prediction latency.
                        "SpecializationStrategy": "FastPrediction",
                    },
                )
            )

        providers.append("CPUExecutionProvider")

        # ====================================================
        # Create session
        # ====================================================

        print(f"[VehicleMind] Loading YOLOPv2: {self.model_path}")

        session_start = time.perf_counter()

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=(session_options),
            providers=providers,
        )

        session_load_ms = (time.perf_counter() - session_start) * 1000.0

        print(f"[VehicleMind] Active providers: {self.session.get_providers()}")

        print(f"[VehicleMind] Session load: {session_load_ms:.1f} ms")

        # ====================================================
        # Input metadata
        # ====================================================

        input_meta = self.session.get_inputs()[0]

        self.input_name = input_meta.name

        input_shape = input_meta.shape

        print(f"[VehicleMind] Raw ONNX input shape: {input_shape}")

        # ----------------------------------------------------
        # Static vs dynamic ONNX
        # ----------------------------------------------------

        self.static_input = (
            len(input_shape) >= 4
            and isinstance(
                input_shape[2],
                int,
            )
            and isinstance(
                input_shape[3],
                int,
            )
        )

        if self.static_input:
            self.input_height = int(input_shape[2])

            self.input_width = int(input_shape[3])

            print("[VehicleMind] Input mode: STATIC")

        else:
            self.input_height = self.default_input_size

            self.input_width = self.default_input_size

            print("[VehicleMind] Input mode: DYNAMIC")

        print(
            "[VehicleMind] "
            f"YOLOPv2 target input: "
            f"{self.input_width}x"
            f"{self.input_height}"
        )

        # ====================================================
        # Output metadata
        # ====================================================

        outputs = self.session.get_outputs()

        print(f"[VehicleMind] YOLOPv2 outputs: {len(outputs)}")

        for index, output in enumerate(outputs):
            print(f"  [{index}] {output.name} {output.shape}")

        if len(outputs) != 6:
            raise RuntimeError(
                f"Unexpected YOLOPv2 output count: {len(outputs)}. Expected 6."
            )

        # ====================================================
        # Warmup
        # ====================================================

        if warmup_runs > 0:
            self._warmup(warmup_runs)

    # ========================================================
    # Warmup
    # ========================================================

    def _warmup(
        self,
        runs: int,
    ) -> None:
        warm_up_session(
            self.session,
            input_name=self.input_name,
            input_height=self.input_height,
            input_width=self.input_width,
            runs=runs,
        )

    # ========================================================
    # Preprocess
    # ========================================================

    def _preprocess(
        self,
        frame: np.ndarray,
    ):
        return preprocess_frame(
            frame,
            input_height=self.input_height,
            input_width=self.input_width,
            static_input=self.static_input,
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

        prediction = yolopv2_utils.split_for_trace_model(
            detection_heads,
            anchor_grid,
        )

        detections = yolopv2_utils.non_max_suppression(
            prediction,
            conf_thres=(self.score_threshold),
            iou_thres=(self.nms_threshold),
        )

        objects: List[DrivingObject] = []

        for detection in detections:
            if detection is None or len(detection) == 0:
                continue

            # ------------------------------------------------
            # Model input coordinates -> original video
            # ------------------------------------------------

            detection[:, :4] = yolopv2_utils.scale_coords(
                input_image.shape[2:],
                detection[:, :4],
                original_frame.shape,
            ).round()

            for item in detection:
                x1 = int(item[0])

                y1 = int(item[1])

                x2 = int(item[2])

                y2 = int(item[3])

                confidence = float(item[4])

                class_id = int(item[5])

                if 0 <= class_id < len(BDD100K_CLASSES):
                    class_name = BDD100K_CLASSES[class_id]

                else:
                    class_name = f"class_{class_id}"

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
        return decode_masks(outputs, pad, original_size)

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

        total_start = time.perf_counter()

        # ====================================================
        # Preprocess
        # ====================================================

        preprocess_start = time.perf_counter()

        (
            input_image,
            _,
            pad,
        ) = self._preprocess(frame)

        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        # ====================================================
        # ONNX inference
        # ====================================================

        inference_start = time.perf_counter()

        outputs = self.session.run(
            None,
            {self.input_name: input_image},
        )

        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        # ====================================================
        # Validate outputs
        # ====================================================

        if len(outputs) != 6:
            raise RuntimeError(
                f"Unexpected YOLOPv2 runtime output count: {len(outputs)}"
            )

        # ====================================================
        # Postprocess
        # ====================================================

        postprocess_start = time.perf_counter()

        objects = self._decode_objects(
            outputs=outputs,
            input_image=input_image,
            original_frame=frame,
        )

        height, width = frame.shape[:2]

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

        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return DrivingSceneResult(
            objects=objects,
            drivable_mask=(drivable_mask),
            lane_mask=(lane_mask),
            inference_ms=(inference_ms),
            preprocess_ms=(preprocess_ms),
            postprocess_ms=(postprocess_ms),
            total_ms=(total_ms),
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
            "model": str(self.model_path),
            "providers": (self.session.get_providers()),
            "static_input": (self.static_input),
            "input_width": (self.input_width),
            "input_height": (self.input_height),
        }
