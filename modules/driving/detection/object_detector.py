from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class RoadObject:
    """
    One detected road object.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    confidence: float

    class_id: int
    class_name: str


@dataclass
class RoadObjectResult:
    """
    Detection result for one video frame.
    """

    objects: List[RoadObject]

    counts: Dict[str, int]


class RoadObjectDetector:
    """
    Generic road-object detector for VehicleMind.

    This module only performs visual perception.

    It does NOT perform:
        - object tracking
        - distance estimation
        - TTC estimation
        - collision-risk assessment
        - planning

    Those belong to higher-level modules.
    """

    DEFAULT_CLASSES = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "traffic light",
        "stop sign",
    )

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        target_classes: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold

        self.image_size = image_size

        # ====================================================
        # Device
        # ====================================================

        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"

            elif torch.cuda.is_available():
                self.device = "cuda"

            else:
                self.device = "cpu"

        else:
            self.device = device

        print(f"[VehicleMind] RoadObjectDetector device: {self.device}")

        # ====================================================
        # Model
        # ====================================================

        self.model = YOLO(model_name)

        # ====================================================
        # Target classes
        # ====================================================

        if target_classes is None:
            target_classes = self.DEFAULT_CLASSES

        self.target_classes = {name.strip().lower() for name in target_classes}

        # Resolve names from model metadata instead of
        # hard-coding COCO class IDs.
        self.target_class_ids = []

        for class_id, name in self.model.names.items():
            normalized_name = name.strip().lower()

            if normalized_name in self.target_classes:
                self.target_class_ids.append(int(class_id))

        if not self.target_class_ids:
            raise RuntimeError(
                "No requested road-object classes exist in the loaded detection model."
            )

        resolved_names = [
            self.model.names[class_id] for class_id in self.target_class_ids
        ]

        print("[VehicleMind] Road classes: " + ", ".join(resolved_names))

    def detect(
        self,
        frame: np.ndarray,
    ) -> RoadObjectResult:
        """
        Detect generic road objects in one BGR frame.
        """

        results = self.model.predict(
            source=frame,
            conf=(self.confidence_threshold),
            imgsz=(self.image_size),
            classes=(self.target_class_ids),
            device=self.device,
            verbose=False,
        )

        objects: List[RoadObject] = []

        counts: Dict[str, int] = {}

        if not results:
            return RoadObjectResult(
                objects=[],
                counts={},
            )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return RoadObjectResult(
                objects=[],
                counts={},
            )

        # ====================================================
        # Parse detections
        # ====================================================

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()

            confidence = float(box.conf[0].detach().cpu().item())

            class_id = int(box.cls[0].detach().cpu().item())

            class_name = str(self.model.names[class_id])

            road_object = RoadObject(
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )

            objects.append(road_object)

            counts[class_name] = (
                counts.get(
                    class_name,
                    0,
                )
                + 1
            )

        return RoadObjectResult(
            objects=objects,
            counts=counts,
        )
