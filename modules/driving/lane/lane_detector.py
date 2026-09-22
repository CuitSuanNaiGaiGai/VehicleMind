from dataclasses import dataclass, replace
from typing import Optional, Tuple

import cv2
import numpy as np

from modules.config import LanePerceptionConfig


Line = Tuple[int, int, int, int]


@dataclass
class LaneResult:
    """
    Generic lane-perception result.

    left_lane / right_lane:
        (x1, y1, x2, y2)

    lane_center:
        Estimated lane center at the bottom of the frame.
    """

    left_detected: bool
    right_detected: bool

    left_lane: Optional[Line]
    right_lane: Optional[Line]

    lane_center: Optional[int]


class LaneDetector:
    """
    Lightweight lane perception for VehicleMind.

    Pipeline:

        BGR Frame
            ↓
        HLS color mask
            +
        Canny edges
            ↓
        Road ROI
            ↓
        Hough segments
            ↓
        Left / Right classification
            ↓
        Weighted line fitting
            ↓
        Temporal smoothing

    This is an engineering baseline for a general
    Driving Perception demo, not a production lane model.
    """

    def __init__(
        self,
        config: LanePerceptionConfig | None = None,
        *,
        smoothing: float | None = None,
        min_abs_slope: float | None = None,
        max_abs_slope: float | None = None,
    ):
        resolved = config or LanePerceptionConfig()
        compatibility_overrides = {
            name: value
            for name, value in (
                ("smoothing", smoothing),
                ("min_abs_slope", min_abs_slope),
                ("max_abs_slope", max_abs_slope),
            )
            if value is not None
        }
        if compatibility_overrides:
            resolved = replace(resolved, **compatibility_overrides)

        self.config = resolved
        self.smoothing = resolved.smoothing
        self.min_abs_slope = resolved.min_abs_slope
        self.max_abs_slope = resolved.max_abs_slope

        self._left_line = None
        self._right_line = None

    # ========================================================
    # Color mask
    # ========================================================

    def _lane_color_mask(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Detect common white / yellow lane markings.
        """

        hls = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HLS,
        )

        # ----------------------------------------------------
        # White lane
        # ----------------------------------------------------

        white_lower = np.array(self.config.white_hls_lower, dtype=np.uint8)

        white_upper = np.array(self.config.white_hls_upper, dtype=np.uint8)

        white_mask = cv2.inRange(
            hls,
            white_lower,
            white_upper,
        )

        # ----------------------------------------------------
        # Yellow lane
        # ----------------------------------------------------

        yellow_lower = np.array(self.config.yellow_hls_lower, dtype=np.uint8)

        yellow_upper = np.array(self.config.yellow_hls_upper, dtype=np.uint8)

        yellow_mask = cv2.inRange(
            hls,
            yellow_lower,
            yellow_upper,
        )

        return cv2.bitwise_or(
            white_mask,
            yellow_mask,
        )

    # ========================================================
    # Edge extraction
    # ========================================================

    def _extract_edges(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        edges = cv2.Canny(
            blurred,
            self.config.canny_low,
            self.config.canny_high,
        )

        color_mask = self._lane_color_mask(frame)

        color_edges = cv2.Canny(
            color_mask,
            self.config.color_canny_low,
            self.config.color_canny_high,
        )

        return cv2.bitwise_or(
            edges,
            color_edges,
        )

    # ========================================================
    # Road ROI
    # ========================================================

    def _road_roi(
        self,
        edges: np.ndarray,
    ) -> np.ndarray:

        height, width = edges.shape[:2]

        mask = np.zeros_like(edges)

        # Broad trapezoid representing the road ahead.
        polygon = np.array(
            [
                [
                    (
                        int(width * self.config.roi_left),
                        height,
                    ),
                    (
                        int(width * self.config.roi_top_left),
                        int(height * self.config.roi_top_height),
                    ),
                    (
                        int(width * self.config.roi_top_right),
                        int(height * self.config.roi_top_height),
                    ),
                    (
                        int(width * self.config.roi_right),
                        height,
                    ),
                ]
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(
            mask,
            polygon,
            255,
        )

        return cv2.bitwise_and(
            edges,
            mask,
        )

    # ========================================================
    # Fit one lane
    # ========================================================

    def _fit_lane(
        self,
        segments,
        height: int,
        width: int,
        left: bool,
    ) -> Optional[Line]:

        xs = []
        ys = []
        weights = []

        center_x = width / 2.0

        for segment in segments:
            x1, y1, x2, y2 = segment

            dx = x2 - x1

            dy = y2 - y1

            if abs(dx) < 1:
                continue

            slope = dy / dx

            abs_slope = abs(slope)

            if abs_slope < self.min_abs_slope or abs_slope > self.max_abs_slope:
                continue

            midpoint_x = (x1 + x2) / 2.0

            # ------------------------------------------------
            # In image coordinates:
            #
            # Left lane generally has negative slope.
            # Right lane generally has positive slope.
            # ------------------------------------------------

            if left:
                if slope >= 0 or midpoint_x >= center_x:
                    continue

            else:
                if slope <= 0 or midpoint_x <= center_x:
                    continue

            length = float(
                np.hypot(
                    x2 - x1,
                    y2 - y1,
                )
            )

            xs.extend([x1, x2])

            ys.extend([y1, y2])

            weights.extend([length, length])

        if len(xs) < 4:
            return None

        xs = np.asarray(
            xs,
            dtype=np.float64,
        )

        ys = np.asarray(
            ys,
            dtype=np.float64,
        )

        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Fit:
        #
        # x = a*y + b
        #
        # This is more stable than fitting y=f(x) for
        # near-vertical lane markings.
        # ----------------------------------------------------

        try:
            coefficients = np.polyfit(
                ys,
                xs,
                deg=1,
                w=weights,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
        ):
            return None

        a, b = coefficients

        y_bottom = height - 1

        y_top = int(height * self.config.fit_top_height)

        x_bottom = int(a * y_bottom + b)

        x_top = int(a * y_top + b)

        # Keep coordinates inside a reasonable range.
        x_bottom = int(
            np.clip(
                x_bottom,
                -width,
                width * 2,
            )
        )

        x_top = int(
            np.clip(
                x_top,
                -width,
                width * 2,
            )
        )

        return (
            x_bottom,
            y_bottom,
            x_top,
            y_top,
        )

    # ========================================================
    # Temporal smoothing
    # ========================================================

    def _smooth_line(
        self,
        previous,
        current,
    ):

        if current is None:
            return previous

        if previous is None:
            return current

        alpha = self.smoothing

        smoothed = []

        for old, new in zip(
            previous,
            current,
        ):
            value = alpha * old + (1.0 - alpha) * new

            smoothed.append(int(value))

        return tuple(smoothed)

    # ========================================================
    # Main
    # ========================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> LaneResult:

        height, width = frame.shape[:2]

        edges = self._extract_edges(frame)

        roi = self._road_roi(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        segments = []

        if lines is not None:
            lines = np.asarray(
                lines,
                dtype=np.int32,
            ).reshape(-1, 4)

            for (
                x1,
                y1,
                x2,
                y2,
            ) in lines:
                segments.append(
                    (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )
                )

        current_left = self._fit_lane(
            segments,
            height,
            width,
            left=True,
        )

        current_right = self._fit_lane(
            segments,
            height,
            width,
            left=False,
        )

        self._left_line = self._smooth_line(
            self._left_line,
            current_left,
        )

        self._right_line = self._smooth_line(
            self._right_line,
            current_right,
        )

        # ----------------------------------------------------
        # Lane center
        # ----------------------------------------------------

        lane_center = None

        if self._left_line is not None and self._right_line is not None:
            left_bottom_x = self._left_line[0]

            right_bottom_x = self._right_line[0]

            lane_center = int((left_bottom_x + right_bottom_x) / 2.0)

        return LaneResult(
            left_detected=(current_left is not None),
            right_detected=(current_right is not None),
            left_lane=(self._left_line),
            right_lane=(self._right_line),
            lane_center=lane_center,
        )

    def reset(
        self,
    ):

        self._left_line = None
        self._right_line = None
