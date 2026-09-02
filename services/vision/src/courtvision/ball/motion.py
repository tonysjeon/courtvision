"""Color and motion based tennis-ball candidate detection."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from courtvision.ball.detection import BallCandidate
from courtvision.geometry.mapper import CourtMapper
from courtvision.players.detection import BoundingBox


class MotionBallDetector:
    """Find small, moving yellow-green objects near the calibrated court."""

    def __init__(
        self,
        mapper: CourtMapper,
        *,
        min_area: float = 2.0,
        max_area: float = 450.0,
        max_dimension: int = 42,
        court_margin: float = 35.0,
    ) -> None:
        self.mapper = mapper
        self.min_area = min_area
        self.max_area = max_area
        self.max_dimension = max_dimension
        self.court_margin = court_margin
        self._previous_gray: np.ndarray | None = None

    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: float,
        excluded_boxes: Sequence[BoundingBox] = (),
    ) -> list[BallCandidate]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_mask = cv2.inRange(
            hsv,
            np.array((18, 105, 150), dtype=np.uint8),
            np.array((48, 255, 255), dtype=np.uint8),
        )

        if self._previous_gray is None:
            motion_mask = np.full(gray.shape, 255, dtype=np.uint8)
        else:
            difference = cv2.absdiff(gray, self._previous_gray)
            motion_mask = cv2.threshold(difference, 12, 255, cv2.THRESH_BINARY)[1]
            motion_mask = cv2.dilate(motion_mask, np.ones((5, 5), np.uint8), iterations=1)
        self._previous_gray = gray

        mask = cv2.bitwise_and(color_mask, motion_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            component_area = width * height
            if not self.min_area <= max(area, component_area) <= self.max_area:
                continue
            if width > self.max_dimension or height > self.max_dimension:
                continue
            aspect_ratio = width / height
            if not 0.25 <= aspect_ratio <= 4.0:
                continue

            (pixel_x, pixel_y), radius = cv2.minEnclosingCircle(contour)
            if self._inside_player(pixel_x, pixel_y, excluded_boxes):
                continue
            court_x, court_y = self.mapper.transform(pixel_x, pixel_y)
            margin = self.court_margin
            if not -margin <= court_x <= 100 + margin or not -margin <= court_y <= 100 + margin:
                continue

            region = hsv[y : y + height, x : x + width]
            saturation = float(region[:, :, 1].mean()) / 255
            brightness = float(region[:, :, 2].mean()) / 255
            moving = float(cv2.countNonZero(motion_mask[y : y + height, x : x + width]))
            motion_ratio = moving / component_area
            compactness = min(width, height) / max(width, height)
            confidence = min(
                1.0,
                0.30 * saturation + 0.25 * brightness + 0.25 * motion_ratio + 0.20 * compactness,
            )
            candidates.append(
                BallCandidate(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    confidence=confidence,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    radius=max(radius, 1.0),
                )
            )
        return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)[:12]

    @staticmethod
    def _inside_player(
        pixel_x: float,
        pixel_y: float,
        boxes: Sequence[BoundingBox],
    ) -> bool:
        return any(box.x1 <= pixel_x <= box.x2 and box.y1 <= pixel_y <= box.y2 for box in boxes)
