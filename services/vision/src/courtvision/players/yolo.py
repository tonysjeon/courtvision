"""Ultralytics YOLO person detector."""

from __future__ import annotations

from typing import Any

import numpy as np

from courtvision.players.detection import BoundingBox, PersonDetection


class YoloPersonDetector:
    """Run a pretrained YOLO model against the COCO person class."""

    def __init__(
        self,
        model: str = "yolo11n.pt",
        *,
        confidence_threshold: float = 0.20,
        device: str | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        from ultralytics import YOLO

        self._model = YOLO(model)
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: float,
    ) -> list[PersonDetection]:
        predictions: list[Any] = self._model.predict(
            source=image,
            classes=[0],
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not predictions or predictions[0].boxes is None:
            return []

        boxes = predictions[0].boxes
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        return [
            PersonDetection(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                confidence=float(confidence),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )
            for (x1, y1, x2, y2), confidence in zip(coordinates, confidences, strict=True)
        ]
