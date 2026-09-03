from __future__ import annotations

import numpy as np
import pytest
from courtvision.players.detection import BoundingBox, PersonDetection
from courtvision.players.yolo import YoloPersonDetector
from pydantic import ValidationError


def test_bounding_box_positions() -> None:
    bbox = BoundingBox(x1=10, y1=20, x2=30, y2=60)

    assert bbox.center == (20, 40)
    assert bbox.ground_position == (20, 60)


def test_rejects_invalid_bounding_box() -> None:
    with pytest.raises(ValidationError, match="positive width"):
        BoundingBox(x1=30, y1=20, x2=10, y2=60)


class FakeTensor:
    def __init__(self, values: list[list[float]] | list[float]) -> None:
        self.values = np.asarray(values)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[10, 20, 30, 60], [40, 10, 80, 90]])
    conf = FakeTensor([0.9, 0.75])


class FakePrediction:
    boxes = FakeBoxes()


class FakeModel:
    def predict(self, **_: object) -> list[FakePrediction]:
        return [FakePrediction()]


def test_yolo_adapter_returns_common_detection_types() -> None:
    detector = YoloPersonDetector.__new__(YoloPersonDetector)
    detector._model = FakeModel()
    detector.confidence_threshold = 0.35
    detector.device = None

    detections = detector.detect(
        np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=12,
        timestamp_ms=400,
    )

    assert detections == [
        PersonDetection(
            frame_id=12,
            timestamp_ms=400,
            confidence=0.9,
            bbox=BoundingBox(x1=10, y1=20, x2=30, y2=60),
        ),
        PersonDetection(
            frame_id=12,
            timestamp_ms=400,
            confidence=0.75,
            bbox=BoundingBox(x1=40, y1=10, x2=80, y2=90),
        ),
    ]
