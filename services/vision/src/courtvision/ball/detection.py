"""Common tennis-ball detection types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from courtvision.players.detection import BoundingBox


class BallCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    pixel_x: float = Field(ge=0)
    pixel_y: float = Field(ge=0)
    radius: float = Field(gt=0)


class BallDetector(Protocol):
    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: float,
        excluded_boxes: Sequence[BoundingBox] = (),
    ) -> list[BallCandidate]: ...
