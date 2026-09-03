"""Common person-detection types and interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")
        return self

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def ground_position(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, self.y2


class PersonDetection(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class PersonDetector(Protocol):
    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: float,
    ) -> list[PersonDetection]: ...
