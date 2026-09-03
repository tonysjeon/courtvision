"""Manual court calibration file models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PixelPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def support_coordinate_pairs(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"x": value[0], "y": value[1]}
        return value


class CourtKeypoints(BaseModel):
    """Image locations of the four singles-court baseline corners."""

    model_config = ConfigDict(frozen=True)

    near_left_baseline: PixelPoint
    near_right_baseline: PixelPoint
    far_left_baseline: PixelPoint
    far_right_baseline: PixelPoint

    def ordered_points(self) -> list[tuple[float, float]]:
        return [
            (self.near_left_baseline.x, self.near_left_baseline.y),
            (self.near_right_baseline.x, self.near_right_baseline.y),
            (self.far_left_baseline.x, self.far_left_baseline.y),
            (self.far_right_baseline.x, self.far_right_baseline.y),
        ]


class CourtCalibration(BaseModel):
    """Versioned manual calibration for a particular video."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    video: str | None = None
    frame_width: int | None = Field(default=None, gt=0)
    frame_height: int | None = Field(default=None, gt=0)
    keypoints: CourtKeypoints

    @model_validator(mode="before")
    @classmethod
    def support_flat_keypoint_files(cls, value: object) -> object:
        if isinstance(value, dict) and "keypoints" not in value:
            return {"keypoints": value}
        return value

    @model_validator(mode="after")
    def require_complete_frame_dimensions(self) -> CourtCalibration:
        if (self.frame_width is None) != (self.frame_height is None):
            raise ValueError("frame_width and frame_height must be provided together")
        return self

    def ordered_points_for_frame(
        self,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> list[tuple[float, float]]:
        """Return calibration points scaled to the dimensions of a video frame."""
        if (frame_width is None) != (frame_height is None):
            raise ValueError("frame_width and frame_height must be provided together")

        points = self.keypoints.ordered_points()
        if frame_width is None or self.frame_width is None:
            return points

        scale_x = frame_width / self.frame_width
        scale_y = frame_height / self.frame_height
        return [(x * scale_x, y * scale_y) for x, y in points]

    @classmethod
    def from_json(cls, path: str | Path) -> CourtCalibration:
        calibration_path = Path(path)
        try:
            data = json.loads(calibration_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration does not exist: {calibration_path}") from None
        except json.JSONDecodeError as error:
            raise ValueError(f"Calibration is not valid JSON: {calibration_path}") from error
        return cls.model_validate(data)
