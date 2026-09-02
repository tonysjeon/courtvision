"""Stable event schemas for tracked players."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from courtvision.players.filtering import PlayerId


class PixelPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class CourtPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class PlayerTrackingEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    match_id: str
    frame_id: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0)
    object_type: Literal["player"] = "player"
    object_id: PlayerId
    track_id: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    pixel_position: PixelPosition
    court_position: CourtPosition
