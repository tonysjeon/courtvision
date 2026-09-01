"""Reusable, timestamp-aware video frame iteration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class VideoMetadata(BaseModel):
    """Metadata reported by the video container."""

    model_config = ConfigDict(frozen=True)

    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(ge=0)


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded frame and its position in the source video."""

    frame_id: int
    timestamp_ms: float
    image: np.ndarray


class VideoReader:
    """Iterate over selected frames from an MP4 or MOV recording."""

    supported_extensions = {".mp4", ".mov"}

    def __init__(
        self,
        path: str | Path,
        *,
        frame_skip: int = 0,
        max_frames: int | None = None,
        start_timestamp_ms: float = 0.0,
        end_timestamp_ms: float | None = None,
    ) -> None:
        self.path = Path(path)
        self._validate_options(frame_skip, max_frames, start_timestamp_ms, end_timestamp_ms)

        if self.path.suffix.lower() not in self.supported_extensions:
            raise ValueError("Video must use an .mp4 or .mov extension")
        if not self.path.is_file():
            raise FileNotFoundError(f"Video does not exist: {self.path}")

        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.start_timestamp_ms = start_timestamp_ms
        self.end_timestamp_ms = end_timestamp_ms
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"OpenCV could not open video: {self.path}")

        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            self._capture.release()
            raise ValueError(f"Video contains invalid metadata: {self.path}")

        self.metadata = VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=frame_count / fps,
        )

    @staticmethod
    def _validate_options(
        frame_skip: int,
        max_frames: int | None,
        start_timestamp_ms: float,
        end_timestamp_ms: float | None,
    ) -> None:
        if frame_skip < 0:
            raise ValueError("frame_skip must be non-negative")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if start_timestamp_ms < 0:
            raise ValueError("start_timestamp_ms must be non-negative")
        if end_timestamp_ms is not None and end_timestamp_ms <= start_timestamp_ms:
            raise ValueError("end_timestamp_ms must be greater than start_timestamp_ms")

    def __iter__(self) -> Iterator[VideoFrame]:
        if not self._capture.isOpened():
            raise RuntimeError("VideoReader is closed")

        start_frame = ceil(self.start_timestamp_ms * self.metadata.fps / 1000)
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        step = self.frame_skip + 1
        yielded = 0
        frame_id = start_frame

        while self.max_frames is None or yielded < self.max_frames:
            ok, image = self._capture.read()
            if not ok:
                break

            timestamp_ms = frame_id * 1000 / self.metadata.fps
            if self.end_timestamp_ms is not None and timestamp_ms >= self.end_timestamp_ms:
                break

            yield VideoFrame(frame_id=frame_id, timestamp_ms=timestamp_ms, image=image)
            yielded += 1

            for _ in range(step - 1):
                if not self._capture.grab():
                    return
                frame_id += 1
            frame_id += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
