from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from courtvision.video.reader import VideoReader


class FakeCapture:
    def __init__(self, _: str) -> None:
        self.opened = True
        self.position = 0
        self.frames = [np.full((48, 64, 3), index, dtype=np.uint8) for index in range(10)]

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        fixed_properties = {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 10.0,
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        }
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return max(0.0, (self.position - 1) * 40.0)
        return fixed_properties[property_id]

    def set(self, property_id: int, value: float) -> bool:
        assert property_id == cv2.CAP_PROP_POS_MSEC
        self.position = int(np.ceil(value / 40.0))
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def grab(self) -> bool:
        if self.position >= len(self.frames):
            return False
        self.position += 1
        return True

    def release(self) -> None:
        self.opened = False


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.touch()
    return path


def test_reports_metadata(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    with VideoReader(video_file) as reader:
        assert reader.metadata.model_dump() == {
            "fps": 25.0,
            "frame_count": 10,
            "width": 64,
            "height": 48,
            "duration_seconds": 0.4,
        }


def test_iterates_requested_frame_range(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    with VideoReader(
        video_file,
        start_timestamp_ms=40,
        end_timestamp_ms=300,
        frame_skip=1,
        max_frames=3,
    ) as reader:
        frames = list(reader)

    assert [frame.frame_id for frame in frames] == [1, 3, 5]
    assert [frame.timestamp_ms for frame in frames] == [40.0, 120.0, 200.0]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1, 3, 5]


def test_starts_at_first_frame_on_or_after_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    video_file: Path,
) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    with VideoReader(video_file, start_timestamp_ms=41, max_frames=1) as reader:
        frame = next(iter(reader))

    assert frame.frame_id == 2
    assert frame.timestamp_ms == 80.0


@pytest.mark.parametrize("extension", [".avi", ".mkv", ""])
def test_rejects_unsupported_extensions(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"clip{extension}"
    path.touch()
    with pytest.raises(ValueError, match="mp4 or .mov"):
        VideoReader(path)


def test_rejects_invalid_range(video_file: Path) -> None:
    with pytest.raises(ValueError, match="greater than"):
        VideoReader(video_file, start_timestamp_ms=100, end_timestamp_ms=100)
