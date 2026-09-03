"""TrackNet-based tennis-ball detection."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from courtvision.ball.detection import BallCandidate
from courtvision.players.detection import BoundingBox

INPUT_WIDTH = 640
INPUT_HEIGHT = 360


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _TrackNet(nn.Module):
    """TrackNet V1 architecture for three consecutive video frames."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _ConvBlock(9, 64)
        self.conv2 = _ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = _ConvBlock(64, 128)
        self.conv4 = _ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = _ConvBlock(128, 256)
        self.conv6 = _ConvBlock(256, 256)
        self.conv7 = _ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = _ConvBlock(256, 512)
        self.conv9 = _ConvBlock(512, 512)
        self.conv10 = _ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = _ConvBlock(512, 256)
        self.conv12 = _ConvBlock(256, 256)
        self.conv13 = _ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = _ConvBlock(256, 128)
        self.conv15 = _ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = _ConvBlock(128, 64)
        self.conv17 = _ConvBlock(64, 64)
        self.conv18 = _ConvBlock(64, 256)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        outputs = self.conv2(self.conv1(inputs))
        outputs = self.conv4(self.conv3(self.pool1(outputs)))
        outputs = self.conv7(self.conv6(self.conv5(self.pool2(outputs))))
        outputs = self.conv10(self.conv9(self.conv8(self.pool3(outputs))))
        outputs = self.conv13(self.conv12(self.conv11(self.ups1(outputs))))
        outputs = self.conv15(self.conv14(self.ups2(outputs)))
        outputs = self.conv18(self.conv17(self.conv16(self.ups3(outputs))))
        return outputs.reshape(batch_size, 256, -1)


class TrackNetBallDetector:
    """Locate a tennis ball from three consecutive frames using TrackNet."""

    def __init__(self, model_path: Path | str, *, device: str | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"TrackNet weights not found at {self.model_path}. "
                "Download them as described in the README or pass --ball-model."
            )
        self.device = torch.device(device or self._default_device())
        self.model = _TrackNet().to(self.device)
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self._frames: deque[np.ndarray] = deque(maxlen=3)

    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: float,
        excluded_boxes: Sequence[BoundingBox] = (),
    ) -> list[BallCandidate]:
        # Keep detections at racket contact; the ball often overlaps a player's box.
        del excluded_boxes
        resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        self._frames.appendleft(resized)
        if len(self._frames) < 3:
            return []

        stacked = np.concatenate(tuple(self._frames), axis=2).astype(np.float32) / 255.0
        inputs = torch.from_numpy(stacked.transpose(2, 0, 1)[None]).to(self.device)
        with torch.inference_mode():
            labels = self.model(inputs).argmax(dim=1).reshape(INPUT_HEIGHT, INPUT_WIDTH)
        heatmap = labels.to("cpu", dtype=torch.uint8).numpy()
        center = _largest_foreground_center(heatmap)
        if center is None:
            return []

        center_x, center_y, radius, peak = center
        source_height, source_width = image.shape[:2]
        scale_x = source_width / INPUT_WIDTH
        scale_y = source_height / INPUT_HEIGHT
        return [
            BallCandidate(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                confidence=max(0.01, peak / 255.0),
                pixel_x=center_x * scale_x,
                pixel_y=center_y * scale_y,
                radius=max(1.0, radius * (scale_x + scale_y) / 2),
            )
        ]

    @staticmethod
    def _default_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


def _largest_foreground_center(heatmap: np.ndarray) -> tuple[float, float, float, int] | None:
    mask = np.where(heatmap > 127, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    x, y, width, height = cv2.boundingRect(contour)
    peak = int(heatmap[y : y + height, x : x + width].max())
    return center_x, center_y, max(radius, 1.0), peak
