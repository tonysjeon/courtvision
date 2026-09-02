"""Court drawings used to validate calibration visually."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper

COURT_LENGTH_METERS = 23.77
SINGLES_WIDTH_METERS = 8.23
DOUBLES_WIDTH_METERS = 10.97
SERVICE_LINE_FROM_NET_METERS = 6.40
DOUBLES_ALLEY_RATIO = (DOUBLES_WIDTH_METERS - SINGLES_WIDTH_METERS) / (2 * DOUBLES_WIDTH_METERS)
DOUBLES_COURT_ASPECT_RATIO = COURT_LENGTH_METERS / DOUBLES_WIDTH_METERS


@dataclass(frozen=True, slots=True)
class _CourtLayout:
    doubles_left: int
    doubles_right: int
    singles_left: int
    singles_right: int
    top: int
    bottom: int
    net_y: int
    net_left: int
    net_right: int


def _court_layout(width: int, height: int) -> _CourtLayout:
    available_width = round(width * 0.68)
    available_height = round(height * 0.76)
    doubles_width = min(
        available_width,
        round(available_height / DOUBLES_COURT_ASPECT_RATIO),
    )
    court_length = round(doubles_width * DOUBLES_COURT_ASPECT_RATIO)
    doubles_left = (width - doubles_width) // 2
    doubles_right = doubles_left + doubles_width
    top = (height - court_length) // 2
    bottom = top + court_length

    alley_width = round(doubles_width * DOUBLES_ALLEY_RATIO)
    singles_left = doubles_left + alley_width
    singles_right = doubles_right - alley_width
    net_y = (top + bottom) // 2
    net_overhang = round(doubles_width * 0.06)
    return _CourtLayout(
        doubles_left=doubles_left,
        doubles_right=doubles_right,
        singles_left=singles_left,
        singles_right=singles_right,
        top=top,
        bottom=bottom,
        net_y=net_y,
        net_left=doubles_left - net_overhang,
        net_right=doubles_right + net_overhang,
    )


def court_position_to_canvas(
    court_x: float,
    court_y: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Map normalized singles-court coordinates onto the court drawing."""
    layout = _court_layout(width, height)
    x = round(layout.singles_left + court_x / 100 * (layout.singles_right - layout.singles_left))
    y = round(layout.bottom - court_y / 100 * (layout.bottom - layout.top))
    return x, y


def render_normalized_court(width: int = 600, height: int = 900) -> np.ndarray:
    """Render singles and doubles lines with surrounding run-off space."""
    if width < 200 or height < 300:
        raise ValueError("Court canvas must be at least 200x300 pixels")

    canvas = np.full((height, width, 3), (54, 125, 76), dtype=np.uint8)
    layout = _court_layout(width, height)
    line_color = (255, 255, 255)
    thickness = max(2, min(width, height) // 250)

    cv2.rectangle(
        canvas,
        (layout.doubles_left, layout.top),
        (layout.doubles_right, layout.bottom),
        line_color,
        thickness,
    )
    cv2.line(
        canvas,
        (layout.singles_left, layout.top),
        (layout.singles_left, layout.bottom),
        line_color,
        thickness,
    )
    cv2.line(
        canvas,
        (layout.singles_right, layout.top),
        (layout.singles_right, layout.bottom),
        line_color,
        thickness,
    )
    cv2.line(
        canvas,
        (layout.net_left, layout.net_y),
        (layout.net_right, layout.net_y),
        (225, 225, 225),
        thickness + 1,
    )

    normalized_service_offset = SERVICE_LINE_FROM_NET_METERS / COURT_LENGTH_METERS
    service_offset = round((layout.bottom - layout.top) * normalized_service_offset)
    near_service_y = layout.net_y - service_offset
    far_service_y = layout.net_y + service_offset
    cv2.line(
        canvas,
        (layout.singles_left, near_service_y),
        (layout.singles_right, near_service_y),
        line_color,
        thickness,
    )
    cv2.line(
        canvas,
        (layout.singles_left, far_service_y),
        (layout.singles_right, far_service_y),
        line_color,
        thickness,
    )
    center_x = (layout.singles_left + layout.singles_right) // 2
    cv2.line(canvas, (center_x, near_service_y), (center_x, far_service_y), line_color, thickness)
    return canvas


def render_calibration_preview(
    frame: np.ndarray,
    calibration: CourtCalibration,
    mapper: CourtMapper,
) -> np.ndarray:
    """Place an annotated source frame beside its normalized court."""
    annotated = frame.copy()
    points = np.asarray(calibration.keypoints.ordered_points(), dtype=np.int32)
    polygon = points[[0, 1, 3, 2]].reshape((-1, 1, 2))
    cv2.polylines(annotated, [polygon], True, (0, 255, 255), 3, cv2.LINE_AA)

    labels = ("near left", "near right", "far left", "far right")
    for label, point in zip(labels, points, strict=True):
        cv2.circle(annotated, tuple(point), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            label,
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    court = render_normalized_court(width=max(300, frame.shape[1] // 2), height=frame.shape[0])
    for source in calibration.keypoints.ordered_points():
        court_x, court_y = mapper.transform(*source)
        px, py = court_position_to_canvas(
            court_x,
            court_y,
            width=court.shape[1],
            height=court.shape[0],
        )
        cv2.circle(court, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)

    return np.hstack((annotated, court))
