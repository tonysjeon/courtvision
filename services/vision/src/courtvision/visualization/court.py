"""Court drawings used to validate calibration visually."""

from __future__ import annotations

import cv2
import numpy as np

from courtvision.court.calibration import CourtCalibration
from courtvision.geometry.mapper import CourtMapper

COURT_LENGTH_METERS = 23.77
SERVICE_LINE_FROM_NET_METERS = 6.40


def render_normalized_court(width: int = 600, height: int = 900) -> np.ndarray:
    """Render a normalized singles court with baselines and service boxes."""
    if width < 200 or height < 300:
        raise ValueError("Court canvas must be at least 200x300 pixels")

    canvas = np.full((height, width, 3), (54, 125, 76), dtype=np.uint8)
    margin = max(30, min(width, height) // 12)
    left, right = margin, width - margin
    top, bottom = margin, height - margin
    line_color = (255, 255, 255)
    thickness = max(2, min(width, height) // 250)

    cv2.rectangle(canvas, (left, top), (right, bottom), line_color, thickness)
    net_y = (top + bottom) // 2
    cv2.line(canvas, (left, net_y), (right, net_y), (225, 225, 225), thickness)

    normalized_service_offset = SERVICE_LINE_FROM_NET_METERS / COURT_LENGTH_METERS
    service_offset = round((bottom - top) * normalized_service_offset)
    near_service_y = net_y - service_offset
    far_service_y = net_y + service_offset
    cv2.line(canvas, (left, near_service_y), (right, near_service_y), line_color, thickness)
    cv2.line(canvas, (left, far_service_y), (right, far_service_y), line_color, thickness)
    center_x = (left + right) // 2
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
    margin = max(30, min(court.shape[1], court.shape[0]) // 12)
    for source in calibration.keypoints.ordered_points():
        court_x, court_y = mapper.transform(*source)
        px = round(margin + court_x / 100 * (court.shape[1] - 2 * margin))
        py = round(court.shape[0] - margin - court_y / 100 * (court.shape[0] - 2 * margin))
        cv2.circle(court, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)

    return np.hstack((annotated, court))
