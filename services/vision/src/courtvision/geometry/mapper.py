"""Perspective mapping between video pixels and normalized court space."""

from __future__ import annotations

import cv2
import numpy as np

from courtvision.court.calibration import CourtCalibration


class CourtMapper:
    """Apply a validated homography to pixel or court coordinates."""

    normalized_corners = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]],
        dtype=np.float32,
    )

    def __init__(self, image_to_court: np.ndarray) -> None:
        matrix = np.asarray(image_to_court, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Homography must be a finite 3x3 matrix")
        if abs(np.linalg.det(matrix)) < 1e-10:
            raise ValueError("Homography matrix is singular")
        self.image_to_court = matrix
        self.court_to_image = np.linalg.inv(matrix)

    @classmethod
    def from_calibration(cls, calibration: CourtCalibration) -> CourtMapper:
        source = np.asarray(calibration.keypoints.ordered_points(), dtype=np.float32)
        polygon = source[[0, 1, 3, 2]]
        if abs(cv2.contourArea(polygon)) < 1.0:
            raise ValueError("Calibration points do not define a usable court area")
        matrix = cv2.getPerspectiveTransform(source, cls.normalized_corners)
        return cls(matrix)

    def transform(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        return self._transform_point(pixel_x, pixel_y, self.image_to_court)

    def inverse_transform(self, court_x: float, court_y: float) -> tuple[float, float]:
        return self._transform_point(court_x, court_y, self.court_to_image)

    @staticmethod
    def _transform_point(x: float, y: float, matrix: np.ndarray) -> tuple[float, float]:
        point = np.array([[[x, y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(point, matrix)[0, 0]
        return float(transformed[0]), float(transformed[1])
