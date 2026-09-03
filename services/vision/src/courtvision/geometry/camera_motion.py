"""Compensate for broadcast camera movement relative to a calibrated frame."""

from __future__ import annotations

import cv2
import numpy as np

from courtvision.geometry.mapper import CourtMapper


class CameraMotionCompensator:
    """Align each frame to the reference image using stable visual features."""

    def __init__(
        self,
        reference_image: np.ndarray,
        reference_mapper: CourtMapper,
        *,
        max_features: int = 5000,
        match_ratio: float = 0.75,
        minimum_inliers: int = 40,
        smoothing: float = 0.35,
    ) -> None:
        if not 0 < match_ratio < 1:
            raise ValueError("match_ratio must be between 0 and 1")
        if minimum_inliers < 4:
            raise ValueError("minimum_inliers must be at least 4")
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be greater than 0 and at most 1")

        self.reference_mapper = reference_mapper
        self.match_ratio = match_ratio
        self.minimum_inliers = minimum_inliers
        self.smoothing = smoothing
        self._detector = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._court_hsv = (
            self._sample_court_color(reference_image) if reference_image.ndim == 3 else None
        )
        self._reference_court = self._court_corners(reference_image)
        self._reference_keypoints, self._reference_descriptors = self._features(reference_image)
        if self._reference_descriptors is None and self._reference_court is None:
            raise ValueError("Could not find camera reference features or court surface")
        self._image_to_reference = np.eye(3, dtype=np.float64)
        self._has_estimate = False

    def mapper_for_frame(self, image: np.ndarray) -> CourtMapper:
        """Return a mapper corrected for the current frame's pan and zoom."""
        court = self._court_corners(image)
        if court is not None and self._reference_court is not None:
            transform = cv2.getPerspectiveTransform(court, self._reference_court)
            return self._update_mapper(transform)

        keypoints, descriptors = self._features(image)
        if descriptors is None or self._reference_descriptors is None:
            return self.reference_mapper.after_image_transform(self._image_to_reference)

        pairs = self._matcher.knnMatch(descriptors, self._reference_descriptors, k=2)
        matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self.match_ratio * second.distance
        ]
        if len(matches) < self.minimum_inliers:
            return self.reference_mapper.after_image_transform(self._image_to_reference)

        current_points = np.float32([keypoints[item.queryIdx].pt for item in matches])
        reference_points = np.float32(
            [self._reference_keypoints[item.trainIdx].pt for item in matches]
        )
        transform, inlier_mask = cv2.findHomography(
            current_points,
            reference_points,
            cv2.RANSAC,
            3.0,
        )
        if (
            transform is None
            or inlier_mask is None
            or int(inlier_mask.sum()) < self.minimum_inliers
            or not np.isfinite(transform).all()
            or abs(transform[2, 2]) < 1e-10
        ):
            return self.reference_mapper.after_image_transform(self._image_to_reference)

        return self._update_mapper(transform)

    def _update_mapper(self, transform: np.ndarray) -> CourtMapper:
        transform = transform.astype(np.float64) / transform[2, 2]
        if self._has_estimate:
            alpha = self.smoothing
            self._image_to_reference += alpha * (transform - self._image_to_reference)
        else:
            self._image_to_reference = transform
            self._has_estimate = True
        self._image_to_reference /= self._image_to_reference[2, 2]
        return self.reference_mapper.after_image_transform(self._image_to_reference)

    def _sample_court_color(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        samples = []
        for court_y in (15, 30, 70, 85):
            for court_x in (15, 35, 65, 85):
                pixel_x, pixel_y = self.reference_mapper.inverse_transform(court_x, court_y)
                x = round(pixel_x)
                y = round(pixel_y)
                if 2 <= x < image.shape[1] - 2 and 2 <= y < image.shape[0] - 2:
                    samples.extend(hsv[y - 2 : y + 3, x - 2 : x + 3].reshape(-1, 3))
        if not samples:
            return np.array([0, 0, 0], dtype=np.float32)
        return np.median(np.asarray(samples), axis=0)

    def _court_corners(self, image: np.ndarray) -> np.ndarray | None:
        if image.ndim != 3 or self._court_hsv is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_delta = np.abs(hsv[:, :, 0].astype(np.float32) - self._court_hsv[0])
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        saturation_delta = np.abs(hsv[:, :, 1].astype(np.float32) - self._court_hsv[1])
        value_delta = np.abs(hsv[:, :, 2].astype(np.float32) - self._court_hsv[2])
        mask = np.where(
            (hue_delta <= 12) & (saturation_delta <= 65) & (value_delta <= 75),
            255,
            0,
        ).astype(np.uint8)
        kernel_size = max(9, round(min(image.shape[:2]) * 0.018))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * 0.08:
            return None

        points = contour.reshape(-1, 2)
        min_y = int(points[:, 1].min())
        max_y = int(points[:, 1].max())
        band = max(8, round((max_y - min_y) * 0.02))
        far = points[points[:, 1] <= min_y + band]
        near = points[points[:, 1] >= max_y - band]
        if len(far) < 2 or len(near) < 2:
            return None

        near_left = near[np.argmin(near[:, 0])]
        near_right = near[np.argmax(near[:, 0])]
        far_left = far[np.argmin(far[:, 0])]
        far_right = far[np.argmax(far[:, 0])]
        corners = np.float32([near_left, near_right, far_left, far_right])
        if far_right[0] - far_left[0] < image.shape[1] * 0.15:
            return None
        if near_right[0] - near_left[0] < image.shape[1] * 0.3:
            return None
        return corners

    def _features(self, image: np.ndarray) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        return tuple(keypoints), descriptors
