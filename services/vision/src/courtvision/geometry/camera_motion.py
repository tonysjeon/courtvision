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
        self._reference_keypoints, self._reference_descriptors = self._features(reference_image)
        if self._reference_descriptors is None:
            raise ValueError("Could not find camera reference features")
        self._image_to_reference = np.eye(3, dtype=np.float64)
        self._has_estimate = False

    def mapper_for_frame(self, image: np.ndarray) -> CourtMapper:
        """Return a mapper corrected for the current frame's pan and zoom."""
        keypoints, descriptors = self._features(image)
        if descriptors is None:
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

        transform = transform.astype(np.float64) / transform[2, 2]
        if self._has_estimate:
            alpha = self.smoothing
            self._image_to_reference += alpha * (transform - self._image_to_reference)
        else:
            self._image_to_reference = transform
            self._has_estimate = True
        self._image_to_reference /= self._image_to_reference[2, 2]
        return self.reference_mapper.after_image_transform(self._image_to_reference)

    def _features(self, image: np.ndarray) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        return tuple(keypoints), descriptors
